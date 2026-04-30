"""
Train a LoRA adapter on Stable Diffusion 1.5 to learn Van Gogh's painting style.

Designed for Apple Silicon (MPS) with 16GB RAM. Also supports CUDA and CPU.
Total disk footprint: ~15GB (model ~4GB, deps ~8GB, data+LoRA <1GB).
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "runwayml/stable-diffusion-v1-5"
TRIGGER_PHRASE = "a painting in the style of van gogh"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class StyleDataset(Dataset):
    def __init__(self, image_dir: str, tokenizer: CLIPTokenizer, resolution: int = 512):
        self.image_paths = sorted(
            p
            for p in Path(image_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(image)

        tokens = self.tokenizer(
            TRIGGER_PHRASE,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
        }


def encode_images(vae: AutoencoderKL, pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents


def train(args):
    device = get_device()
    frozen_dtype = torch.float16 if device.type == "cuda" else torch.float32
    unet_dtype = torch.float32

    print(f"Using device: {device} (frozen={frozen_dtype}, unet={unet_dtype})")
    print(f"Model: {MODEL_ID}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")

    # --- Load model components ---
    print("Loading tokenizer and text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=frozen_dtype
    )
    text_encoder.to(device)
    text_encoder.requires_grad_(False)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=frozen_dtype)
    vae.to(device)
    vae.requires_grad_(False)

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=unet_dtype
    )
    unet.to(device)
    unet.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # --- Inject LoRA ---
    print("Injecting LoRA layers...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in lora_params)
    print(f"Trainable LoRA parameters: {num_params:,} ({num_params * 4 / 1e6:.1f} MB in fp32)")

    # --- Dataset & DataLoader ---
    print(f"Loading dataset from {args.data_dir}...")
    dataset = StyleDataset(args.data_dir, tokenizer, resolution=args.resolution)
    print(f"Found {len(dataset)} training images")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=1e-2)

    num_epochs = math.ceil(args.max_steps / (len(dataloader) // args.gradient_accumulation))
    num_epochs = max(num_epochs, 1)

    # --- Pre-encode text (same caption for all samples) ---
    with torch.no_grad():
        sample_tokens = tokenizer(
            TRIGGER_PHRASE,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        encoder_hidden_states = text_encoder(sample_tokens)[0]  # (1, 77, 768)

    # --- Training loop ---
    print(f"\nStarting training for {args.max_steps} steps ({num_epochs} epochs)...")
    global_step = 0
    unet.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=frozen_dtype)

            latents = encode_images(vae, pixel_values).to(dtype=unet_dtype)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            batch_encoder_states = encoder_hidden_states.to(dtype=unet_dtype).expand(
                latents.shape[0], -1, -1
            )

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=batch_encoder_states,
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            loss = loss / args.gradient_accumulation
            loss.backward()

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += loss.item() * args.gradient_accumulation
                num_batches += 1

                if global_step % args.log_every == 0:
                    avg = epoch_loss / num_batches if num_batches > 0 else 0
                    print(f"  Step {global_step}/{args.max_steps} | Loss: {loss.item() * args.gradient_accumulation:.4f} | Avg: {avg:.4f}")

                if global_step % args.save_every == 0:
                    save_lora(unet, args.output_dir, f"checkpoint-{global_step}")
                    print(f"  Saved checkpoint at step {global_step}")

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} complete | Avg loss: {avg_epoch_loss:.4f}")

    # --- Save final LoRA ---
    save_lora(unet, args.output_dir, "final")
    print(f"\nTraining complete! LoRA saved to {args.output_dir}/final")
    print(f"Final LoRA size: {get_dir_size(Path(args.output_dir) / 'final'):.1f} MB")


def save_lora(unet, output_dir: str, name: str):
    save_path = Path(output_dir) / name
    save_path.mkdir(parents=True, exist_ok=True)

    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

    from diffusers import StableDiffusionPipeline

    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(save_path),
        unet_lora_layers=unet_lora_state_dict,
    )


def get_dir_size(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Train Van Gogh style LoRA on SD 1.5")
    parser.add_argument("--data-dir", type=str, default="data/van_gogh", help="Training images directory")
    parser.add_argument("--output-dir", type=str, default="output/lora", help="Where to save LoRA weights")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (1 recommended for 16GB)")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (lower = smaller adapter)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (scaling factor)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory")
    parser.add_argument("--log-every", type=int, default=25, help="Log every N steps")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--trigger-phrase", type=str, default=None, help="Caption used for all training images")
    args = parser.parse_args()

    if args.trigger_phrase:
        global TRIGGER_PHRASE
        TRIGGER_PHRASE = args.trigger_phrase

    train(args)


if __name__ == "__main__":
    main()
