from dataclasses import dataclass
import warnings

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# runwayml/stable-diffusion-v1-5 was pulled from HF; this is the community mirror.
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_SCALE = 0.18215


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        warnings.warn("Running on MPS; expect ~10x slowdown vs A100. Mac is for editing, not benchmarks.")
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    # MPS is flaky with fp16 on attention ops; keep fp32 there.
    return torch.float16 if device.type == "cuda" else torch.float32


@dataclass
class SDComponents:
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    scheduler: DDIMScheduler
    device: torch.device
    dtype: torch.dtype


def load_sd(model_id: str = MODEL_ID) -> SDComponents:
    device = get_device()
    dtype = get_dtype(device)

    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return SDComponents(unet, vae, tokenizer, text_encoder, scheduler, device, dtype)


@torch.no_grad()
def encode_prompt(c: SDComponents, prompt: str | list[str]) -> torch.Tensor:
    prompts = [prompt] if isinstance(prompt, str) else prompt
    ids = c.tokenizer(
        prompts,
        padding="max_length",
        max_length=c.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(c.device)
    return c.text_encoder(ids).last_hidden_state


@torch.no_grad()
def encode_image(c: SDComponents, image: torch.Tensor) -> torch.Tensor:
    # image in [-1, 1], shape B x 3 x H x W
    latents = c.vae.encode(image.to(c.device, c.dtype)).latent_dist.mean
    return latents * VAE_SCALE


@torch.no_grad()
def decode_latents(c: SDComponents, latents: torch.Tensor) -> torch.Tensor:
    return c.vae.decode(latents / VAE_SCALE).sample
