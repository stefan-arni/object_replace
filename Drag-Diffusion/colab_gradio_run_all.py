"""
Self-contained Colab runner with an in-notebook Gradio UI.

How to use in Colab:
1. Open a new notebook.
2. Runtime -> Change runtime type -> GPU.
3. Paste this entire file into one code cell.
4. Run the cell.
5. Use the Gradio UI to:
   - upload an image
   - paint the source mask
   - paint the target mask
   - enter a prompt
   - click Run
"""

import os
import sys
import math
import subprocess
from dataclasses import dataclass
from typing import Dict


def install_dependencies():
    packages = [
        "torch",
        "torchvision",
        "diffusers>=0.27.0",
        "transformers>=4.39.0",
        "accelerate",
        "safetensors>=0.4.0",
        "Pillow",
        "numpy",
        "matplotlib",
        "gradio>=4.0",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


install_dependencies()

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
import gradio as gr


MODEL_ID = "sd2-community/stable-diffusion-2-1-base"
INPAINT_MODEL_ID = "sd2-community/stable-diffusion-2-inpainting"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=torch.float32)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).float().clamp(-1, 1)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr)


def encode_image(vae, pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latent = vae.encode(pixel_values).latent_dist.sample() * 0.18215
    return latent.float()


def decode_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        decoded = vae.decode(latent / 0.18215).sample
    return decoded.float()


def create_composite(image: Image.Image, source_mask: Image.Image, target_mask: Image.Image) -> Image.Image:
    img = np.array(image.convert("RGB")).astype(np.float32)
    src = np.array(source_mask.convert("L")) > 127
    tgt = np.array(target_mask.convert("L")) > 127

    h, w = img.shape[:2]
    sy, sx = np.where(src)
    ty, tx = np.where(tgt)

    if len(sy) == 0 or len(ty) == 0:
        return image.copy()

    dy = int(round(ty.mean() - sy.mean()))
    dx = int(round(tx.mean() - sx.mean()))
    composite = img.copy()

    y_min, y_max = sy.min(), sy.max()
    x_min, x_max = sx.min(), sx.max()
    box_h = int(y_max - y_min + 1)
    box_w = int(x_max - x_min + 1)

    patch_pixels = None
    for dy_p, dx_p in [(0, box_w), (0, -box_w), (box_h, 0), (-box_h, 0)]:
        new_sy = sy + dy_p
        new_sx = sx + dx_p
        if (
            new_sy.min() >= 0
            and new_sy.max() < h
            and new_sx.min() >= 0
            and new_sx.max() < w
            and src[new_sy, new_sx].sum() == 0
        ):
            patch_pixels = img[new_sy, new_sx]
            break

    if patch_pixels is not None:
        composite[src] = patch_pixels
    else:
        bg_color = img[~src].mean(axis=0) if (~src).any() else np.array([128.0, 128.0, 128.0])
        composite[src] = bg_color

    new_ys = np.clip(sy + dy, 0, h - 1)
    new_xs = np.clip(sx + dx, 0, w - 1)
    composite[new_ys, new_xs] = img[sy, sx]
    return Image.fromarray(composite.clip(0, 255).astype(np.uint8))


def prepare_latent_mask(mask_pil: Image.Image, device: torch.device, latent_size: int) -> torch.Tensor:
    arr = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = (t > 0.5).float()
    t = F.interpolate(t, size=(latent_size, latent_size), mode="nearest")
    return t.to(device=device, dtype=torch.float32)


def compute_centroid(mask: torch.Tensor, device: torch.device):
    mask = mask.squeeze()
    h, w = mask.shape
    if mask.sum() < 1e-6:
        return h / 2.0, w / 2.0
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    total = mask.sum()
    cy = (mask * ys.unsqueeze(1)).sum() / total
    cx = (mask * xs.unsqueeze(0)).sum() / total
    return cy.item(), cx.item()


def gaussian_blur_mask(mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    kernel_size = int(6 * sigma + 1) | 1
    coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    blurred = F.conv2d(mask, kernel, padding=pad)
    return blurred.clamp(0.0, 1.0)


def shift_noise_map(
    eps_t: torch.Tensor,
    m_src: torch.Tensor,
    m_tgt: torch.Tensor,
    device: torch.device,
    feather_sigma: float = 2.0,
) -> torch.Tensor:
    _, _, h, w = eps_t.shape
    cy_s, cx_s = compute_centroid(m_src, device)
    cy_t, cx_t = compute_centroid(m_tgt, device)
    dy = cy_t - cy_s
    dx = cx_t - cx_s
    dx_norm = -2.0 * dx / max(w - 1, 1)
    dy_norm = -2.0 * dy / max(h - 1, 1)
    theta = torch.tensor(
        [[1.0, 0.0, dx_norm], [0.0, 1.0, dy_norm]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 4, h, w), align_corners=False)
    m_src_soft = gaussian_blur_mask(m_src, sigma=feather_sigma) if feather_sigma > 0 else m_src
    eps_src_region = eps_t * m_src_soft
    eps_src_shifted = F.grid_sample(
        eps_src_region, grid, align_corners=False, padding_mode="zeros", mode="bilinear"
    )
    return eps_t * (1.0 - m_src_soft) + eps_src_shifted


def shift_all_noise_maps(
    noise_maps: dict,
    m_src: torch.Tensor,
    m_tgt: torch.Tensor,
    device: torch.device,
    feather_sigma: float = 2.0,
) -> dict:
    return {
        t: shift_noise_map(eps, m_src, m_tgt, device, feather_sigma)
        for t, eps in noise_maps.items()
    }


@dataclass
class DDPMInversionTrajectory:
    latents: Dict[int, torch.Tensor]
    marginal_noises: Dict[int, torch.Tensor]
    transition_noises: Dict[int, torch.Tensor]


def _make_generator(device: torch.device, seed: int):
    gen_device = device if device.type in {"cuda", "mps"} else torch.device("cpu")
    try:
        generator = torch.Generator(device=gen_device).manual_seed(seed)
    except RuntimeError:
        gen_device = torch.device("cpu")
        generator = torch.Generator(device=gen_device).manual_seed(seed)
    return generator, gen_device


def _randn_like(x: torch.Tensor, generator: torch.Generator, sample_device: torch.device) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=generator, device=sample_device, dtype=torch.float32)
    if sample_device != x.device:
        noise = noise.to(x.device)
    return noise


def ddpm_invert(
    x0_latent: torch.Tensor,
    scheduler: DDPMScheduler,
    num_inference_steps: int = 50,
    seed: int = 42,
    device: torch.device = None,
) -> DDPMInversionTrajectory:
    if device is None:
        device = x0_latent.device
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    generator, sample_device = _make_generator(device, seed)

    latents = {}
    marginal_noises = {}
    transition_noises = {}

    first_t = timesteps[0].item()
    abar_first = scheduler.alphas_cumprod[first_t].to(device=device, dtype=torch.float32)
    eps_first = _randn_like(x0_latent, generator, sample_device)
    x_t = abar_first.sqrt() * x0_latent + (1 - abar_first).sqrt() * eps_first

    latents[first_t] = x_t
    marginal_noises[first_t] = eps_first

    for idx, t in enumerate(timesteps[:-1]):
        t_int = t.item()
        t_prev_int = timesteps[idx + 1].item()
        abar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=torch.float32)
        abar_prev = scheduler.alphas_cumprod[t_prev_int].to(device=device, dtype=torch.float32)

        alpha_ratio = abar_t / abar_prev
        beta_t = (1 - alpha_ratio).clamp(min=0.0)
        coeff1 = abar_prev.sqrt() * beta_t / (1 - abar_t)
        coeff2 = alpha_ratio.sqrt() * (1 - abar_prev) / (1 - abar_t)
        mu = coeff1 * x0_latent + coeff2 * x_t

        sigma_t = (beta_t * (1 - abar_prev) / (1 - abar_t)).clamp(min=0.0).sqrt()
        z_t = _randn_like(x0_latent, generator, sample_device)
        x_prev = mu + sigma_t * z_t

        latents[t_prev_int] = x_prev
        transition_noises[t_prev_int] = z_t
        marginal_scale = (1 - abar_prev).sqrt().clamp(min=1e-8)
        marginal_noises[t_prev_int] = (x_prev - abar_prev.sqrt() * x0_latent) / marginal_scale
        x_t = x_prev

    return DDPMInversionTrajectory(
        latents=latents,
        marginal_noises=marginal_noises,
        transition_noises=transition_noises,
    )


def reconstruct_xt(
    x0_latent: torch.Tensor,
    t_int: int,
    eps_t: torch.Tensor,
    scheduler: DDPMScheduler,
    device: torch.device,
) -> torch.Tensor:
    abar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=torch.float32)
    return abar_t.sqrt() * x0_latent + (1 - abar_t).sqrt() * eps_t


class ObjectRelocationPipeline:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        inpaint_model_id: str = INPAINT_MODEL_ID,
        device: torch.device = None,
        local_files_only: bool = False,
    ):
        if device is None:
            device = get_device()
        self.device = device
        self.model_id = model_id
        self.inpaint_model_id = inpaint_model_id
        self.local_files_only = local_files_only
        self.dtype = get_dtype(device)

        print(f"Loading base model on {device} ({self.dtype})...")
        kw = dict(local_files_only=local_files_only)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **kw).to(device=device, dtype=self.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **kw)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **kw).to(device=device, dtype=self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **kw).to(device=device, dtype=self.dtype)
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", **kw)
        self.prediction_type = self.scheduler.config.prediction_type
        native_size = self.unet.config.sample_size * 8
        self.image_size = 512 if device.type == "mps" else native_size
        self.latent_size = self.image_size // 8
        self.inpaint_pipe = None
        print(f"Base model ready. Prediction type: {self.prediction_type}, image size: {self.image_size}")

    def _make_generator(self, seed: int) -> torch.Generator:
        gen_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        return torch.Generator(device=gen_device).manual_seed(seed)

    def _load_inpaint_pipe(self):
        if self.inpaint_pipe is not None:
            return self.inpaint_pipe
        print(f"Loading inpainting model on {self.device} ({self.dtype})...")
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.inpaint_model_id,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self.inpaint_pipe.set_progress_bar_config(disable=True)
        if hasattr(self.inpaint_pipe, "enable_attention_slicing"):
            self.inpaint_pipe.enable_attention_slicing()
        return self.inpaint_pipe

    def _masks_are_effectively_same(self, source_mask: Image.Image, target_mask: Image.Image) -> bool:
        src = np.array(source_mask.convert("L")) > 127
        tgt = np.array(target_mask.convert("L")) > 127
        union = np.logical_or(src, tgt).sum()
        if union == 0:
            return True
        iou = np.logical_and(src, tgt).sum() / union
        return iou > 0.98

    def _restore_background(
        self,
        original: Image.Image,
        edited: Image.Image,
        source_mask: Image.Image,
        target_mask: Image.Image,
        feather_sigma: float,
    ) -> Image.Image:
        orig_arr = np.array(original.convert("RGB")).astype(np.float32)
        edited_arr = np.array(edited.convert("RGB")).astype(np.float32)
        blur_r = max(1, int(feather_sigma * 6))
        src_soft = np.array(
            source_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_r))
        ).astype(np.float32) / 255.0
        tgt_soft = np.array(
            target_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_r))
        ).astype(np.float32) / 255.0
        keep_orig = ((1.0 - src_soft) * (1.0 - tgt_soft))[..., None]
        merged = edited_arr * (1.0 - keep_orig) + orig_arr * keep_orig
        return Image.fromarray(merged.clip(0, 255).astype(np.uint8))

    def _cleanup_source_with_inpainting(
        self,
        image: Image.Image,
        prompt: str,
        source_mask: Image.Image,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Image.Image:
        pipe = self._load_inpaint_pipe()
        generator = self._make_generator(seed)
        mask = source_mask.convert("L").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        result = pipe(
            prompt=prompt,
            negative_prompt="",
            image=image,
            mask_image=mask,
            num_inference_steps=max(4, num_inference_steps),
            guidance_scale=guidance_scale,
            strength=0.35,
            generator=generator,
        ).images[0]
        return result

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0].float()

    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev_int: int,
        eps_pred: torch.Tensor,
        stored_noise: torch.Tensor,
    ) -> torch.Tensor:
        t_int = t.item()
        abar_t = self.scheduler.alphas_cumprod[t_int].to(device=self.device, dtype=torch.float32)
        if self.prediction_type == "v_prediction":
            eps_pred = abar_t.sqrt() * eps_pred + (1 - abar_t).sqrt() * x_t
        pred_x0 = (x_t - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
        pred_x0 = pred_x0.clamp(-4.0, 4.0)
        if t_prev_int < 0:
            return pred_x0
        abar_prev = self.scheduler.alphas_cumprod[t_prev_int].to(device=self.device, dtype=torch.float32)
        coeff1 = abar_prev.sqrt() * (1 - abar_t / abar_prev) / (1 - abar_t)
        coeff2 = (abar_t / abar_prev).sqrt() * (1 - abar_prev) / (1 - abar_t)
        mu = coeff1 * pred_x0 + coeff2 * x_t
        beta_t = (1 - abar_t / abar_prev).clamp(min=0.0)
        sigma_t = (beta_t * (1 - abar_prev) / (1 - abar_t)).clamp(min=0.0).sqrt()
        return mu + sigma_t * stored_noise

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        source_mask: Image.Image,
        target_mask: Image.Image,
        use_noise_shift: bool = True,
        seed: int = 42,
        num_inference_steps: int = 4,
        sdedit_strength: float = 0.35,
        guidance_scale: float = 5.0,
        feather_sigma: float = 2.0,
    ):
        device = self.device
        sz = self.image_size

        if self._masks_are_effectively_same(source_mask, target_mask):
            identity = image.convert("RGB").resize((self.image_size, self.image_size))
            return identity.copy(), identity

        img_sz = image.resize((sz, sz))
        source_mask = source_mask.resize((sz, sz))
        target_mask = target_mask.resize((sz, sz))
        composite = create_composite(img_sz, source_mask, target_mask)

        vae_dtype = next(self.vae.parameters()).dtype
        x0_orig = encode_image(self.vae, pil_to_tensor(img_sz, device).to(vae_dtype))
        x0_composite = encode_image(self.vae, pil_to_tensor(composite, device).to(vae_dtype))

        encoder_hs = self._encode_prompt(prompt)
        uncond_hs = self._encode_prompt("")

        m_src = prepare_latent_mask(source_mask, device, self.latent_size)
        m_tgt = prepare_latent_mask(target_mask, device, self.latent_size)
        m_src_soft = gaussian_blur_mask(m_src, sigma=2.0)
        m_tgt_soft = gaussian_blur_mask(m_tgt, sigma=2.0)
        bg_mask = (1.0 - m_src_soft.clamp(0, 1)) * (1.0 - m_tgt_soft.clamp(0, 1))
        m_tgt_lock = gaussian_blur_mask(m_tgt, sigma=2.0)

        inversion = ddpm_invert(x0_orig, self.scheduler, num_inference_steps, seed, device)
        if use_noise_shift:
            start_noises = shift_all_noise_maps(inversion.marginal_noises, m_src, m_tgt, device, feather_sigma)
            transition_noises = shift_all_noise_maps(inversion.transition_noises, m_src, m_tgt, device, feather_sigma)
        else:
            start_noises = inversion.marginal_noises
            transition_noises = inversion.transition_noises

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        start_idx = max(0, min(int((1.0 - sdedit_strength) * len(timesteps)), len(timesteps) - 1))
        t_start_int = timesteps[start_idx].item()
        abar_start = self.scheduler.alphas_cumprod[t_start_int].to(device=device, dtype=torch.float32)
        x_t = abar_start.sqrt() * x0_composite + (1 - abar_start).sqrt() * start_noises[t_start_int]

        unet_dtype = next(self.unet.parameters()).dtype
        active_timesteps = timesteps[start_idx:]
        lock_cutoff = len(active_timesteps) // 2

        for i, t in enumerate(active_timesteps):
            t_global_idx = start_idx + i
            t_prev_int = timesteps[t_global_idx + 1].item() if t_global_idx + 1 < len(timesteps) else -1
            with torch.no_grad():
                x_t_input = x_t.to(unet_dtype)
                t_batch = t.unsqueeze(0).to(device)
                latent_batch = torch.cat([x_t_input, x_t_input], dim=0)
                cond_batch = torch.cat([uncond_hs, encoder_hs], dim=0).to(unet_dtype)
                noise_pred = self.unet(latent_batch, t_batch.repeat(2), cond_batch).sample.float()
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                eps_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            stored = transition_noises.get(t_prev_int, torch.zeros_like(x_t))
            x_t = self._ddpm_step(x_t, t, t_prev_int, eps_pred, stored)

            if t_prev_int >= 0:
                x_t_orig = inversion.latents[t_prev_int]
                x_t = x_t * (1.0 - bg_mask) + x_t_orig * bg_mask
                if i < lock_cutoff:
                    x_t_comp = reconstruct_xt(
                        x0_composite,
                        t_prev_int,
                        start_noises[t_prev_int],
                        self.scheduler,
                        device,
                    )
                    x_t = x_t * (1.0 - m_tgt_lock) + x_t_comp * m_tgt_lock

        first_pass = tensor_to_pil(decode_latent(self.vae, x_t.to(vae_dtype)))
        cleaned = self._cleanup_source_with_inpainting(
            first_pass,
            prompt,
            source_mask,
            seed + 1,
            num_inference_steps,
            guidance_scale,
        )
        final = self._restore_background(img_sz, cleaned, source_mask, target_mask, feather_sigma)
        return final, composite


device = get_device()
print("Device:", device)
pipe = ObjectRelocationPipeline(device=device, local_files_only=False)


def extract_mask(editor_value, fallback_size=(512, 512)):
    if editor_value is None:
        return Image.new("L", fallback_size, 0)
    layers = editor_value.get("layers") or []
    if not layers or layers[0] is None:
        return Image.new("L", fallback_size, 0)
    layer = layers[0]
    if layer.mode == "RGBA":
        return layer.split()[3]
    return layer.convert("L")


def set_image_on_editors(image):
    if image is None:
        return gr.update(), gr.update()
    blank = {"background": image, "layers": [], "composite": image}
    return blank, blank


def run_pipeline(image, src_editor, tgt_editor, prompt, use_noise_shift, seed, steps, strength, cfg):
    if image is None:
        return None, None, "Upload an image first."
    if not prompt.strip():
        return None, None, "Enter a prompt."

    size = image.size
    source_mask = extract_mask(src_editor, size)
    target_mask = extract_mask(tgt_editor, size)

    if np.array(source_mask).sum() == 0:
        return None, None, "Draw a source mask."
    if np.array(target_mask).sum() == 0:
        return None, None, "Draw a target mask."

    result, composite = pipe(
        image,
        prompt,
        source_mask,
        target_mask,
        use_noise_shift=use_noise_shift,
        seed=int(seed),
        num_inference_steps=int(steps),
        sdedit_strength=float(strength),
        guidance_scale=float(cfg),
    )
    return composite, result, "Done."


with gr.Blocks(title="Drag Diffusion Colab UI") as demo:
    gr.Markdown(
        "## Drag Diffusion\n"
        "Upload an image, paint the source mask, paint the target mask, enter a prompt, and click Run."
    )

    with gr.Row():
        with gr.Column(scale=3):
            image_input = gr.Image(label="Upload image", type="pil", height=320)
            with gr.Row():
                src_editor = gr.ImageEditor(
                    label="Paint SOURCE mask",
                    type="pil",
                    layers=True,
                    height=320,
                )
                tgt_editor = gr.ImageEditor(
                    label="Paint TARGET mask",
                    type="pil",
                    layers=True,
                    height=320,
                )
            image_input.change(
                fn=set_image_on_editors,
                inputs=image_input,
                outputs=[src_editor, tgt_editor],
            )

        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt",
                value="a photo of the same scene with the selected object moved to the target location",
                lines=3,
            )
            use_shift = gr.Checkbox(label="Use DDPM noise shift", value=True)
            seed_input = gr.Number(label="Seed", value=42, precision=0)
            steps_slider = gr.Slider(4, 20, value=4, step=1, label="Inference steps")
            strength_slider = gr.Slider(0.1, 0.7, value=0.35, step=0.05, label="SDEdit strength")
            cfg_slider = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance scale")
            run_btn = gr.Button("Run", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        composite_out = gr.Image(label="Composite", type="pil")
        result_out = gr.Image(label="Result", type="pil")

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            image_input,
            src_editor,
            tgt_editor,
            prompt_input,
            use_shift,
            seed_input,
            steps_slider,
            strength_slider,
            cfg_slider,
        ],
        outputs=[composite_out, result_out, status],
    )


demo.launch(share=True, debug=False)
