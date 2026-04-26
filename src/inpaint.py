"""Blended-diffusion inpainting using the same SD components we already have.

For shape-changing edits (firetruck -> mouse): the source-shape mask covers more
area than the target needs. The leftover region (source_mask - target_mask) shows
source pixels (still firetruck), which is wrong. This module inpaints that
leftover region with a background prompt so the gap fills in plausibly.

The technique (Avrahami et al. 2022, "Blended Latent Diffusion"):
  - Encode the source image to z_0_src.
  - Start from full noise z_T.
  - At each timestep:
      1. Standard CFG denoising step on z (with background prompt) -> z_denoised.
      2. Compute z_src_noised = sqrt(a_t) * z_0_src + sqrt(1-a_t) * eps  (source
         re-noised to current timestep level).
      3. Blend: z = mask * z_denoised + (1 - mask) * z_src_noised.
  - Outside the mask stays anchored to source at the right noise level so the
    inpainted result blends in seamlessly.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ddim import _alpha_bar
from sd_components import SDComponents, decode_latents, encode_image, encode_prompt


def _preprocess_image(image_pil: Image.Image, size: int = 512) -> torch.Tensor:
    img = image_pil.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def blended_inpaint(
    c: SDComponents,
    source_image: Image.Image,
    mask: torch.Tensor,           # (1,1,64,64) or (H,W); float in [0,1]; 1 = inpaint here
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int | None = 0,
) -> torch.Tensor:
    """Inpaint the masked region of source_image with `prompt`. Outside the mask
    stays close to the source. Returns pixel tensor in [-1, 1], shape (1, 3, 512, 512).
    """
    z_0_src = encode_image(c, _preprocess_image(source_image).to(c.device, c.dtype))

    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    mask_64 = F.interpolate(mask.float(), size=(64, 64), mode="bilinear", align_corners=False).to(
        c.device, c.dtype
    )

    cond = encode_prompt(c, prompt)
    uncond = encode_prompt(c, [""])
    embed_pair = torch.cat([uncond, cond], dim=0)

    g = torch.Generator(device=c.device).manual_seed(seed) if seed is not None else None
    z = torch.randn(z_0_src.shape, generator=g, device=c.device, dtype=c.dtype)

    c.scheduler.set_timesteps(num_inference_steps, device=c.device)
    timesteps = c.scheduler.timesteps
    final_alpha_bar = c.scheduler.final_alpha_cumprod.to(c.device)

    for i, t in enumerate(timesteps):
        x_in = torch.cat([z, z], dim=0)
        eps_u, eps_c = c.unet(x_in, t, encoder_hidden_states=embed_pair).sample.chunk(2)
        eps = eps_u + guidance_scale * (eps_c - eps_u)

        a_t = _alpha_bar(c, t)
        a_prev = _alpha_bar(c, timesteps[i + 1]) if i + 1 < len(timesteps) else final_alpha_bar
        x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        z_denoised = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

        # Re-noise the source to the same noise level as z_denoised, then blend.
        # Outside the mask: keep the source's noisy state -> denoising is a no-op there.
        # Inside the mask: keep the freely-denoised state -> the inpainted content develops.
        if i + 1 < len(timesteps):
            noise = torch.randn_like(z_0_src)
            z_src_noised = a_prev.sqrt() * z_0_src + (1 - a_prev).sqrt() * noise
        else:
            z_src_noised = z_0_src
        z = mask_64 * z_denoised + (1 - mask_64) * z_src_noised

    return decode_latents(c, z).clamp(-1, 1)


def derive_background_prompt(source_prompt: str, source_word: str) -> str:
    """Strip the source object from the source prompt to get a 'scene only' prompt.

    Heuristic: remove 'a/an/the {word}' and clean up dangling 'of'/'with'/'on'
    fragments. Falls back to a generic if the result is too short.
    """
    import re
    bg = source_prompt
    bg = re.sub(rf"\b(?:a|an|the)?\s*{re.escape(source_word)}s?\b\s*", "", bg, flags=re.IGNORECASE)
    bg = re.sub(r"\bof\s+(on|in|at|with|next|near)\b", r"\1", bg, flags=re.IGNORECASE)
    bg = re.sub(r"\s+", " ", bg).strip().rstrip(",.")
    return bg if len(bg.split()) >= 3 else "an empty scene"
