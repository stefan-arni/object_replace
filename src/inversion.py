"""Null-text inversion (Mokady et al. 2022).

For real images at CFG > 1, naive DDIM inversion drifts. Null-text fixes this
by optimizing the *unconditional* embedding at each timestep so that the
CFG-guided trajectory at sampling time hits the same latents that naive
inversion produced at CFG=1. The text embedding and the latents are not touched.
"""
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ddim import _alpha_bar
from sd_components import SDComponents, encode_image, encode_prompt

CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "null_text_cache"


@dataclass
class NullTextResult:
    z_T: torch.Tensor                # 1 x 4 x 64 x 64, on CPU
    null_embeds: list[torch.Tensor]  # one per timestep, ordered by descending t, on CPU
    prompt: str
    num_inference_steps: int
    guidance_scale: float


def _preprocess(image_pil: Image.Image, size: int = 512) -> Image.Image:
    return image_pil.convert("RGB").resize((size, size), Image.BICUBIC)


def image_to_latent(c: SDComponents, image_pil: Image.Image, size: int = 512) -> torch.Tensor:
    arr = np.asarray(_preprocess(image_pil, size)).astype("float32") / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return encode_image(c, t)


def _cache_key(image_pil, prompt, num_inference_steps, guidance_scale, size=512) -> str:
    h = hashlib.sha256()
    h.update(_preprocess(image_pil, size).tobytes())
    h.update(prompt.encode())
    h.update(f"{num_inference_steps}|{guidance_scale}".encode())
    return h.hexdigest()[:16]


@torch.no_grad()
def _ddim_invert_trajectory(c: SDComponents, z_0: torch.Tensor, cond: torch.Tensor, num_inference_steps: int) -> list[torch.Tensor]:
    """Naive DDIM inversion at CFG=1. Returns [z_0, z_1, ..., z_T] (length S+1)."""
    c.scheduler.set_timesteps(num_inference_steps, device=c.device)
    timesteps_asc = list(reversed(c.scheduler.timesteps.tolist()))
    final_alpha_bar = c.scheduler.final_alpha_cumprod.to(c.device)
    stride = c.scheduler.config.num_train_timesteps // num_inference_steps

    traj = [z_0.clone()]
    latents = z_0.clone()
    for t in timesteps_asc:
        t_prev = t - stride
        a_cur = _alpha_bar(c, t_prev) if t_prev >= 0 else final_alpha_bar
        a_next = _alpha_bar(c, t)
        eps = c.unet(latents, t, encoder_hidden_states=cond).sample
        x0 = (latents - (1 - a_cur).sqrt() * eps) / a_cur.sqrt()
        latents = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        traj.append(latents.clone())
    return traj


def null_text_inversion(
    c: SDComponents,
    image_pil: Image.Image,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    inner_steps: int = 10,
    inner_lr: float = 1e-2,
    early_stop_eps: float = 1e-5,
    use_cache: bool = True,
    verbose: bool = True,
) -> NullTextResult:
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{_cache_key(image_pil, prompt, num_inference_steps, guidance_scale)}.pkl"
        if cache_path.exists():
            if verbose:
                print(f"loaded cached null-text result from {cache_path.name}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    # Freeze UNet so backward through it doesn't allocate grads on params we won't update.
    for p in c.unet.parameters():
        p.requires_grad_(False)

    cond = encode_prompt(c, prompt)
    init_uncond = encode_prompt(c, [""])

    z_0 = image_to_latent(c, image_pil)
    traj = _ddim_invert_trajectory(c, z_0, cond, num_inference_steps)
    z_T = traj[-1]

    c.scheduler.set_timesteps(num_inference_steps, device=c.device)
    timesteps_desc = c.scheduler.timesteps
    final_alpha_bar = c.scheduler.final_alpha_cumprod.to(c.device)
    S = num_inference_steps

    null_embeds: list[torch.Tensor] = []
    # Optimize null in fp32 even though the UNet runs fp16 -- Adam is unstable in fp16.
    null_t = init_uncond.float().clone().detach()
    z_t = z_T.clone()

    for i, t in enumerate(timesteps_desc):
        a_t = _alpha_bar(c, t)
        a_prev = _alpha_bar(c, timesteps_desc[i + 1]) if i + 1 < S else final_alpha_bar
        target = traj[S - i - 1]

        # cond eps doesn't depend on null, compute it once outside the grad-tracked loop.
        with torch.no_grad():
            eps_c = c.unet(z_t, t, encoder_hidden_states=cond).sample

        null_t = null_t.detach().requires_grad_(True)
        # LR decay from the null-text paper: lets the harder later timesteps take smaller steps.
        opt = torch.optim.Adam([null_t], lr=inner_lr * (1.0 - i / 100.0))

        last_loss = None
        for _ in range(inner_steps):
            opt.zero_grad()
            eps_u = c.unet(z_t, t, encoder_hidden_states=null_t.to(c.dtype)).sample
            eps = eps_u + guidance_scale * (eps_c - eps_u)

            x0 = (z_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            z_pred = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

            loss = ((z_pred.float() - target.float()) ** 2).mean()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            if last_loss < early_stop_eps + i * 2e-5:
                break

        null_embeds.append(null_t.detach().to(c.dtype).cpu().clone())

        # Apply the actual step with the optimized null so error doesn't accumulate.
        with torch.no_grad():
            eps_u = c.unet(z_t, t, encoder_hidden_states=null_t.detach().to(c.dtype)).sample
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            x0 = (z_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            z_t = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

        if verbose and (i % 10 == 0 or i == S - 1):
            print(f"  step {i+1:>2}/{S} t={int(t):>4} loss={last_loss:.2e}")

    result = NullTextResult(
        z_T=z_T.detach().cpu(),
        null_embeds=null_embeds,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
    return result


@torch.no_grad()
def sample_with_null(
    c: SDComponents,
    cond_embeds: torch.Tensor,
    null_embeds: list[torch.Tensor],
    z_T: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    """DDIM sampling using per-timestep null embeddings (the inversion result)."""
    c.scheduler.set_timesteps(num_inference_steps, device=c.device)
    timesteps = c.scheduler.timesteps
    final_alpha_bar = c.scheduler.final_alpha_cumprod.to(c.device)

    latents = z_T.to(c.device, c.dtype).clone()
    for i, t in enumerate(timesteps):
        null_t = null_embeds[i].to(c.device, c.dtype)
        embed = torch.cat([null_t, cond_embeds], dim=0)
        x_in = torch.cat([latents, latents], dim=0)
        eps_u, eps_c = c.unet(x_in, t, encoder_hidden_states=embed).sample.chunk(2)
        eps = eps_u + guidance_scale * (eps_c - eps_u)

        a_t = _alpha_bar(c, t)
        a_prev = _alpha_bar(c, timesteps[i + 1]) if i + 1 < len(timesteps) else final_alpha_bar
        x0 = (latents - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        latents = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

    return latents
