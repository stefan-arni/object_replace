"""Step 3 sanity: generate -> invert -> re-denoise. Should round-trip cleanly.

CFG is held at 1.0 throughout. With CFG > 1 the naive inversion drifts; that's
what null-text inversion (Step 4) fixes. If this script doesn't round-trip at
CFG=1, the bug is in the math, not in inversion-on-real-images difficulty.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image

from ddim import ddim_invert, ddim_sample
from sd_components import decode_latents, encode_prompt, load_sd

PROMPT = "a photograph of a cat sitting on a couch"
SEED = 0
STEPS = 50
GUIDANCE = 1.0

c = load_sd()
cond = encode_prompt(c, PROMPT)
uncond = encode_prompt(c, [""])

# 1. Generate from a fixed seed.
g = torch.Generator(device=c.device).manual_seed(SEED)
z_T_orig = torch.randn(
    (1, c.unet.config.in_channels, 64, 64),
    generator=g, device=c.device, dtype=c.dtype,
)
z_0 = ddim_sample(
    c, cond, uncond,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    latents=z_T_orig.clone(),
)

# 2. Invert clean latents back toward noise.
z_T_inv = ddim_invert(
    c, z_0, cond, uncond,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
)

# 3. Re-denoise from the recovered noise.
z_0_recon = ddim_sample(
    c, cond, uncond,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    latents=z_T_inv,
)


def stats(name, a, b):
    diff = (a - b).abs()
    print(f"{name:40s} mean={diff.mean().item():.6f}  max={diff.max().item():.6f}")


stats("noise (z_T_inv vs z_T_orig)", z_T_inv, z_T_orig)
stats("latents (z_0_recon vs z_0)", z_0_recon, z_0)

img_orig = decode_latents(c, z_0).clamp(-1, 1)
img_recon = decode_latents(c, z_0_recon).clamp(-1, 1)
stats("pixels (recon vs orig)", img_orig, img_recon)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "sanity_invert"
out_dir.mkdir(parents=True, exist_ok=True)


def to_pil(t):
    arr = ((t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() + 1) / 2 * 255)
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


to_pil(img_orig).save(out_dir / "original.png")
to_pil(img_recon).save(out_dir / "reconstructed.png")
print(f"saved to {out_dir}")
