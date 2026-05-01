"""Step 4 sanity: null-text inversion should round-trip a real-ish image at CFG=7.5.

We don't have real photos yet, so we generate one at CFG=7.5, decode to pixels,
and treat that PNG as the 'real' image. The null-text recon should match within
VAE-encode-decode-roundtrip slop (a few hundredths in pixel mean abs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image

from ddim import ddim_sample
from null_text_inv import null_text_inversion, sample_with_null
from sd_components import decode_latents, encode_prompt, load_sd

PROMPT = "a photograph of a cat sitting on a couch"
SEED = 0
STEPS = 50
GUIDANCE = 7.5

c = load_sd()
cond = encode_prompt(c, PROMPT)
uncond = encode_prompt(c, [""])

g = torch.Generator(device=c.device).manual_seed(SEED)
z_T_gen = torch.randn(
    (1, c.unet.config.in_channels, 64, 64),
    generator=g, device=c.device, dtype=c.dtype,
)
z_0_gen = ddim_sample(
    c, cond, uncond,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    latents=z_T_gen.clone(),
)
img_orig_t = decode_latents(c, z_0_gen).clamp(-1, 1)


def to_pil(t):
    arr = ((t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() + 1) / 2 * 255)
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


img_orig = to_pil(img_orig_t)

print("Running null-text inversion (~1-2 min on A100)...")
result = null_text_inversion(
    c, img_orig, PROMPT,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    inner_steps=10,
)

z_0_recon = sample_with_null(
    c, cond, result.null_embeds, result.z_T,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
)
img_recon_t = decode_latents(c, z_0_recon).clamp(-1, 1)

diff = (img_orig_t - img_recon_t).abs()
print(f"pixels (recon vs orig)  mean={diff.mean().item():.6f}  max={diff.max().item():.6f}")

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "sanity_null_text"
out_dir.mkdir(parents=True, exist_ok=True)
img_orig.save(out_dir / "original.png")
to_pil(img_recon_t).save(out_dir / "reconstructed.png")
print(f"saved to {out_dir}")
