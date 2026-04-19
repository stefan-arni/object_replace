"""Step 8 smoke: cat -> dog edit, with and without the attention mask.

The pillow that disappears in vanilla P2P should be preserved when mask_mode='attention'.
Also dumps the derived mask itself for inspection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image

from ddim import ddim_sample
from editor import Editor, _to_pil
from masks import visualize_mask
from sd_components import decode_latents, encode_prompt, load_sd

SRC = "a photograph of a cat sitting on a couch"
TGT = "a photograph of a dog sitting on a couch"
SEED = 0
STEPS = 50
GUIDANCE = 7.5

c = load_sd()
cond = encode_prompt(c, SRC)
uncond = encode_prompt(c, [""])

g = torch.Generator(device=c.device).manual_seed(SEED)
z_T = torch.randn(
    (1, c.unet.config.in_channels, 64, 64),
    generator=g, device=c.device, dtype=c.dtype,
)
z0 = ddim_sample(c, cond, uncond, num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=z_T.clone())
img_orig = _to_pil(decode_latents(c, z0).clamp(-1, 1))

editor = Editor(c)
out_dir = Path(__file__).resolve().parent.parent / "outputs" / "edit_mask_smoke"
out_dir.mkdir(parents=True, exist_ok=True)
img_orig.save(out_dir / "00_source.png")

print("editing without mask...")
img_no_mask = editor.edit(img_orig, SRC, TGT, num_inference_steps=STEPS, guidance_scale=GUIDANCE, tau=0.8)
img_no_mask.save(out_dir / "01_no_mask.png")

print("editing with attention mask...")
img_with_mask, mask = editor.edit(
    img_orig, SRC, TGT,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    tau=0.8, mask_mode="attention", return_mask=True,
)
img_with_mask.save(out_dir / "02_with_mask.png")

mask_vis = visualize_mask(mask, size=512)  # (512, 512) in [0, 1]
mask_pil = Image.fromarray((mask_vis.cpu().numpy() * 255).astype("uint8"))
mask_pil.save(out_dir / "03_mask.png")

print(f"saved 4 images to {out_dir}")
print(f"mask coverage: {mask.float().mean().item():.3f}  (fraction of pixels inside the mask)")
