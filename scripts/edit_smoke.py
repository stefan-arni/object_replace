"""Step 6 smoke: generate a cat image, then use the Editor to swap cat -> dog
via vanilla P2P replace at tau=0.8. Save source and edited side by side.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from ddim import ddim_sample
from editor import Editor, _to_pil
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
print("Editing (will reuse cached null-text inversion if cat sanity ran first)...")
img_edit = editor.edit(
    img_orig, SRC, TGT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    tau=0.8,
)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "edit_smoke"
out_dir.mkdir(parents=True, exist_ok=True)
img_orig.save(out_dir / "source.png")
img_edit.save(out_dir / "edited.png")
print(f"saved to {out_dir}")
