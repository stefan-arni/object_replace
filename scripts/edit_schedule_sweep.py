"""Step 7 smoke / mini-ablation: same cat -> dog edit, 5 different schedules.

Inversion is cached, so all five edits share the same null-text result and
only differ in how the controller blends source/target cross-attn at each
timestep. Inspect the PNGs to see how schedule shape affects the swap.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from ddim import ddim_sample
from editor import Editor, _to_pil
from sd_components import decode_latents, encode_prompt, load_sd
from schedules import (
    constant_replaced,
    cosine_replaced,
    linear_decay_replaced,
    piecewise_demo,
    vanilla_p2p,
)

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
out_dir = Path(__file__).resolve().parent.parent / "outputs" / "edit_schedule_sweep"
out_dir.mkdir(parents=True, exist_ok=True)
img_orig.save(out_dir / "00_source.png")

# Hypothesis under test: replaced-token schedule shape matters more than the preserved one.
# For cat<->dog (structural edit, similar silhouette), we expect fast-decay shapes
# (vanilla_p2p, linear_decay) to look better than slow shapes (constant_0.5).
schedules = {
    "01_vanilla_p2p_tau0.8":   vanilla_p2p(0.8),
    "02_linear_decay":         linear_decay_replaced(),
    "03_cosine":               cosine_replaced(),
    "04_constant_0.5":         constant_replaced(0.5),
    "05_piecewise_demo":       piecewise_demo(),
}

for name, sched in schedules.items():
    print(f"editing with schedule={name}")
    img = editor.edit(
        img_orig, SRC, TGT,
        schedule=sched,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
    )
    img.save(out_dir / f"{name}.png")

print(f"saved {len(schedules) + 1} images to {out_dir}")
