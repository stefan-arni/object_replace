"""Step 5 smoke: install StoreController, run a few sampling steps, verify
attention maps were captured at all expected resolutions and layer counts.

SD 1.5 UNet has 16 cross-attn layers (3 down-block stages x 2 + 1 mid + 3 up-block
stages x 3 = 16) at resolutions {64, 32, 16, 8}.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from attention_store import StoreController, install_controller, uninstall_controller
from ddim import _alpha_bar
from sd_components import encode_prompt, load_sd

PROMPT = "a photograph of a cat sitting on a couch"
STEPS = 5

c = load_sd()
cond = encode_prompt(c, PROMPT)
uncond = encode_prompt(c, [""])

ctrl = StoreController(store_self=False)
install_controller(c.unet, ctrl)

c.scheduler.set_timesteps(STEPS, device=c.device)
timesteps = c.scheduler.timesteps
final_alpha_bar = c.scheduler.final_alpha_cumprod.to(c.device)
embed_pair = torch.cat([uncond, cond], dim=0)

g = torch.Generator(device=c.device).manual_seed(0)
latents = torch.randn((1, c.unet.config.in_channels, 64, 64), generator=g, device=c.device, dtype=c.dtype)

with torch.no_grad():
    for i, t in enumerate(timesteps):
        ctrl.cur_t = int(t)
        x_in = torch.cat([latents, latents], dim=0)
        eps_u, eps_c = c.unet(x_in, t, encoder_hidden_states=embed_pair).sample.chunk(2)
        eps = eps_u + 7.5 * (eps_c - eps_u)
        a_t = _alpha_bar(c, t)
        a_prev = _alpha_bar(c, timesteps[i + 1]) if i + 1 < len(timesteps) else final_alpha_bar
        x0 = (latents - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        latents = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

uninstall_controller(c.unet)

print(f"total maps captured: {len(ctrl.maps)}  (expect {STEPS * 16})")

by_t = defaultdict(list)
for (t, name), v in ctrl.maps.items():
    by_t[t].append((name, tuple(v.shape)))

print()
for t in sorted(by_t.keys(), reverse=True):
    print(f"t={t}: {len(by_t[t])} cross-attn layers")
    for name, shape in by_t[t][:2]:
        print(f"    {name}  shape={shape}")
    if len(by_t[t]) > 2:
        print(f"    ... ({len(by_t[t]) - 2} more)")

spatial = defaultdict(int)
for (_, name), v in ctrl.maps.items():
    HW = v.shape[1]
    R = int(math.isqrt(HW))
    spatial[R] += 1

print()
print("layers per resolution (across all timesteps):")
for r in sorted(spatial.keys(), reverse=True):
    print(f"  {r:>3}x{r:<3}  count={spatial[r]}  (per-step={spatial[r] // STEPS})")
print("expected resolutions: [64, 32, 16, 8]")
