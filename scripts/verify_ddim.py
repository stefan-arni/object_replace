"""One-shot verification: my DDIM sampler vs the diffusers pipeline.

Same prompt, same seed, same init noise, same scheduler, same guidance.
Outputs should match within fp16 slop. Delete this script after it passes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image

from ddim import ddim_sample
from sd_components import MODEL_ID, decode_latents, encode_prompt, load_sd

PROMPT = "a photograph of an astronaut riding a horse"
SEED = 0
STEPS = 50
GUIDANCE = 7.5

c = load_sd()

g = torch.Generator(device=c.device).manual_seed(SEED)
init_noise = torch.randn(
    (1, c.unet.config.in_channels, 64, 64),
    generator=g, device=c.device, dtype=c.dtype,
)

cond = encode_prompt(c, PROMPT)
uncond = encode_prompt(c, [""])
mine_latents = ddim_sample(
    c, cond, uncond,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    latents=init_noise.clone(),
)
mine_pixels = decode_latents(c, mine_latents)
mine_pixels = (mine_pixels.clamp(-1, 1) + 1) / 2  # to [0, 1]

# diffusers reference. Delete this block once verified.
from diffusers import DDIMScheduler, StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=c.dtype, safety_checker=None
).to(c.device)
pipe.scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
pipe.set_progress_bar_config(disable=True)

ref = pipe(
    PROMPT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    latents=init_noise.clone(),
    output_type="pt",
).images  # [0, 1], B x 3 x H x W

diff = (mine_pixels - ref).abs()
print(f"mean abs diff: {diff.mean().item():.6f}")
print(f"max  abs diff: {diff.max().item():.6f}")

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "verify_ddim"
out_dir.mkdir(parents=True, exist_ok=True)


def to_pil(t):
    arr = (t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


to_pil(mine_pixels).save(out_dir / "mine.png")
to_pil(ref).save(out_dir / "ref.png")
print(f"saved comparison images to {out_dir}")
