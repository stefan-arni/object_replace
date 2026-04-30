"""
Quick self-contained test — no external images needed.
Creates a synthetic scene, runs baseline vs. noise-shift, saves comparison.

Usage:  python quick_test.py
Output: data/results/quick_test.png
"""

import numpy as np
from PIL import Image, ImageDraw
import os

os.makedirs("data/results", exist_ok=True)


def make_test_scene(size=512):
    """Textured green background + orange 'ball' on the left side."""
    rng = np.random.default_rng(7)

    # Background: green with grass-like texture
    bg = np.zeros((size, size, 3), dtype=np.float32)
    bg[:, :, 1] = 0.35 + 0.15 * rng.random((size, size))  # green channel
    bg[:, :, 0] = 0.05 + 0.05 * rng.random((size, size))
    bg[:, :, 2] = 0.05 + 0.05 * rng.random((size, size))

    # Orange ball at left-center
    cy, cx = size // 2, size // 5
    r = size // 8
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    ball_mask = dist < r

    # Give ball a gradient + speckle texture so texture preservation is visible
    angle = np.arctan2(yy - cy, xx - cx)
    gradient = 0.5 + 0.3 * np.sin(angle * 3) + 0.1 * rng.random((size, size))
    bg[ball_mask, 0] = (0.85 + 0.1 * gradient[ball_mask]).clip(0, 1)  # orange-red
    bg[ball_mask, 1] = (0.45 + 0.1 * gradient[ball_mask]).clip(0, 1)  # orange-green
    bg[ball_mask, 2] = 0.05

    img_arr = (bg * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)


def make_circle_mask(size, cy, cx, r, pad=15):
    """Binary PIL mask with a circular region."""
    arr = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    arr[(yy - cy)**2 + (xx - cx)**2 < (r + pad)**2] = 255
    return Image.fromarray(arr)


if __name__ == "__main__":
    from utils.image_utils import get_device
    from pipeline.relocation_pipeline import ObjectRelocationPipeline
    from eval.perceptual_loss import VGGPerceptualLoss
    from eval.visualize import make_comparison_figure

    SIZE = 512
    device = get_device()

    print("Building test scene...")
    image = make_test_scene(SIZE)
    image.save("data/results/quick_test_input.png")

    cy, cx_src = SIZE // 2, SIZE // 5
    cx_tgt = SIZE * 4 // 5
    r = SIZE // 8

    src_mask = make_circle_mask(SIZE, cy, cx_src, r)
    tgt_mask = make_circle_mask(SIZE, cy, cx_tgt, r)

    src_mask.save("data/results/quick_test_src_mask.png")
    tgt_mask.save("data/results/quick_test_tgt_mask.png")

    print("Loading pipeline...")
    pipe = ObjectRelocationPipeline(device=device)
    loss_fn = VGGPerceptualLoss(device=device)

    prompt = "an orange ball on the right side of a green grassy field"
    steps = 30   # reasonable for local testing; 50 for final results
    strength = 0.5  # add 50% noise to composite before denoising

    print(f"Running baseline (SDEdit strength={strength}, {steps} steps)...")
    baseline, composite = pipe(image, prompt, src_mask, tgt_mask,
                    use_noise_shift=False, seed=42, num_inference_steps=steps,
                    sdedit_strength=strength)
    baseline.save("data/results/quick_test_baseline.png")
    composite.save("data/results/quick_test_composite.png")
    print("  → composite (rough copy-paste, before harmonization) saved")

    print(f"Running ours (DDPM noise shift, {steps} steps)...")
    ours, _ = pipe(image, prompt, src_mask, tgt_mask,
                use_noise_shift=True, seed=42, num_inference_steps=steps,
                sdedit_strength=strength)
    ours.save("data/results/quick_test_ours.png")

    from eval.metrics import evaluate_relocation, print_metrics_table

    print("\nComputing metrics...")
    bl_metrics = evaluate_relocation(
        image, baseline, src_mask, tgt_mask, prompt, device,
        perceptual_loss_fn=loss_fn,
    )
    ou_metrics = evaluate_relocation(
        image, ours, src_mask, tgt_mask, prompt, device,
        perceptual_loss_fn=loss_fn,
    )

    print_metrics_table(bl_metrics, ou_metrics)

    out = "data/results/quick_test.png"
    make_comparison_figure(image, baseline, ours, out,
                           title=prompt,
                           baseline_metrics=bl_metrics,
                           ours_metrics=ou_metrics)
    print(f"Saved comparison → {out}")
    print("Open it with:  open data/results/quick_test.png")
