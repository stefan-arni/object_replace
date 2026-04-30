"""
Diagnostic evaluation for the drag-diffusion pipeline.

Runs two synthetic checks:
1. Move case: compare the dragged result to a known ground-truth relocation.
2. No-op case: source and target are identical, so the pipeline should preserve the input.

Usage:
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 ./venv/bin/python diagnose_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.relocation_pipeline import ObjectRelocationPipeline
from utils.image_utils import get_device


def make_background(size: int = 512, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bg = np.zeros((size, size, 3), dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    bg[..., 1] = 0.28 + 0.18 * rng.random((size, size)) + 0.06 * np.sin(xx / 21.0) * np.cos(yy / 19.0)
    bg[..., 0] = 0.08 + 0.04 * rng.random((size, size))
    bg[..., 2] = 0.06 + 0.03 * rng.random((size, size))
    return np.clip(bg, 0, 1)


def add_ball(bg: np.ndarray, cy: int, cx: int, r: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = bg.copy()
    yy, xx = np.ogrid[:bg.shape[0], :bg.shape[1]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = dist < r
    angle = np.arctan2(yy - cy, xx - cx)
    grad = 0.55 + 0.25 * np.sin(angle * 3.0) + 0.08 * rng.random(bg.shape[:2])
    img[mask, 0] = (0.84 + 0.11 * grad[mask]).clip(0, 1)
    img[mask, 1] = (0.42 + 0.10 * grad[mask]).clip(0, 1)
    img[mask, 2] = (0.04 + 0.03 * grad[mask]).clip(0, 1)
    return img


def make_circle_mask(size: int, cy: int, cx: int, r: int, pad: int = 15) -> Image.Image:
    yy, xx = np.ogrid[:size, :size]
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[(yy - cy) ** 2 + (xx - cx) ** 2 < (r + pad) ** 2] = 255
    return Image.fromarray(arr)


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def masked_psnr(img1: Image.Image, img2: Image.Image, mask: Image.Image | None = None) -> float:
    a = np.asarray(img1.convert("RGB")).astype(np.float32) / 255.0
    b = np.asarray(img2.convert("RGB")).astype(np.float32) / 255.0
    if mask is not None:
        m = np.asarray(mask.convert("L")) > 127
        diff = (a[m] - b[m]) ** 2
    else:
        diff = (a - b) ** 2
    mse = diff.mean()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def invert_mask(mask: Image.Image) -> Image.Image:
    return Image.fromarray((255 - np.asarray(mask.convert("L"))).astype(np.uint8))


def union_mask(*masks: Image.Image) -> Image.Image:
    arr = np.zeros_like(np.asarray(masks[0].convert("L")))
    for mask in masks:
        arr = np.maximum(arr, np.asarray(mask.convert("L")))
    return Image.fromarray(arr)


def run_diagnostics(
    out_dir: Path,
    steps: int,
    strength: float,
    guidance: float,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    size = 512
    cy = size // 2
    cx_src = size // 5
    cx_tgt = size * 4 // 5
    r = size // 8

    bg = make_background(size)
    input_image = to_pil(add_ball(bg, cy, cx_src, r))
    gt_move = to_pil(add_ball(bg, cy, cx_tgt, r))

    src_mask = make_circle_mask(size, cy, cx_src, r)
    tgt_mask = make_circle_mask(size, cy, cx_tgt, r)
    bg_only = invert_mask(union_mask(src_mask, tgt_mask))

    input_image.save(out_dir / "input.png")
    gt_move.save(out_dir / "ground_truth_move.png")
    src_mask.save(out_dir / "src_mask.png")
    tgt_mask.save(out_dir / "tgt_mask.png")

    device = get_device()
    print(f"Using device: {device}", flush=True)
    pipe = ObjectRelocationPipeline(device=device, local_files_only=False)
    prompt = "an orange ball on the right side of a green grassy field"
    settings = dict(
        seed=seed,
        num_inference_steps=steps,
        sdedit_strength=strength,
        guidance_scale=guidance,
    )

    print("Running move case...", flush=True)
    baseline, composite = pipe(input_image, prompt, src_mask, tgt_mask, use_noise_shift=False, **settings)
    ours, _ = pipe(input_image, prompt, src_mask, tgt_mask, use_noise_shift=True, **settings)

    composite.save(out_dir / "move_composite.png")
    baseline.save(out_dir / "move_baseline.png")
    ours.save(out_dir / "move_ours.png")

    print("Running no-op case...", flush=True)
    noop_baseline, noop_composite = pipe(
        input_image,
        "an orange ball on grass",
        src_mask,
        src_mask,
        use_noise_shift=False,
        **settings,
    )
    noop_ours, _ = pipe(
        input_image,
        "an orange ball on grass",
        src_mask,
        src_mask,
        use_noise_shift=True,
        **settings,
    )

    noop_composite.save(out_dir / "noop_composite.png")
    noop_baseline.save(out_dir / "noop_baseline.png")
    noop_ours.save(out_dir / "noop_ours.png")

    results = {}
    for name, img in [("composite", composite), ("baseline", baseline), ("ours", ours)]:
        results[f"move_{name}"] = {
            "full_psnr_vs_gt": masked_psnr(gt_move, img),
            "background_psnr_vs_gt": masked_psnr(gt_move, img, bg_only),
            "source_hole_psnr_vs_gt": masked_psnr(gt_move, img, src_mask),
            "target_region_psnr_vs_gt": masked_psnr(gt_move, img, tgt_mask),
        }

    for name, img in [("composite", noop_composite), ("baseline", noop_baseline), ("ours", noop_ours)]:
        results[f"noop_{name}"] = {
            "full_psnr_vs_input": masked_psnr(input_image, img),
            "background_psnr_vs_input": masked_psnr(input_image, img, bg_only),
            "object_region_psnr_vs_input": masked_psnr(input_image, img, src_mask),
        }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/results/diagnostics")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_diagnostics(
        Path(args.out_dir),
        steps=args.steps,
        strength=args.strength,
        guidance=args.guidance,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
