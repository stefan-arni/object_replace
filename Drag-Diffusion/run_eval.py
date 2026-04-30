"""
Ablation evaluation — runs baseline vs. ours on all test cases in data/test_images/
and prints a perceptual distance table.

Expected directory structure per test case:
    data/test_images/<name>/image.jpg (or .png)
    data/test_images/<name>/src_mask.png
    data/test_images/<name>/tgt_mask.png
    data/test_images/<name>/prompt.txt
"""

import os
import json
from PIL import Image
import torch
from utils.image_utils import get_device, pil_to_tensor
from pipeline.relocation_pipeline import ObjectRelocationPipeline
from eval.perceptual_loss import VGGPerceptualLoss
from eval.visualize import make_comparison_figure

DATA_DIR = "data/test_images"
RESULTS_DIR = "data/results"


def crop_to_mask(image: Image.Image, mask: Image.Image) -> torch.Tensor:
    """Crop image to bounding box of mask, return as [1,3,512,512] tensor for perceptual loss."""
    import numpy as np
    from utils.image_utils import pil_to_tensor, get_device
    mask_arr = (np.array(mask.convert("L")) > 127).astype(np.uint8)
    rows = np.any(mask_arr, axis=1); cols = np.any(mask_arr, axis=0)
    if not rows.any():
        return pil_to_tensor(image.resize((512, 512)), get_device())
    r0, r1 = rows.argmax(), len(rows) - rows[::-1].argmax()
    c0, c1 = cols.argmax(), len(cols) - cols[::-1].argmax()
    cropped = image.crop((c0, r0, c1, r1)).resize((512, 512))
    return pil_to_tensor(cropped, get_device())


def main():
    device = get_device()
    pipe = ObjectRelocationPipeline(device=device)
    loss_fn = VGGPerceptualLoss(device=device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    test_cases = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not test_cases:
        print(f"No test cases found in {DATA_DIR}. Create subdirectories with image, src_mask, tgt_mask, prompt.txt.")
        return

    for name in test_cases:
        case_dir = os.path.join(DATA_DIR, name)
        image_path = next(
            (os.path.join(case_dir, f) for f in os.listdir(case_dir) if f.startswith("image")), None
        )
        if image_path is None:
            print(f"Skipping {name}: no image file found.")
            continue

        image = Image.open(image_path).convert("RGB")
        src_mask = Image.open(os.path.join(case_dir, "src_mask.png")).convert("L")
        tgt_mask = Image.open(os.path.join(case_dir, "tgt_mask.png")).convert("L")
        with open(os.path.join(case_dir, "prompt.txt")) as f:
            prompt = f.read().strip()

        print(f"\n[{name}] prompt: {prompt}")

        baseline, _ = pipe(image, prompt, src_mask, tgt_mask, use_noise_shift=False, seed=42)
        ours, _ = pipe(image, prompt, src_mask, tgt_mask, use_noise_shift=True, seed=42)

        # Perceptual distance: compare source object crop vs. result at target
        src_crop_t = crop_to_mask(image, src_mask)
        baseline_tgt_crop = crop_to_mask(baseline, tgt_mask)
        ours_tgt_crop = crop_to_mask(ours, tgt_mask)

        baseline_score = loss_fn(src_crop_t, baseline_tgt_crop)
        ours_score = loss_fn(src_crop_t, ours_tgt_crop)

        print(f"  Perceptual dist — baseline: {baseline_score:.4f} | ours: {ours_score:.4f}")
        results.append({"name": name, "baseline": baseline_score, "ours": ours_score})

        out_path = os.path.join(RESULTS_DIR, f"{name}_comparison.png")
        make_comparison_figure(
            image, baseline, ours, out_path, title=f"{name}: {prompt}",
            scores={"baseline": baseline_score, "ours": ours_score},
        )

    # Print summary table
    print("\n" + "=" * 50)
    print(f"{'Case':<20} {'Baseline':>12} {'Ours':>12} {'Δ':>10}")
    print("-" * 50)
    for r in results:
        delta = r["ours"] - r["baseline"]
        marker = "✓" if delta < 0 else "✗"
        print(f"{r['name']:<20} {r['baseline']:>12.4f} {r['ours']:>12.4f} {delta:>+10.4f} {marker}")
    print("=" * 50)

    with open(os.path.join(RESULTS_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
