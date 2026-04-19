"""Sweep schedules x mask_modes across data/prompts.json. Saves a per-image
grid of all variants plus a CSV of metrics for the report.

prompts.json format (list of objects):
  [
    {
      "image": "cat_couch.jpg",
      "source": "a photograph of a cat sitting on a couch",
      "target": "a photograph of a dog sitting on a couch"
    },
    ...
  ]
Image paths are resolved relative to data/real/.

For each image we derive the attention mask once via the source-side scout
pass and reuse it across all schedules so that bg_lpips numbers are directly
comparable. Null-text inversion is cached on disk between runs.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from editor import Editor
from metrics import background_lpips, clip_directional_similarity
from sd_components import load_sd
from schedules import (
    constant_replaced,
    cosine_replaced,
    linear_decay_replaced,
    piecewise_demo,
    vanilla_p2p,
)

SCHEDULES = {
    "vanilla":      lambda: vanilla_p2p(0.8),
    "linear":       linear_decay_replaced,
    "cosine":       cosine_replaced,
    "constant_0.5": lambda: constant_replaced(0.5),
    "piecewise":    piecewise_demo,
}
MASK_MODES = ["none", "attention"]


def make_grid(images, labels, save_path, n_cols=None):
    n = len(images)
    if n_cols is None:
        n_cols = n
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.3))
    axes = np.atleast_2d(axes)
    if axes.ndim == 1:
        axes = axes.reshape(n_rows, n_cols)

    for idx in range(n_rows * n_cols):
        r, col = idx // n_cols, idx % n_cols
        ax = axes[r][col]
        if idx < n:
            ax.imshow(images[idx])
            ax.set_title(labels[idx], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "prompts.json")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "outputs" / "ablation")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    args = ap.parse_args()

    if not args.prompts.exists():
        sys.exit(
            f"prompts file not found: {args.prompts}\n"
            'create it with entries like:\n'
            '  [{"image": "cat.jpg", "source": "...", "target": "..."}]\n'
            "and put the images under data/real/"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    entries = json.loads(args.prompts.read_text())
    print(f"loaded {len(entries)} entries from {args.prompts}")

    c = load_sd()
    editor = Editor(c)

    csv_path = args.out / "metrics.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["image", "schedule", "mask_mode", "clip_dir", "bg_lpips"])

        data_dir = args.prompts.parent / "real"
        for entry in entries:
            img_path = data_dir / entry["image"]
            if not img_path.exists():
                print(f"  SKIP {entry['image']}: not found at {img_path}")
                continue

            src_pil = Image.open(img_path).convert("RGB").resize((512, 512))
            src_prompt = entry["source"]
            tgt_prompt = entry["target"]
            print(f"\n=== {entry['image']}: {src_prompt!r} -> {tgt_prompt!r}")

            # one mask per image, reused across schedules for comparable bg_lpips
            print("  deriving mask via scout pass...")
            mask = editor.derive_mask(
                src_pil, src_prompt, tgt_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
            )

            row_imgs = [src_pil]
            row_labels = ["source"]

            for sched_name, sched_fn in SCHEDULES.items():
                for mask_mode in MASK_MODES:
                    schedule = sched_fn()
                    edit_pil = editor.edit(
                        src_pil, src_prompt, tgt_prompt,
                        schedule=schedule,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        mask_mode=mask_mode,
                        precomputed_mask=mask if mask_mode == "attention" else None,
                    )
                    clip_dir = clip_directional_similarity(src_pil, edit_pil, src_prompt, tgt_prompt)
                    bg = background_lpips(src_pil, edit_pil, mask)
                    print(f"  {sched_name:<14} mask={mask_mode:<10}  clip_dir={clip_dir:.4f}  bg_lpips={bg:.4f}")
                    writer.writerow([entry["image"], sched_name, mask_mode, f"{clip_dir:.4f}", f"{bg:.4f}"])
                    row_imgs.append(edit_pil)
                    row_labels.append(f"{sched_name}\n{mask_mode}")

            grid_path = args.out / f"{Path(entry['image']).stem}_grid.png"
            make_grid(row_imgs, row_labels, grid_path, n_cols=len(SCHEDULES) + 1)
            print(f"  grid: {grid_path}")

    print(f"\nmetrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
