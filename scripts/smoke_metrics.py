"""Step 9 smoke: run metrics on the outputs already sitting in outputs/.

Reuses the cat <-> dog edits from the previous smoke scripts so we don't have
to regenerate anything. Prints a small table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from metrics import background_lpips, clip_directional_similarity, reconstruction_lpips

SRC = "a photograph of a cat sitting on a couch"
TGT = "a photograph of a dog sitting on a couch"

base = Path(__file__).resolve().parent.parent / "outputs"

print("--- reconstruction (null-text inversion sanity) ---")
nt_dir = base / "sanity_null_text"
if (nt_dir / "original.png").exists():
    src_img = Image.open(nt_dir / "original.png")
    recon_img = Image.open(nt_dir / "reconstructed.png")
    print(f"  reconstruction LPIPS = {reconstruction_lpips(src_img, recon_img):.4f}   (target < 0.05)")
else:
    print("  (no sanity_null_text outputs found, skipping)")

print()
print("--- per-schedule edits (cat -> dog) ---")
sweep_dir = base / "edit_schedule_sweep"
if (sweep_dir / "00_source.png").exists():
    src_for_edit = Image.open(sweep_dir / "00_source.png")
    print(f"  {'schedule':<28} {'CLIP-dir':>10} {'full-LPIPS':>12}")
    for f in sorted(sweep_dir.glob("0[1-9]_*.png")):
        edit_img = Image.open(f)
        clip_dir = clip_directional_similarity(src_for_edit, edit_img, SRC, TGT)
        full = reconstruction_lpips(src_for_edit, edit_img)
        print(f"  {f.stem:<28} {clip_dir:>10.4f} {full:>12.4f}")
    print("  (CLIP-dir target > 0.20; higher = better edit semantics)")
else:
    print("  (no edit_schedule_sweep outputs found, skipping)")

print()
print("--- mask vs no-mask (background preservation) ---")
mask_dir = base / "edit_mask_smoke"
if (mask_dir / "03_mask.png").exists():
    src_mask = Image.open(mask_dir / "00_source.png")
    edit_no = Image.open(mask_dir / "01_no_mask.png")
    edit_yes = Image.open(mask_dir / "02_with_mask.png")
    mask_pil = Image.open(mask_dir / "03_mask.png").convert("L")
    mask_arr = np.asarray(mask_pil) / 255.0
    mask_t = torch.from_numpy(mask_arr).float()

    bg_no = background_lpips(src_mask, edit_no, mask_t)
    bg_yes = background_lpips(src_mask, edit_yes, mask_t)
    clip_no = clip_directional_similarity(src_mask, edit_no, SRC, TGT)
    clip_yes = clip_directional_similarity(src_mask, edit_yes, SRC, TGT)

    print(f"  {'mode':<14} {'CLIP-dir':>10} {'bg-LPIPS':>10}")
    print(f"  {'no_mask':<14} {clip_no:>10.4f} {bg_no:>10.4f}")
    print(f"  {'with_mask':<14} {clip_yes:>10.4f} {bg_yes:>10.4f}")
    print("  (bg-LPIPS target < 0.10; with_mask should be lower than no_mask)")
else:
    print("  (no edit_mask_smoke outputs found, skipping)")
