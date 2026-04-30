import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


def make_comparison_figure(
    original: Image.Image,
    baseline: Image.Image,
    ours: Image.Image,
    save_path: str,
    title: str = "",
    scores: dict = None,
    baseline_metrics: dict = None,
    ours_metrics: dict = None,
):
    """
    3-panel figure: original | SDEdit baseline | ours (DDPM noise shift).

    scores: legacy — {'baseline': 0.34, 'ours': 0.21} perceptual dist only.
    baseline_metrics / ours_metrics: full dicts from eval/metrics.py
      (bg_psnr, bg_ssim, perceptual_dist, clip_score).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [original, baseline, ours]
    labels = ["Original", "SDEdit Baseline", "Ours (DDPM Noise Shift)"]

    def _fmt(m, legacy_key, legacy_val):
        if m is None:
            if legacy_val is not None:
                return f"\nPerceptual dist: {legacy_val:.4f}"
            return ""
        lines = []
        if "perceptual_dist" in m and not _isnan(m["perceptual_dist"]):
            lines.append(f"Perceptual: {m['perceptual_dist']:.4f} ↓")
        if "bg_psnr" in m and not _isnan(m["bg_psnr"]):
            lines.append(f"BG PSNR: {m['bg_psnr']:.1f} dB ↑")
        if "bg_ssim" in m and not _isnan(m["bg_ssim"]):
            lines.append(f"BG SSIM: {m['bg_ssim']:.4f} ↑")
        if "clip_score" in m and not _isnan(m["clip_score"]):
            lines.append(f"CLIP: {m['clip_score']:.4f} ↑")
        return "\n" + "\n".join(lines) if lines else ""

    import math
    def _isnan(v):
        try:
            return math.isnan(v)
        except Exception:
            return False

    if baseline_metrics or scores:
        labels[1] += _fmt(baseline_metrics, "baseline",
                           scores.get("baseline") if scores else None)
    if ours_metrics or scores:
        labels[2] += _fmt(ours_metrics, "ours",
                           scores.get("ours") if scores else None)

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(np.array(img.convert("RGB")))
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {save_path}")
