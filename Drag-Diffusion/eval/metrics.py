"""
Eval metrics for object relocation quality.

Metrics:
  - VGG perceptual distance (texture preservation at target) — from perceptual_loss.py
  - PSNR   (background fidelity, masked)
  - SSIM   (background structural similarity, masked)
  - CLIP   (text–image alignment of result)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── PSNR ─────────────────────────────────────────────────────────────────────

def compute_psnr(
    img1: Image.Image,
    img2: Image.Image,
    mask: Image.Image = None,
) -> float:
    """
    Peak signal-to-noise ratio in dB (higher = better background preservation).
    If mask provided (PIL L or 1), computes PSNR only on non-zero pixels.
    img1 / img2: RGB PIL images of the same size.
    """
    a1 = np.array(img1.convert("RGB")).astype(np.float32) / 255.0
    a2 = np.array(img2.convert("RGB")).astype(np.float32) / 255.0

    if mask is not None:
        m = np.array(mask.convert("L")) > 127  # True = background pixel
        if m.sum() == 0:
            return float("nan")
        diff = (a1[m] - a2[m]) ** 2
    else:
        diff = (a1 - a2) ** 2

    mse = diff.mean()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ── SSIM ─────────────────────────────────────────────────────────────────────

def compute_ssim(
    img1: Image.Image,
    img2: Image.Image,
    mask: Image.Image = None,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Structural similarity index (higher = better).
    If mask provided, returns mean SSIM only over masked pixels.
    """
    def _to_gray_tensor(img):
        arr = np.array(img.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    t1 = _to_gray_tensor(img1)
    t2 = _to_gray_tensor(img2)

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

    pad = window_size // 2
    mu1 = F.conv2d(t1, kernel, padding=pad)
    mu2 = F.conv2d(t2, kernel, padding=pad)

    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(t1 * t1, kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(t2 * t2, kernel, padding=pad) - mu2_sq
    sigma12   = F.conv2d(t1 * t2, kernel, padding=pad) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_map.squeeze().numpy()  # [H, W]

    if mask is not None:
        m = np.array(mask.convert("L")) > 127
        if m.sum() == 0:
            return float("nan")
        # ssim_map may be 1px smaller on each side due to conv; use center crop
        h, w = ssim_map.shape
        mh, mw = m.shape
        if h < mh or w < mw:
            m = m[:h, :w]
        return float(ssim_map[m].mean())
    return float(ssim_map.mean())


# ── CLIP score ────────────────────────────────────────────────────────────────

_clip_model = None
_clip_processor = None


def _load_clip(device):
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _id = "openai/clip-vit-base-patch32"
        try:
            _clip_processor = CLIPProcessor.from_pretrained(_id, local_files_only=True)
            _clip_model = CLIPModel.from_pretrained(_id, local_files_only=True)
        except Exception as e:
            print(f"CLIP not cached ({e}) — downloading clip-vit-base-patch32...")
            _clip_processor = CLIPProcessor.from_pretrained(_id)
            _clip_model = CLIPModel.from_pretrained(_id)
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_processor


def compute_clip_score(image: Image.Image, prompt: str, device: torch.device) -> float:
    """
    Cosine similarity between CLIP text and image embeddings (higher = better).
    Returns value in [-1, 1]; typical good results are 0.2–0.35.
    """
    try:
        model, processor = _load_clip(device)
    except Exception as e:
        print(f"CLIP score unavailable: {e}")
        return float("nan")

    inputs = processor(text=[prompt], images=[image.convert("RGB")],
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        score = (img_emb * txt_emb).sum().item()

    return score


# ── Convenience: run all metrics ──────────────────────────────────────────────

def evaluate_relocation(
    original: Image.Image,
    result: Image.Image,
    source_mask: Image.Image,
    target_mask: Image.Image,
    prompt: str,
    device: torch.device,
    perceptual_loss_fn=None,
) -> dict:
    """
    Returns a dict with all four metrics.

    original / result: PIL RGB, same size.
    source_mask / target_mask: PIL L, same size as images.
    perceptual_loss_fn: VGGPerceptualLoss instance (optional; skipped if None).
    """
    from utils.image_utils import pil_to_tensor

    sz = original.size  # (W, H)

    # Background mask = pixels NOT covered by source or target
    src_arr = np.array(source_mask.resize(sz, Image.NEAREST).convert("L"))
    tgt_arr = np.array(target_mask.resize(sz, Image.NEAREST).convert("L"))
    bg_arr = ((src_arr < 128) & (tgt_arr < 128)).astype(np.uint8) * 255
    bg_mask_pil = Image.fromarray(bg_arr, mode="L")

    results = {}

    # 1. Background PSNR
    results["bg_psnr"] = compute_psnr(original, result, mask=bg_mask_pil)

    # 2. Background SSIM
    results["bg_ssim"] = compute_ssim(original, result, mask=bg_mask_pil)

    # 3. Object perceptual distance (texture preservation)
    if perceptual_loss_fn is not None:
        tgt_arr_bool = tgt_arr > 127
        ys, xs = np.where(tgt_arr_bool)
        if len(ys) > 0:
            pad = 10
            y0, y1 = max(0, ys.min() - pad), min(sz[1], ys.max() + pad)
            x0, x1 = max(0, xs.min() - pad), min(sz[0], xs.max() + pad)
            # Source crop (same bounding box in source region)
            src_bool = src_arr > 127
            sys_, sxs = np.where(src_bool)
            if len(sys_) > 0:
                sy0 = max(0, sys_.min() - pad); sy1 = min(sz[1], sys_.max() + pad)
                sx0 = max(0, sxs.min() - pad); sx1 = min(sz[0], sxs.max() + pad)
                src_crop = pil_to_tensor(
                    original.crop((sx0, sy0, sx1, sy1)).resize((256, 256)), device
                )
                tgt_crop = pil_to_tensor(
                    result.crop((x0, y0, x1, y1)).resize((256, 256)), device
                )
                results["perceptual_dist"] = perceptual_loss_fn(src_crop, tgt_crop)
            else:
                results["perceptual_dist"] = float("nan")
        else:
            results["perceptual_dist"] = float("nan")
    else:
        results["perceptual_dist"] = float("nan")

    # 4. CLIP score
    results["clip_score"] = compute_clip_score(result, prompt, device)

    return results


def print_metrics_table(baseline_metrics: dict, ours_metrics: dict):
    """Pretty-print a two-column comparison table."""
    rows = [
        ("Background PSNR (dB) ↑",  "bg_psnr",        True),
        ("Background SSIM ↑",        "bg_ssim",        True),
        ("Perceptual dist ↓",        "perceptual_dist", False),
        ("CLIP score ↑",             "clip_score",      True),
    ]
    print(f"\n{'Metric':<28} {'Baseline':>12} {'Ours':>12}  Winner")
    print("-" * 62)
    for label, key, higher_is_better in rows:
        bv = baseline_metrics.get(key, float("nan"))
        ov = ours_metrics.get(key, float("nan"))
        try:
            if higher_is_better:
                winner = "ours ✓" if ov > bv else ("baseline" if bv > ov else "tie")
            else:
                winner = "ours ✓" if ov < bv else ("baseline" if bv < ov else "tie")
        except Exception:
            winner = "?"
        print(f"{label:<28} {bv:>12.4f} {ov:>12.4f}  {winner}")
    print()
