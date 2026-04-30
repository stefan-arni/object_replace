import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def prepare_latent_mask(
    mask_pil: Image.Image,
    device: torch.device,
    latent_size: int = 96,
) -> torch.Tensor:
    """
    PIL binary mask → [1, 1, latent_size, latent_size] float32 on device.
    Uses nearest-neighbor to preserve binary values.
    latent_size = image_size // 8  (96 for SD 2.1 at 768px, 64 for SD 1.5 at 512px)
    """
    arr = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    t = (t > 0.5).float()
    t = F.interpolate(t, size=(latent_size, latent_size), mode="nearest")
    return t.to(device=device, dtype=torch.float32)


def compute_centroid(mask: torch.Tensor, device: torch.device):
    """
    mask: [1, 1, H, W] float32 — returns (cy, cx) in pixel coordinates.
    Falls back to image center if mask is empty.
    """
    mask = mask.squeeze()  # [H, W]
    H, W = mask.shape
    if mask.sum() < 1e-6:
        return H / 2.0, W / 2.0
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    total = mask.sum()
    cy = (mask * ys.unsqueeze(1)).sum() / total
    cx = (mask * xs.unsqueeze(0)).sum() / total
    return cy.item(), cx.item()


def gaussian_blur_mask(mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    Soft feathering of a binary mask via Gaussian blur.
    mask: [1, 1, H, W] float32 → [1, 1, H, W] float32 (smooth edges)
    """
    # Build a small Gaussian kernel
    kernel_size = int(6 * sigma + 1) | 1  # ensure odd
    coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
    pad = kernel_size // 2
    blurred = F.conv2d(mask, kernel, padding=pad)
    return blurred.clamp(0.0, 1.0)
