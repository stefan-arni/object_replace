import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    # float16 + autocast is safe on CUDA; MPS requires float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL RGB → [1, 3, H, W] float32 in [-1, 1]."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=torch.float32)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[1, 3, H, W] float32 in [-1, 1] → PIL RGB."""
    arr = t.squeeze(0).permute(1, 2, 0).float().clamp(-1, 1)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr)


def encode_image(vae, pixel_values: torch.Tensor) -> torch.Tensor:
    """[1, 3, H, W] float32 → [1, 4, H/8, W/8] latent, scaled by 0.18215."""
    with torch.no_grad():
        latent = vae.encode(pixel_values).latent_dist.sample() * 0.18215
    return latent.float()


def decode_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    """[1, 4, H/8, W/8] latent → [1, 3, H, W] float32 in [-1, 1]."""
    with torch.no_grad():
        decoded = vae.decode(latent / 0.18215).sample
    return decoded.float()


def create_composite(
    image: Image.Image,
    source_mask: Image.Image,
    target_mask: Image.Image,
) -> Image.Image:
    """
    Pixel-space copy-paste: move the object defined by source_mask to the
    centroid of target_mask. Fills the vacated source region with the local
    background mean color. Returns a PIL image the same size as the input.
    """
    img = np.array(image.convert("RGB")).astype(np.float32)
    src = np.array(source_mask.convert("L")) > 127
    tgt = np.array(target_mask.convert("L")) > 127

    H, W = img.shape[:2]
    sy, sx = np.where(src)
    ty, tx = np.where(tgt)

    if len(sy) == 0 or len(ty) == 0:
        return image.copy()

    dy = int(round(ty.mean() - sy.mean()))
    dx = int(round(tx.mean() - sx.mean()))

    composite = img.copy()

    # Fill source with a texture patch sampled from an adjacent background region.
    # Try shifting the source bounding box in 4 directions; fall back to mean color.
    y_min, y_max = sy.min(), sy.max()
    x_min, x_max = sx.min(), sx.max()
    box_h = int(y_max - y_min + 1)
    box_w = int(x_max - x_min + 1)

    patch_pixels = None
    for dy_p, dx_p in [(0, box_w), (0, -box_w), (box_h, 0), (-box_h, 0)]:
        new_sy = sy + dy_p
        new_sx = sx + dx_p
        if (new_sy.min() >= 0 and new_sy.max() < H and
                new_sx.min() >= 0 and new_sx.max() < W and
                src[new_sy, new_sx].sum() == 0):
            patch_pixels = img[new_sy, new_sx]
            break

    if patch_pixels is not None:
        composite[src] = patch_pixels
    else:
        bg_color = img[~src].mean(axis=0) if (~src).any() else np.array([128.0, 128.0, 128.0])
        composite[src] = bg_color

    # Paste object pixels at the shifted target position
    new_ys = np.clip(sy + dy, 0, H - 1)
    new_xs = np.clip(sx + dx, 0, W - 1)
    composite[new_ys, new_xs] = img[sy, sx]

    return Image.fromarray(composite.clip(0, 255).astype(np.uint8))
