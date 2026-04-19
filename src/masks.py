"""Derive a localized 'where the object is' mask from captured cross-attention.

The recipe (from the spec):
  - Pull cross-attn maps at mid timesteps (t_frac in [0.3, 0.7] of the schedule).
  - Restrict to layers at resolutions {16x16, 32x32} -- these encode layout.
    (64x64 is too local, 8x8 too coarse.)
  - Sum attention across the source-token columns (e.g., the column for 'cat'
    when editing cat -> dog).
  - Average over heads, then average all selected layer-timestep maps after
    upsampling them to 64x64.
  - Threshold at the 80th percentile, dilate a couple of pixels.

Returned shape (1, 1, 64, 64), float in {0., 1.}, matches latent spatial dims.
"""
import math

import torch
import torch.nn.functional as F


def derive_attention_mask(
    captured_maps: dict[tuple[int, str], torch.Tensor],
    source_token_indices: list[int],
    timesteps_desc_list: list[int],
    *,
    mid_t_range: tuple[float, float] = (0.3, 0.7),
    target_resolutions: tuple[int, ...] = (16, 32),
    upsample_to: int = 64,
    threshold_quantile: float = 0.8,
    dilate_pixels: int = 2,
    batch_size: int = 2,
    use_sample_index: int = 1,
) -> torch.Tensor:
    """captured_maps: from a StoreController. Each value has shape (B*H, R*R, L).

    `use_sample_index` picks which batch sample's heads to read (default 1 = the
    conditional pass under CFG, which carries the meaningful attention pattern).
    """
    if not source_token_indices:
        return torch.ones(1, 1, upsample_to, upsample_to)

    step_idx_for_t = {t: i for i, t in enumerate(timesteps_desc_list)}
    S = len(timesteps_desc_list)
    i_min = int(mid_t_range[0] * S)
    i_max = int(mid_t_range[1] * S)

    accumulator = torch.zeros(upsample_to, upsample_to)
    count = 0

    for (t_int, _layer), attn in captured_maps.items():
        i = step_idx_for_t.get(t_int)
        if i is None or not (i_min <= i <= i_max):
            continue

        BH, HW, _L = attn.shape
        R = int(math.isqrt(HW))
        if R * R != HW or R not in target_resolutions:
            continue

        H_per_sample = BH // batch_size
        sample_attn = attn[use_sample_index * H_per_sample : (use_sample_index + 1) * H_per_sample]
        col = sample_attn[:, :, source_token_indices].sum(dim=-1).mean(dim=0)  # (HW,)
        col_2d = col.reshape(R, R).float()

        col_up = F.interpolate(
            col_2d[None, None], size=(upsample_to, upsample_to),
            mode="bilinear", align_corners=False,
        )[0, 0]

        accumulator += col_up
        count += 1

    if count == 0:
        return torch.ones(1, 1, upsample_to, upsample_to)

    avg = accumulator / count
    threshold = torch.quantile(avg.flatten(), threshold_quantile)
    binary = (avg >= threshold).float()
    if dilate_pixels > 0:
        binary = _dilate(binary, dilate_pixels)
    return binary[None, None]


def _dilate(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Max-pool dilation. Input (H, W), output (H, W)."""
    k = 2 * pixels + 1
    return F.max_pool2d(mask[None, None], kernel_size=k, stride=1, padding=pixels)[0, 0]


def visualize_mask(mask: torch.Tensor, size: int = 512) -> torch.Tensor:
    """Upsample mask to viewable size for a quick PIL save. Returns (H, W) in [0, 1]."""
    m = mask
    if m.ndim == 4:
        m = m[0, 0]
    return F.interpolate(m[None, None].float(), size=(size, size), mode="nearest")[0, 0]
