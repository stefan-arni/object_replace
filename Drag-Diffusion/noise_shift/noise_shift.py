# -------------------------------------------------------------------
# NOVEL CONTRIBUTION (part 2 of 2):
#   DDPM noise prior shift — InstructUDrag (2024) Equation 4.
#   Spatially shifts per-timestep noise maps from source object
#   position to target position, carrying texture information.
# -------------------------------------------------------------------

import torch
import torch.nn.functional as F
from utils.mask_utils import compute_centroid, gaussian_blur_mask


def shift_noise_map(
    eps_t: torch.Tensor,
    M_src: torch.Tensor,
    M_tgt: torch.Tensor,
    device: torch.device,
    feather_sigma: float = 2.0,
) -> torch.Tensor:
    """
    InstructUDrag Eq. 4:
      ε_shifted = ε ⊙ (1 - M_src) + grid_sample(ε ⊙ M_src, affine_grid(T_Δ))

    eps_t:  [1, 4, H, W]
    M_src:  [1, 1, H, W] binary float32
    M_tgt:  [1, 1, H, W] binary float32
    Returns [1, 4, H, W]
    """
    _, _, H, W = eps_t.shape

    cy_s, cx_s = compute_centroid(M_src, device)
    cy_t, cx_t = compute_centroid(M_tgt, device)
    dy = cy_t - cy_s
    dx = cx_t - cx_s

    # affine_grid translates the *sampling grid*, so negate to move content
    dx_norm = -2.0 * dx / max(W - 1, 1)
    dy_norm = -2.0 * dy / max(H - 1, 1)

    theta = torch.tensor(
        [[1.0, 0.0, dx_norm], [0.0, 1.0, dy_norm]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)  # [1, 2, 3]

    grid = F.affine_grid(theta, (1, 4, H, W), align_corners=False)

    # Feather mask edges to reduce boundary discontinuities
    if feather_sigma > 0:
        M_src_soft = gaussian_blur_mask(M_src, sigma=feather_sigma)
    else:
        M_src_soft = M_src

    eps_src_region = eps_t * M_src_soft
    eps_src_shifted = F.grid_sample(
        eps_src_region, grid, align_corners=False, padding_mode="zeros", mode="bilinear"
    )

    return eps_t * (1.0 - M_src_soft) + eps_src_shifted


def shift_all_noise_maps(
    noise_maps: dict,
    M_src: torch.Tensor,
    M_tgt: torch.Tensor,
    device: torch.device,
    feather_sigma: float = 2.0,
) -> dict:
    """Apply shift_noise_map to every timestep's noise map."""
    return {
        t: shift_noise_map(eps, M_src, M_tgt, device, feather_sigma)
        for t, eps in noise_maps.items()
    }
