# -------------------------------------------------------------------
# NOVEL CONTRIBUTION (part 1 of 2):
#   DDPM forward-process inversion — stores a reconstruction-consistent
#   latent trajectory plus both marginal and transition noise maps.
# -------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict

import torch
from diffusers import DDPMScheduler


@dataclass
class DDPMInversionTrajectory:
    """
    A sampled latent trajectory over the scheduler's inference timesteps.

    latents[t]            = sampled x_t at timestep t
    marginal_noises[t]    = epsilon_t such that x_t = sqrt(abar_t) * x0 + sqrt(1-abar_t) * epsilon_t
    transition_noises[t]  = z used when sampling x_t from q(x_t | x_{t_next}, x0)
                             during the reverse traversal over inference timesteps
    """
    latents: Dict[int, torch.Tensor]
    marginal_noises: Dict[int, torch.Tensor]
    transition_noises: Dict[int, torch.Tensor]


def _make_generator(device: torch.device, seed: int):
    gen_device = device if device.type in {"cuda", "mps"} else torch.device("cpu")
    try:
        generator = torch.Generator(device=gen_device).manual_seed(seed)
    except RuntimeError:
        gen_device = torch.device("cpu")
        generator = torch.Generator(device=gen_device).manual_seed(seed)
    return generator, gen_device


def _randn_like(
    x: torch.Tensor,
    generator: torch.Generator,
    sample_device: torch.device,
) -> torch.Tensor:
    noise = torch.randn(
        x.shape,
        generator=generator,
        device=sample_device,
        dtype=torch.float32,
    )
    if sample_device != x.device:
        noise = noise.to(x.device)
    return noise


def ddpm_invert(
    x0_latent: torch.Tensor,
    scheduler: DDPMScheduler,
    num_inference_steps: int = 50,
    seed: int = 42,
    device: torch.device = None,
) -> DDPMInversionTrajectory:
    """
    Sample a DDPM trajectory that is internally consistent across inference steps.

    x0_latent: [1, 4, 64, 64] float32, already scaled by 0.18215
    Returns a DDPMInversionTrajectory containing:
      - sampled x_t states at each inference timestep
      - marginal noises epsilon_t for closed-form x_t reconstruction
      - transition noises z_t used by q(x_{t_prev} | x_t, x0)
    """
    if device is None:
        device = x0_latent.device

    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    generator, sample_device = _make_generator(device, seed)

    latents = {}
    marginal_noises = {}
    transition_noises = {}

    first_t = timesteps[0].item()
    abar_first = scheduler.alphas_cumprod[first_t].to(device=device, dtype=torch.float32)
    eps_first = _randn_like(x0_latent, generator, sample_device)
    x_t = abar_first.sqrt() * x0_latent + (1 - abar_first).sqrt() * eps_first

    latents[first_t] = x_t
    marginal_noises[first_t] = eps_first

    for idx, t in enumerate(timesteps[:-1]):
        t_int = t.item()
        t_prev_int = timesteps[idx + 1].item()

        abar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=torch.float32)
        abar_prev = scheduler.alphas_cumprod[t_prev_int].to(device=device, dtype=torch.float32)

        alpha_ratio = abar_t / abar_prev
        beta_t = (1 - alpha_ratio).clamp(min=0.0)

        coeff1 = abar_prev.sqrt() * beta_t / (1 - abar_t)
        coeff2 = alpha_ratio.sqrt() * (1 - abar_prev) / (1 - abar_t)
        mu = coeff1 * x0_latent + coeff2 * x_t

        sigma_t = (beta_t * (1 - abar_prev) / (1 - abar_t)).clamp(min=0.0).sqrt()
        z_t = _randn_like(x0_latent, generator, sample_device)
        x_prev = mu + sigma_t * z_t

        latents[t_prev_int] = x_prev
        transition_noises[t_prev_int] = z_t

        marginal_scale = (1 - abar_prev).sqrt().clamp(min=1e-8)
        marginal_noises[t_prev_int] = (x_prev - abar_prev.sqrt() * x0_latent) / marginal_scale
        x_t = x_prev

    return DDPMInversionTrajectory(
        latents=latents,
        marginal_noises=marginal_noises,
        transition_noises=transition_noises,
    )


def reconstruct_xt(
    x0_latent: torch.Tensor,
    t_int: int,
    eps_t: torch.Tensor,
    scheduler: DDPMScheduler,
    device: torch.device,
) -> torch.Tensor:
    """
    x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε_t
    Useful for validation: reconstruct and check noise extraction round-trips.
    """
    abar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=torch.float32)
    return abar_t.sqrt() * x0_latent + (1 - abar_t).sqrt() * eps_t
