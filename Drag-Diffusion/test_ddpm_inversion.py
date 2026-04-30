import torch
from diffusers import DDPMScheduler

from inversion.ddpm_inversion import ddpm_invert, reconstruct_xt


def reverse_step_with_oracle_x0(
    x_t: torch.Tensor,
    x0_latent: torch.Tensor,
    scheduler: DDPMScheduler,
    t_int: int,
    t_prev_int: int,
    stored_noise: torch.Tensor,
) -> torch.Tensor:
    abar_t = scheduler.alphas_cumprod[t_int].to(device=x_t.device, dtype=torch.float32)
    eps_pred = (x_t - abar_t.sqrt() * x0_latent) / (1 - abar_t).sqrt()

    pred_x0 = (x_t - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
    if t_prev_int < 0:
        return pred_x0

    abar_prev = scheduler.alphas_cumprod[t_prev_int].to(device=x_t.device, dtype=torch.float32)
    coeff1 = abar_prev.sqrt() * (1 - abar_t / abar_prev) / (1 - abar_t)
    coeff2 = (abar_t / abar_prev).sqrt() * (1 - abar_prev) / (1 - abar_t)
    mu = coeff1 * pred_x0 + coeff2 * x_t

    beta_t = (1 - abar_t / abar_prev).clamp(min=0.0)
    sigma_t = (beta_t * (1 - abar_prev) / (1 - abar_t)).clamp(min=0.0).sqrt()
    return mu + sigma_t * stored_noise


def test_ddpm_invert_reconstructs_sampled_latents():
    device = torch.device("cpu")
    x0 = torch.randn(1, 4, 8, 8, device=device, dtype=torch.float32)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    traj = ddpm_invert(x0, scheduler, num_inference_steps=12, seed=7, device=device)

    for t in scheduler.timesteps.tolist():
        rebuilt = reconstruct_xt(x0, t, traj.marginal_noises[t], scheduler, device)
        assert torch.allclose(rebuilt, traj.latents[t], atol=1e-5, rtol=1e-5)


def test_ddpm_invert_round_trips_with_transition_noises():
    device = torch.device("cpu")
    x0 = torch.randn(1, 4, 8, 8, device=device, dtype=torch.float32)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    traj = ddpm_invert(x0, scheduler, num_inference_steps=12, seed=11, device=device)
    timesteps = scheduler.timesteps.tolist()

    x_t = traj.latents[timesteps[0]]
    for idx, t_int in enumerate(timesteps):
        t_prev_int = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
        stored = traj.transition_noises.get(t_prev_int, torch.zeros_like(x_t))
        x_t = reverse_step_with_oracle_x0(x_t, x0, scheduler, t_int, t_prev_int, stored)

        if t_prev_int >= 0:
            assert torch.allclose(x_t, traj.latents[t_prev_int], atol=1e-5, rtol=1e-5)

    assert torch.allclose(x_t, x0, atol=1e-5, rtol=1e-5)
