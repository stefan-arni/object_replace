import torch

from sd_components import SDComponents


def _alpha_bar(c: SDComponents, t) -> torch.Tensor:
    return c.scheduler.alphas_cumprod.to(c.device)[t]


@torch.no_grad()
def ddim_sample(
    c: SDComponents,
    cond_embeds: torch.Tensor,
    uncond_embeds: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    latents: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Deterministic DDIM (eta = 0) with classifier-free guidance.

    Paper update:
        x_0_pred  = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)
        x_{t-1}   = sqrt(a_{t-1}) * x_0_pred + sqrt(1 - a_{t-1}) * eps
    where a_t = alpha_bar_t (cumulative product of alphas).
    """
    B = cond_embeds.shape[0]
    if latents is None:
        latents = torch.randn(
            (B, c.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=c.device,
            dtype=c.dtype,
        )

    c.scheduler.set_timesteps(num_inference_steps, device=c.device)
    timesteps = c.scheduler.timesteps  # descending
    embed_pair = torch.cat([uncond_embeds, cond_embeds], dim=0)
    final_alpha_bar = c.scheduler.final_alpha_cumprod.to(c.device)

    for i, t in enumerate(timesteps):
        # CFG: one UNet pass on the (uncond, cond) batch of 2.
        x_in = torch.cat([latents, latents], dim=0)
        eps_uncond, eps_cond = c.unet(x_in, t, encoder_hidden_states=embed_pair).sample.chunk(2)
        eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        alpha_bar_t = _alpha_bar(c, t)
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else None
        alpha_bar_prev = _alpha_bar(c, t_prev) if t_prev is not None else final_alpha_bar

        x0_pred = (latents - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
        dir_xt = (1 - alpha_bar_prev).sqrt() * eps_pred
        latents = alpha_bar_prev.sqrt() * x0_pred + dir_xt

    return latents
