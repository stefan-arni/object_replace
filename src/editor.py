"""Editor: invert a real image, then re-sample target prompt with attention control.

The 'how' is in attention_store.py (the controller that intercepts cross-attn).
The 'where' is in masks.py (Step 8). This file is just orchestration.
"""
import numpy as np
import torch
from PIL import Image

from attention_store import (
    AttentionController,
    P2PReplaceController,
    infer_preserved_token_indices,
    install_controller,
    uninstall_controller,
)
from ddim import _alpha_bar
from inversion import null_text_inversion
from sd_components import SDComponents, decode_latents, encode_prompt


def _to_pil(t: torch.Tensor) -> Image.Image:
    arr = ((t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() + 1) / 2 * 255)
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


class Editor:
    def __init__(self, components: SDComponents):
        self.c = components

    def edit(
        self,
        image: Image.Image,
        source_prompt: str,
        target_prompt: str,
        *,
        controller: AttentionController | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        mask_mode: str = "none",
        inversion_inner_steps: int = 10,
        tau: float = 0.8,
    ) -> Image.Image:
        if mask_mode not in ("none",):
            raise NotImplementedError(f"mask_mode={mask_mode!r} comes in step 8")

        nt = null_text_inversion(
            self.c, image, source_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            inner_steps=inversion_inner_steps,
        )

        source_cond = encode_prompt(self.c, source_prompt)
        target_cond = encode_prompt(self.c, target_prompt)

        if controller is None:
            preserved = infer_preserved_token_indices(self.c.tokenizer, source_prompt, target_prompt)
            controller = P2PReplaceController(
                total_steps=num_inference_steps,
                preserved_token_indices=preserved,
                tau=tau,
            )

        install_controller(self.c.unet, controller)

        z_src = nt.z_T.to(self.c.device, self.c.dtype).clone()
        z_tgt = z_src.clone()

        self.c.scheduler.set_timesteps(num_inference_steps, device=self.c.device)
        timesteps = self.c.scheduler.timesteps
        final_alpha_bar = self.c.scheduler.final_alpha_cumprod.to(self.c.device)

        try:
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    controller.cur_t = int(t)
                    controller.cur_step = i

                    null_t = nt.null_embeds[i].to(self.c.device, self.c.dtype)
                    embeds = torch.cat([null_t, source_cond, null_t, target_cond], dim=0)
                    x_in = torch.cat([z_src, z_src, z_tgt, z_tgt], dim=0)

                    eps_su, eps_sc, eps_tu, eps_tc = self.c.unet(
                        x_in, t, encoder_hidden_states=embeds
                    ).sample.chunk(4)

                    eps_src = eps_su + guidance_scale * (eps_sc - eps_su)
                    eps_tgt = eps_tu + guidance_scale * (eps_tc - eps_tu)

                    a_t = _alpha_bar(self.c, t)
                    a_prev = _alpha_bar(self.c, timesteps[i + 1]) if i + 1 < len(timesteps) else final_alpha_bar

                    x0_src = (z_src - (1 - a_t).sqrt() * eps_src) / a_t.sqrt()
                    z_src = a_prev.sqrt() * x0_src + (1 - a_prev).sqrt() * eps_src

                    x0_tgt = (z_tgt - (1 - a_t).sqrt() * eps_tgt) / a_t.sqrt()
                    z_tgt = a_prev.sqrt() * x0_tgt + (1 - a_prev).sqrt() * eps_tgt
        finally:
            uninstall_controller(self.c.unet)

        return _to_pil(decode_latents(self.c, z_tgt).clamp(-1, 1))
