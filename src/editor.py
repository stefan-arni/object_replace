"""Editor: invert a real image, then re-sample target prompt with attention control.

The 'how' is in attention_store.py (the controller that intercepts cross-attn).
The 'where' is in masks.py (Step 8). This file is just orchestration.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from attention_store import (
    AttentionController,
    P2PReplaceController,
    ScheduleController,
    StoreController,
    classify_token_roles,
    infer_preserved_token_indices,
    install_controller,
    uninstall_controller,
)
from ddim import _alpha_bar
from inpaint import blended_inpaint, derive_background_prompt
from inversion import null_text_inversion
from masks import derive_attention_mask, derive_target_mask
from schedules import ScheduleSet
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
        schedule: ScheduleSet | None = None,
        controller: AttentionController | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        mask_mode: str = "none",
        inversion_inner_steps: int = 10,
        tau: float = 0.8,
        return_mask: bool = False,
        precomputed_mask: torch.Tensor | None = None,
        mask_blend_until_frac: float = 0.7,
        pixel_composite: bool = True,
        composite_mode: str = "strict",
        background_prompt: str | None = None,
    ) -> Image.Image | tuple[Image.Image, torch.Tensor | None]:
        if mask_mode not in ("none", "attention"):
            raise ValueError(f"unknown mask_mode={mask_mode!r}")
        if composite_mode not in ("strict", "inpaint"):
            raise ValueError(f"unknown composite_mode={composite_mode!r}")
        if composite_mode == "inpaint" and mask_mode != "attention":
            raise ValueError("composite_mode='inpaint' requires mask_mode='attention'")

        nt = null_text_inversion(
            self.c, image, source_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            inner_steps=inversion_inner_steps,
        )

        source_cond = encode_prompt(self.c, source_prompt)
        target_cond = encode_prompt(self.c, target_prompt)
        roles = classify_token_roles(self.c.tokenizer, source_prompt, target_prompt)
        replaced_indices = [j for j, r in roles.items() if r == "replaced"]

        if controller is None:
            if schedule is not None:
                controller = ScheduleController(
                    schedule_set=schedule,
                    total_steps=num_inference_steps,
                    token_roles=roles,
                    capture_target_attn=(composite_mode == "inpaint"),
                )
            else:
                preserved = infer_preserved_token_indices(self.c.tokenizer, source_prompt, target_prompt)
                controller = P2PReplaceController(
                    total_steps=num_inference_steps,
                    preserved_token_indices=preserved,
                    tau=tau,
                )
                if composite_mode == "inpaint":
                    raise ValueError("composite_mode='inpaint' requires a `schedule` (uses ScheduleController for target-attn capture)")

        # Scout pass to derive a mask, if requested. Skipped if caller already
        # has one (the ablation runner derives once per image and reuses it
        # across schedules to keep bg_lpips comparable).
        mask = precomputed_mask
        if mask_mode == "attention" and mask is None:
            replaced_indices = [j for j, r in roles.items() if r == "replaced"]
            mask = self._scout_mask(
                nt, source_cond, num_inference_steps, guidance_scale, replaced_indices,
            )

        install_controller(self.c.unet, controller)

        z_src = nt.z_T.to(self.c.device, self.c.dtype).clone()
        z_tgt = z_src.clone()
        mask_dev = mask.to(self.c.device, self.c.dtype) if mask is not None else None

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

                    if mask_dev is not None:
                        # Late-step skip: let the last few steps run free so fine
                        # texture isn't constrained to fit source's high-freq layout.
                        t_frac = i / max(num_inference_steps - 1, 1)
                        if t_frac < mask_blend_until_frac:
                            z_tgt = mask_dev * z_tgt + (1 - mask_dev) * z_src
        finally:
            uninstall_controller(self.c.unet)

        edit_pixels = decode_latents(self.c, z_tgt).clamp(-1, 1)

        # Source pixels at 512x512 (used by both composite paths).
        src_arr = np.asarray(image.convert("RGB").resize((512, 512))).astype("float32") / 127.5 - 1.0
        src_pixels = torch.from_numpy(src_arr).permute(2, 0, 1).unsqueeze(0).to(self.c.device, self.c.dtype)

        if composite_mode == "inpaint" and mask is not None:
            # Three-region composite for shape-mismatch edits:
            #   outside source_mask : source pixels (untouched)
            #   inside  target_mask : edit pixels (the new object as it actually rendered)
            #   inside  diff_mask   : separately inpainted background
            self.c.scheduler.set_timesteps(num_inference_steps, device=self.c.device)
            timesteps_list = self.c.scheduler.timesteps.tolist()
            target_mask = derive_target_mask(
                controller.target_maps,
                replaced_token_indices=replaced_indices,
                timesteps_desc_list=timesteps_list,
            )
            src_mask_512 = F.interpolate(mask.float(), size=(512, 512),
                                          mode="bilinear", align_corners=False).to(self.c.device, self.c.dtype)
            tgt_mask_512 = F.interpolate(target_mask.float(), size=(512, 512),
                                          mode="bilinear", align_corners=False).to(self.c.device, self.c.dtype)
            tgt_mask_512 = torch.minimum(tgt_mask_512, src_mask_512)  # don't paint mouse outside firetruck
            diff_mask_512 = torch.clamp(src_mask_512 - tgt_mask_512, 0.0, 1.0)

            if diff_mask_512.mean().item() > 0.005:
                if background_prompt is None:
                    src_word = self._first_replaced_word(source_prompt, replaced_indices)
                    bg_prompt = derive_background_prompt(source_prompt, src_word)
                else:
                    bg_prompt = background_prompt
                diff_mask_64 = F.interpolate(diff_mask_512[:, :1].float(), size=(64, 64),
                                             mode="bilinear", align_corners=False)
                inpainted = blended_inpaint(
                    self.c, image, diff_mask_64, bg_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
            else:
                # Diff region too small to bother inpainting -- fall back to source pixels there.
                inpainted = src_pixels

            final = (
                (1 - src_mask_512) * src_pixels
                + tgt_mask_512 * edit_pixels
                + diff_mask_512 * inpainted
            )
            img = _to_pil(final)
            if return_mask:
                return img, mask
            return img

        if pixel_composite and mask is not None:
            # Strict mode: outside the mask = exact source pixels.
            pixel_mask = F.interpolate(
                mask.float(), size=(512, 512), mode="bilinear", align_corners=False,
            ).to(self.c.device, self.c.dtype)
            edit_pixels = pixel_mask * edit_pixels + (1 - pixel_mask) * src_pixels

        img = _to_pil(edit_pixels)
        if return_mask:
            return img, mask
        return img

    def _first_replaced_word(self, source_prompt: str, replaced_indices: list[int]) -> str:
        """Decode the first non-special replaced token back to a word."""
        max_len = self.c.tokenizer.model_max_length
        ids = self.c.tokenizer(
            source_prompt, padding="max_length", max_length=max_len,
            truncation=True, return_tensors="pt",
        ).input_ids[0]
        special = {self.c.tokenizer.bos_token_id, self.c.tokenizer.eos_token_id, self.c.tokenizer.pad_token_id}
        for j in replaced_indices:
            tid = ids[j].item()
            if tid in special:
                continue
            tok = self.c.tokenizer.decode([tid]).strip()
            if tok:
                return tok
        return "object"

    def derive_mask(
        self,
        image: Image.Image,
        source_prompt: str,
        target_prompt: str,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        inversion_inner_steps: int = 10,
    ) -> torch.Tensor:
        """Derive the attention mask without doing a full edit. Useful when the
        caller wants one mask per image to evaluate background-LPIPS fairly
        across multiple schedules."""
        nt = null_text_inversion(
            self.c, image, source_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            inner_steps=inversion_inner_steps,
        )
        source_cond = encode_prompt(self.c, source_prompt)
        roles = classify_token_roles(self.c.tokenizer, source_prompt, target_prompt)
        replaced_indices = [j for j, r in roles.items() if r == "replaced"]
        return self._scout_mask(nt, source_cond, num_inference_steps, guidance_scale, replaced_indices)

    def _scout_mask(
        self,
        nt,
        source_cond: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        replaced_indices: list[int],
    ) -> torch.Tensor:
        """Source-only sampling pass with attention capture, then derive the mask."""
        store = StoreController(store_self=False)
        install_controller(self.c.unet, store)

        self.c.scheduler.set_timesteps(num_inference_steps, device=self.c.device)
        timesteps = self.c.scheduler.timesteps
        final_alpha_bar = self.c.scheduler.final_alpha_cumprod.to(self.c.device)

        z = nt.z_T.to(self.c.device, self.c.dtype).clone()

        try:
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    store.cur_t = int(t)
                    store.cur_step = i

                    null_t = nt.null_embeds[i].to(self.c.device, self.c.dtype)
                    embeds = torch.cat([null_t, source_cond], dim=0)
                    x_in = torch.cat([z, z], dim=0)
                    eps_u, eps_c = self.c.unet(x_in, t, encoder_hidden_states=embeds).sample.chunk(2)
                    eps = eps_u + guidance_scale * (eps_c - eps_u)

                    a_t = _alpha_bar(self.c, t)
                    a_prev = _alpha_bar(self.c, timesteps[i + 1]) if i + 1 < len(timesteps) else final_alpha_bar
                    x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
                    z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        finally:
            uninstall_controller(self.c.unet)

        return derive_attention_mask(
            store.maps,
            source_token_indices=replaced_indices,
            timesteps_desc_list=timesteps.tolist(),
            batch_size=2,
            use_sample_index=1,  # cond pass under CFG
        )
