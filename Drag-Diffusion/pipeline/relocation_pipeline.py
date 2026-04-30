import torch
import numpy as np
from PIL import Image, ImageFilter
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionInpaintPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer

from utils.image_utils import get_dtype, pil_to_tensor, tensor_to_pil, encode_image, decode_latent, create_composite
from utils.mask_utils import prepare_latent_mask, gaussian_blur_mask
from inversion.ddpm_inversion import ddpm_invert, reconstruct_xt
from noise_shift.noise_shift import shift_all_noise_maps


MODEL_ID = "sd2-community/stable-diffusion-2-1-base"
INPAINT_MODEL_ID = "sd2-community/stable-diffusion-2-inpainting"


class ObjectRelocationPipeline:
    """
    Texture-preserving object relocation via DDPM noise prior shift.

    Pipeline:
      1. Pixel-space copy-paste → composite image (object at target, source filled)
      2. DDPM inversion of original → reconstruction-consistent latent trajectory
      3. (ours) Shift noise maps source→target (InstructUDrag Eq. 4)
      4. SDEdit from the composite with background and source cleanup locks

    Ablation: use_noise_shift=False keeps the same inverted trajectory without
    the source→target shift. Lower perceptual distance = better texture preservation.

    Novel contribution: DDPM inversion + noise shift (InstructUDrag 2024, Eq. 4).
    Base model: SD 2.1 (Rombach et al., 2022).
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        inpaint_model_id: str = INPAINT_MODEL_ID,
        device: torch.device = None,
        local_files_only: bool = False,
    ):
        if device is None:
            from utils.image_utils import get_device
            device = get_device()
        self.device = device
        self.model_id = model_id
        self.inpaint_model_id = inpaint_model_id
        self.local_files_only = local_files_only
        dtype = get_dtype(device)
        self.dtype = dtype

        print(f"Loading SD 2.1 on {device} ({dtype})...")
        kw = dict(local_files_only=local_files_only)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **kw).to(device=device, dtype=dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **kw)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **kw).to(device=device, dtype=dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **kw).to(device=device, dtype=dtype)
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", **kw)
        self.prediction_type = self.scheduler.config.prediction_type
        # Cap at 512 on MPS to avoid OOM; Colab (CUDA) uses the native 768
        native_size = self.unet.config.sample_size * 8
        self.image_size = 512 if device.type == "mps" else native_size
        self.latent_size = self.image_size // 8
        self.inpaint_pipe = None
        print(f"All components loaded. Prediction type: {self.prediction_type}, image size: {self.image_size}")

    def _make_generator(self, seed: int) -> torch.Generator:
        gen_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        return torch.Generator(device=gen_device).manual_seed(seed)

    def _load_inpaint_pipe(self):
        if self.inpaint_pipe is not None:
            return self.inpaint_pipe

        print(f"Loading inpainting model on {self.device} ({self.dtype})...")
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.inpaint_model_id,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self.inpaint_pipe.set_progress_bar_config(disable=True)
        if hasattr(self.inpaint_pipe, "enable_attention_slicing"):
            self.inpaint_pipe.enable_attention_slicing()
        return self.inpaint_pipe

    def _masks_are_effectively_same(
        self,
        source_mask: Image.Image,
        target_mask: Image.Image,
    ) -> bool:
        src = np.array(source_mask.convert("L")) > 127
        tgt = np.array(target_mask.convert("L")) > 127
        union = np.logical_or(src, tgt).sum()
        if union == 0:
            return True
        iou = np.logical_and(src, tgt).sum() / union
        return iou > 0.98

    def _restore_background(
        self,
        original: Image.Image,
        edited: Image.Image,
        source_mask: Image.Image,
        target_mask: Image.Image,
        feather_sigma: float,
    ) -> Image.Image:
        orig_arr = np.array(original.convert("RGB")).astype(np.float32)
        edited_arr = np.array(edited.convert("RGB")).astype(np.float32)

        blur_r = max(1, int(feather_sigma * 6))
        src_soft = np.array(
            source_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_r))
        ).astype(np.float32) / 255.0
        tgt_soft = np.array(
            target_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_r))
        ).astype(np.float32) / 255.0
        keep_orig = ((1.0 - src_soft) * (1.0 - tgt_soft))[..., None]

        merged = edited_arr * (1.0 - keep_orig) + orig_arr * keep_orig
        return Image.fromarray(merged.clip(0, 255).astype(np.uint8))

    def _cleanup_source_with_inpainting(
        self,
        image: Image.Image,
        prompt: str,
        source_mask: Image.Image,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Image.Image:
        pipe = self._load_inpaint_pipe()
        generator = self._make_generator(seed)
        mask = source_mask.convert("L").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        result = pipe(
            prompt=prompt,
            negative_prompt="",
            image=image,
            mask_image=mask,
            num_inference_steps=max(4, num_inference_steps),
            guidance_scale=guidance_scale,
            strength=0.35,
            generator=generator,
        ).images[0]
        return result

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0].float()

    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev_int: int,
        eps_pred: torch.Tensor,
        stored_noise: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM reverse step, injecting stored (optionally shifted) noise."""
        t_int = t.item()
        abar_t = self.scheduler.alphas_cumprod[t_int].to(device=self.device, dtype=torch.float32)

        # SD 2.1 v-prediction → convert to noise prediction
        if self.prediction_type == "v_prediction":
            eps_pred = abar_t.sqrt() * eps_pred + (1 - abar_t).sqrt() * x_t

        pred_x0 = (x_t - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
        pred_x0 = pred_x0.clamp(-4.0, 4.0)

        if t_prev_int < 0:
            return pred_x0

        abar_prev = self.scheduler.alphas_cumprod[t_prev_int].to(device=self.device, dtype=torch.float32)
        coeff1 = abar_prev.sqrt() * (1 - abar_t / abar_prev) / (1 - abar_t)
        coeff2 = (abar_t / abar_prev).sqrt() * (1 - abar_prev) / (1 - abar_t)
        mu = coeff1 * pred_x0 + coeff2 * x_t

        beta_t = (1 - abar_t / abar_prev).clamp(min=0.0)
        sigma_t = (beta_t * (1 - abar_prev) / (1 - abar_t)).clamp(min=0.0).sqrt()
        return mu + sigma_t * stored_noise

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        source_mask: Image.Image,
        target_mask: Image.Image,
        use_noise_shift: bool = True,
        seed: int = 42,
        num_inference_steps: int = 50,
        sdedit_strength: float = 0.7,
        guidance_scale: float = 7.5,
        feather_sigma: float = 2.0,
    ) -> Image.Image:
        """
        Move the object defined by source_mask to target_mask.

        Pipeline:
          1. Pixel copy-paste → composite (establishes WHERE the object goes)
          2. DDPM inversion of original → a reconstruction-consistent latent trajectory
          3. (ours) Shift the per-step noise maps source→target
          4. SDEdit from composite with RePaint-style locks for background/source cleanup

        use_noise_shift=True → ours; False → baseline without spatial noise shifting.
        """
        device = self.device
        sz = self.image_size

        # 0. Fast path: if nothing moves, preserve the input exactly.
        if self._masks_are_effectively_same(source_mask, target_mask):
            identity = image.convert("RGB").resize((self.image_size, self.image_size))
            return identity.copy(), identity

        # 1. Pixel-space copy-paste
        img_sz = image.resize((sz, sz))
        composite = create_composite(
            img_sz,
            source_mask.resize((sz, sz)),
            target_mask.resize((sz, sz)),
        )

        # 2. Encode original + composite
        vae_dtype = next(self.vae.parameters()).dtype
        x0_orig = encode_image(self.vae, pil_to_tensor(img_sz, device).to(vae_dtype))
        x0_composite = encode_image(self.vae, pil_to_tensor(composite, device).to(vae_dtype))

        # 3. Encode prompts for CFG
        encoder_hs = self._encode_prompt(prompt)
        uncond_hs = self._encode_prompt("")

        # 4. Prepare latent masks and soft locks
        M_src = prepare_latent_mask(source_mask.resize((sz, sz)), device, self.latent_size)
        M_tgt = prepare_latent_mask(target_mask.resize((sz, sz)), device, self.latent_size)
        M_src_soft = gaussian_blur_mask(M_src, sigma=2.0)
        M_tgt_soft = gaussian_blur_mask(M_tgt, sigma=2.0)
        bg_mask = (1.0 - M_src_soft.clamp(0, 1)) * (1.0 - M_tgt_soft.clamp(0, 1))
        M_tgt_lock = gaussian_blur_mask(M_tgt, sigma=2.0)

        # 5. Invert the original once, then optionally shift the stored noise maps
        inversion = ddpm_invert(x0_orig, self.scheduler, num_inference_steps, seed, device)

        if use_noise_shift:
            start_noises = shift_all_noise_maps(
                inversion.marginal_noises, M_src, M_tgt, device, feather_sigma
            )
            transition_noises = shift_all_noise_maps(
                inversion.transition_noises, M_src, M_tgt, device, feather_sigma
            )
        else:
            start_noises = inversion.marginal_noises
            transition_noises = inversion.transition_noises

        # 6. SDEdit start from the composite at the chosen timestep
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        start_idx = max(0, min(int((1.0 - sdedit_strength) * len(timesteps)), len(timesteps) - 1))
        t_start_int = timesteps[start_idx].item()
        abar_start = self.scheduler.alphas_cumprod[t_start_int].to(device=device, dtype=torch.float32)
        start_noise = start_noises[t_start_int]
        x_t = abar_start.sqrt() * x0_composite + (1 - abar_start).sqrt() * start_noise

        # 7. Denoising loop with background and early target locking
        unet_dtype = next(self.unet.parameters()).dtype
        active_timesteps = timesteps[start_idx:]
        lock_cutoff = len(active_timesteps) // 2

        for i, t in enumerate(active_timesteps):
            t_global_idx = start_idx + i
            t_prev_int = timesteps[t_global_idx + 1].item() if t_global_idx + 1 < len(timesteps) else -1

            with torch.no_grad():
                x_t_input = x_t.to(unet_dtype)
                t_batch = t.unsqueeze(0).to(device)
                latent_batch = torch.cat([x_t_input, x_t_input], dim=0)
                cond_batch = torch.cat([uncond_hs, encoder_hs], dim=0).to(unet_dtype)
                noise_pred = self.unet(latent_batch, t_batch.repeat(2), cond_batch).sample.float()
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                eps_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            stored = transition_noises.get(t_prev_int, torch.zeros_like(x_t))
            x_t = self._ddpm_step(x_t, t, t_prev_int, eps_pred, stored)

            if t_prev_int >= 0:
                x_t_orig = inversion.latents[t_prev_int]
                x_t = x_t * (1.0 - bg_mask) + x_t_orig * bg_mask

                if i < lock_cutoff:
                    x_t_comp = reconstruct_xt(
                        x0_composite,
                        t_prev_int,
                        start_noises[t_prev_int],
                        self.scheduler,
                        device,
                    )
                    x_t = x_t * (1.0 - M_tgt_lock) + x_t_comp * M_tgt_lock

        # 8. Decode the drag result, then use a true inpainting model to repair the source hole.
        decoded = decode_latent(self.vae, x_t.to(vae_dtype))
        first_pass = tensor_to_pil(decoded)
        cleaned = self._cleanup_source_with_inpainting(
            first_pass,
            prompt,
            source_mask.resize((sz, sz)),
            seed + 1,
            num_inference_steps,
            guidance_scale,
        )

        # 9. Restore untouched background pixels exactly in image space.
        final = self._restore_background(
            img_sz,
            cleaned,
            source_mask.resize((sz, sz)),
            target_mask.resize((sz, sz)),
            feather_sigma,
        )
        return final, composite
