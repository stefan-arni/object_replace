"""Cross-attention intercept for the SD UNet.

Diffusers' Attention modules expose a `processor` attribute we can swap. We
install a custom processor that runs the standard attention math but calls
into a `Controller` right after softmax. The controller decides whether to
store the maps, return them unchanged, or inject precomputed ones.

Naming convention in SD 1.5 UNet:
    *.attn1 -> self-attention
    *.attn2 -> cross-attention (encoder_hidden_states = text embeddings)

Attention shape after diffusers' head_to_batch_dim:
    (batch_size * num_heads, query_seq_len, key_seq_len)
For cross-attn at resolution R: (B*H, R*R, 77).
"""
import torch

from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

from schedules import ScheduleSet


class AttentionController:
    """Base controller. Default behavior is pass-through. Subclass to store/inject."""

    def __init__(self):
        # The Editor sets these before every UNet call.
        self.cur_t: int | None = None      # raw timestep value
        self.cur_step: int | None = None   # 0-indexed sampling step

    def __call__(self, attn_probs: torch.Tensor, layer_name: str, is_cross: bool) -> torch.Tensor:
        return attn_probs


class StoreController(AttentionController):
    """Captures attention maps. Cross-attn only by default; flip `store_self` for both."""

    def __init__(self, store_self: bool = False, on_device: bool = False):
        super().__init__()
        self.store_self = store_self
        self.on_device = on_device
        self.maps: dict[tuple[int, str], torch.Tensor] = {}

    def __call__(self, attn_probs, layer_name, is_cross):
        if is_cross or self.store_self:
            assert self.cur_t is not None, "controller.cur_t must be set before the UNet call"
            saved = attn_probs.detach()
            if not self.on_device:
                saved = saved.cpu()
            self.maps[(int(self.cur_t), layer_name)] = saved
        return attn_probs


class HookedAttnProcessor:
    """Diffusers AttnProcessor variant that calls a controller after softmax.

    This is essentially diffusers.AttnProcessor reimplemented inline. We can't
    use AttnProcessor2_0 (the sdpa-backed default) because it never materializes
    attention_probs as a tensor we can grab.
    """

    def __init__(self, controller: AttentionController, layer_name: str):
        self.controller = controller
        self.layer_name = layer_name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        batch_size, seq_len, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # ===== controller hook =====
        attention_probs = self.controller(attention_probs, self.layer_name, is_cross)
        # ===========================
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


class P2PReplaceController(AttentionController):
    """Vanilla Prompt-to-Prompt 'replace' mode.

    Assumes the editor batches the UNet call as
        [src_uncond, src_cond, tgt_uncond, tgt_cond]
    For cross-attn columns at `preserved_token_indices` (positions where source
    and target tokenize identically), replace target's attention with source's
    until step `tau * total_steps`. Tokens that differ keep target's own attn,
    which is how the swap actually happens.
    """

    def __init__(self, total_steps: int, preserved_token_indices: list[int], tau: float = 0.8):
        super().__init__()
        self.total_steps = total_steps
        self.preserved = preserved_token_indices
        self.cutoff_step = int(tau * total_steps)

    def __call__(self, attn_probs, layer_name, is_cross):
        if not is_cross or self.cur_step is None or self.cur_step >= self.cutoff_step:
            return attn_probs
        if not self.preserved:
            return attn_probs

        BH = attn_probs.shape[0]
        assert BH % 4 == 0, f"P2PReplaceController expects batch=4 arrangement, got BH={BH}"
        H = BH // 4
        src_u, src_c, tgt_u, tgt_c = attn_probs.split(H, dim=0)

        idx = torch.tensor(self.preserved, device=attn_probs.device)
        tgt_u = tgt_u.clone()
        tgt_c = tgt_c.clone()
        tgt_u[:, :, idx] = src_u[:, :, idx]
        tgt_c[:, :, idx] = src_c[:, :, idx]

        return torch.cat([src_u, src_c, tgt_u, tgt_c], dim=0)


class ScheduleController(AttentionController):
    """The novel-schedule controller. For each cross-attn call, looks up the
    swap weight per token (via ScheduleSet + role map) and blends source's
    cross-attn into target's column-by-column.

    Same batch arrangement assumption as P2PReplaceController:
        [src_uncond, src_cond, tgt_uncond, tgt_cond]
    """

    def __init__(self, schedule_set: ScheduleSet, total_steps: int, token_roles: dict[int, str]):
        super().__init__()
        self.schedule = schedule_set
        self.total_steps = total_steps
        self.roles = token_roles
        self._weights_cache: torch.Tensor | None = None
        self._cached_step: int | None = None

    def _weights_for_step(self, L: int, device, dtype) -> torch.Tensor:
        if self._cached_step == self.cur_step and self._weights_cache is not None:
            return self._weights_cache
        t_frac = self.cur_step / max(self.total_steps - 1, 1)
        w = torch.zeros(L, device=device, dtype=dtype)
        for j in range(L):
            w[j] = self.schedule(t_frac, self.roles.get(j, "context"))
        self._weights_cache = w
        self._cached_step = self.cur_step
        return w

    def __call__(self, attn_probs, layer_name, is_cross):
        if not is_cross or self.cur_step is None:
            return attn_probs

        BH, _, L = attn_probs.shape
        assert BH % 4 == 0, f"ScheduleController expects batch=4 arrangement, got BH={BH}"
        H = BH // 4
        src_u, src_c, tgt_u, tgt_c = attn_probs.split(H, dim=0)

        w = self._weights_for_step(L, attn_probs.device, attn_probs.dtype)  # (L,)
        # Broadcasts as (1, 1, L) over (B*H, HW, L)
        tgt_u_blended = w * src_u + (1 - w) * tgt_u
        tgt_c_blended = w * src_c + (1 - w) * tgt_c

        return torch.cat([src_u, src_c, tgt_u_blended, tgt_c_blended], dim=0)


def classify_token_roles(tokenizer, source_prompt: str, target_prompt: str) -> dict[int, str]:
    """Map token position -> role for use by ScheduleController."""
    max_len = tokenizer.model_max_length
    src = tokenizer(source_prompt, padding="max_length", max_length=max_len,
                    truncation=True, return_tensors="pt").input_ids[0].tolist()
    tgt = tokenizer(target_prompt, padding="max_length", max_length=max_len,
                    truncation=True, return_tensors="pt").input_ids[0].tolist()
    special = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}

    roles: dict[int, str] = {}
    for i in range(max_len):
        if src[i] in special and tgt[i] in special:
            roles[i] = "context"
        elif src[i] != tgt[i]:
            roles[i] = "replaced"
        else:
            roles[i] = "preserved"
    return roles


def infer_preserved_token_indices(tokenizer, source_prompt: str, target_prompt: str) -> list[int]:
    """Indices where source and target tokenize to the same id (after padding to max_len).
    Padding tokens at the tail will match in both, which is fine -- they carry no meaning.
    """
    max_len = tokenizer.model_max_length
    src = tokenizer(source_prompt, padding="max_length", max_length=max_len,
                    truncation=True, return_tensors="pt").input_ids[0]
    tgt = tokenizer(target_prompt, padding="max_length", max_length=max_len,
                    truncation=True, return_tensors="pt").input_ids[0]
    return [i for i in range(max_len) if src[i].item() == tgt[i].item()]


def _attn_modules(unet) -> list[tuple[str, Attention]]:
    return [
        (name, m) for name, m in unet.named_modules()
        if name.endswith(".attn1") or name.endswith(".attn2")
    ]


def install_controller(unet, controller: AttentionController) -> None:
    for name, module in _attn_modules(unet):
        module.set_processor(HookedAttnProcessor(controller, name))


def uninstall_controller(unet) -> None:
    """Restore the default sdpa-backed processor."""
    default = AttnProcessor2_0() if hasattr(torch.nn.functional, "scaled_dot_product_attention") else AttnProcessor()
    for _, module in _attn_modules(unet):
        module.set_processor(default)
