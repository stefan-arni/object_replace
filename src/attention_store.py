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


class AttentionController:
    """Base controller. Default behavior is pass-through. Subclass to store/inject."""

    def __init__(self):
        # The Editor sets this before every UNet call so the controller can key by timestep.
        self.cur_t: int | None = None

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
