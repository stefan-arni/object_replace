"""Three metrics for the report:

  reconstruction_lpips      -- inversion sanity (proposal target: < 0.05)
  clip_directional_similarity -- did the edit go in the right semantic direction?
                               (proposal target: > 0.20)
  background_lpips          -- how much did we disturb outside the mask?
                               (proposal target: < 0.10)

LPIPS uses AlexNet (small, fast, downloads on first call). CLIP uses
open_clip's ViT-B-32 with the OpenAI weights -- standard for P2P-family work.
Both models are cached at module level so repeated calls in a sweep don't pay
the load cost.
"""
import lpips
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

_lpips_model = None
_clip_model = None
_clip_tokenizer = None
_clip_preprocess = None


def _device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_lpips(device):
    global _lpips_model
    if _lpips_model is None or next(_lpips_model.parameters()).device.type != device:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _lpips_model


def _get_clip(device):
    global _clip_model, _clip_tokenizer, _clip_preprocess
    if _clip_model is None:
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_tokenizer, _clip_preprocess


def _pil_to_lpips_input(img: Image.Image, device, size: int = 256) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((size, size))).astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def reconstruction_lpips(src_img: Image.Image, recon_img: Image.Image, device=None) -> float:
    device = _device(device)
    model = _get_lpips(device)
    a = _pil_to_lpips_input(src_img, device)
    b = _pil_to_lpips_input(recon_img, device)
    return model(a, b).item()


@torch.no_grad()
def background_lpips(
    src_img: Image.Image,
    edit_img: Image.Image,
    mask: torch.Tensor,
    device=None,
) -> float:
    """LPIPS computed only outside the mask. mask: 1 = foreground/edit, 0 = background.
    Accepts (1,1,H,W), (H,W), or arbitrary 2D shapes; resized via nearest to 256."""
    device = _device(device)
    model = _get_lpips(device)

    m = mask.float()
    if m.ndim == 2:
        m = m[None, None]
    elif m.ndim == 3:
        m = m[None]
    m_256 = F.interpolate(m, size=(256, 256), mode="nearest").to(device)
    bg = 1.0 - m_256

    a = _pil_to_lpips_input(src_img, device) * bg
    b = _pil_to_lpips_input(edit_img, device) * bg
    return model(a, b).item()


@torch.no_grad()
def clip_directional_similarity(
    src_img: Image.Image,
    edit_img: Image.Image,
    src_prompt: str,
    tgt_prompt: str,
    device=None,
) -> float:
    device = _device(device)
    model, tokenizer, preprocess = _get_clip(device)

    img_src = preprocess(src_img.convert("RGB")).unsqueeze(0).to(device)
    img_tgt = preprocess(edit_img.convert("RGB")).unsqueeze(0).to(device)
    text_src = tokenizer([src_prompt]).to(device)
    text_tgt = tokenizer([tgt_prompt]).to(device)

    img_feat_src = model.encode_image(img_src).float()
    img_feat_tgt = model.encode_image(img_tgt).float()
    text_feat_src = model.encode_text(text_src).float()
    text_feat_tgt = model.encode_text(text_tgt).float()

    img_delta = img_feat_tgt - img_feat_src
    text_delta = text_feat_tgt - text_feat_src
    img_delta = img_delta / img_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    text_delta = text_delta / text_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (img_delta * text_delta).sum(dim=-1).item()
