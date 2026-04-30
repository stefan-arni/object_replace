"""
Apply art styles to any input image using trained LoRA adapters.
Supports single styles and blending up to 3 styles with custom weights.
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from styles import STYLES

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(device: torch.device) -> StableDiffusionImg2ImgPipeline:
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading SD 1.5 img2img pipeline ({dtype})...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def load_style_loras(pipe: StableDiffusionImg2ImgPipeline, style_keys: list[str]):
    """Load LoRA weights for the given styles (idempotent per adapter name)."""
    loaded = set(pipe.get_list_adapters().get("unet", []))

    for key in style_keys:
        if key in loaded:
            continue
        cfg = STYLES[key]
        lora_path = cfg["lora_dir"]
        if not Path(lora_path).exists():
            raise FileNotFoundError(
                f"LoRA weights not found for '{cfg['display_name']}' at {lora_path}. "
                f"Run training first."
            )
        print(f"Loading LoRA: {cfg['display_name']} from {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name=key)


def set_style_mix(
    pipe: StableDiffusionImg2ImgPipeline,
    styles_and_weights: list[tuple[str, float]],
):
    """Set active LoRA adapters and their blending weights."""
    active = [s for s, _ in styles_and_weights if s]
    weights = [w for s, w in styles_and_weights if s]

    if not active:
        pipe.set_adapters([], adapter_weights=[])
        return

    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]

    load_style_loras(pipe, active)
    pipe.set_adapters(active, adapter_weights=weights)


def build_prompt(styles_and_weights: list[tuple[str, float]]) -> str:
    """Build a combined prompt from active styles."""
    parts = []
    for key, weight in styles_and_weights:
        if key and weight > 0:
            parts.append(STYLES[key]["trigger"])
    return ", ".join(parts) if parts else "a painting"


def stylize_image(
    pipe: StableDiffusionImg2ImgPipeline,
    input_image: Image.Image,
    prompt: str,
    strength: float = 0.65,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: int | None = None,
) -> Image.Image:
    input_image = input_image.convert("RGB")
    target_dim = 512
    min_dim = 256
    w, h = input_image.size

    if max(w, h) > target_dim:
        scale = target_dim / max(w, h)
        w, h = int(w * scale), int(h * scale)
    elif max(w, h) < min_dim:
        scale = min_dim / max(w, h)
        w, h = int(w * scale), int(h * scale)

    w, h = max((w // 8) * 8, 256), max((h // 8) * 8, 256)
    input_image = input_image.resize((w, h), Image.LANCZOS)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=input_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    return result


def main():
    parser = argparse.ArgumentParser(description="Apply art style(s) to an image")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--styles", type=str, nargs="+", default=["van_gogh"],
        help="Style(s) to apply, e.g. --styles van_gogh monet",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help="Weights for each style (auto-normalized to 100%%)",
    )
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if args.output is None:
        suffix = "_".join(args.styles)
        args.output = f"{input_path.stem}_{suffix}.png"

    weights = args.weights or [1.0] * len(args.styles)
    if len(weights) != len(args.styles):
        raise ValueError("Number of --weights must match number of --styles")

    styles_and_weights = list(zip(args.styles, weights))

    device = get_device()
    pipe = load_pipeline(device)
    set_style_mix(pipe, styles_and_weights)
    prompt = build_prompt(styles_and_weights)

    print(f"Styles: {styles_and_weights}")
    print(f"Prompt: {prompt}")
    print(f"Processing {input_path}...")

    input_image = Image.open(input_path).convert("RGB")
    result = stylize_image(
        pipe, input_image,
        prompt=prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    result.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
