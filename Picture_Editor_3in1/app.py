"""
Gradio web UI for multi-style art transfer.
Mix up to 3 art styles with custom weights.
"""

import torch
import gradio as gr
from PIL import Image
from pathlib import Path

from styles import STYLES, DISPLAY_NAMES
from inference import (
    load_pipeline, set_style_mix, build_prompt,
    stylize_image, get_device,
)

pipe = None
NONE_LABEL = "None"
STYLE_CHOICES = [NONE_LABEL] + [STYLES[k]["display_name"] for k in STYLES]
DISPLAY_TO_KEY = {v["display_name"]: k for k, v in STYLES.items()}


def ensure_pipeline():
    global pipe
    if pipe is None:
        device = get_device()
        pipe = load_pipeline(device)
    return pipe


def get_available_styles():
    """Return list of styles that have trained LoRA weights."""
    available = []
    for key, cfg in STYLES.items():
        path = Path(cfg["lora_dir"]) / "pytorch_lora_weights.safetensors"
        if path.exists():
            available.append(cfg["display_name"])
    return available


def on_upload(image):
    if image is None:
        return ""
    w, h = image.size
    if max(w, h) > 512:
        return f"**{w} x {h} px** — Will be resized to 512px max."
    elif max(w, h) < 256:
        scale = 256 / max(w, h)
        nw = max((int(w * scale) // 8) * 8, 256)
        nh = max((int(h * scale) // 8) * 8, 256)
        return f"**{w} x {h} px** — Will be upscaled to **{nw} x {nh} px**."
    return f"**{w} x {h} px** — Good size, ready to transform."


def update_total(w1, w2, w3, s1, s2, s3):
    """Show the current total weight and which styles are active."""
    total = 0
    if s1 != NONE_LABEL:
        total += w1
    if s2 != NONE_LABEL:
        total += w2
    if s3 != NONE_LABEL:
        total += w3

    if total == 0:
        return "Select at least one style."

    parts = []
    if s1 != NONE_LABEL:
        parts.append(f"{s1} {w1:.0f}%")
    if s2 != NONE_LABEL:
        parts.append(f"{s2} {w2:.0f}%")
    if s3 != NONE_LABEL:
        parts.append(f"{s3} {w3:.0f}%")

    mix_str = " + ".join(parts)

    if abs(total - 100) < 1:
        return f"**Mix:** {mix_str} = **100%**"
    return f"**Mix:** {mix_str} = **{total:.0f}%** (will be normalized to 100%)"


def process_image(
    input_image,
    style1, weight1,
    style2, weight2,
    style3, weight3,
    strength, guidance_scale, num_steps, seed,
):
    if input_image is None:
        return None

    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    input_image = input_image.convert("RGB")

    styles_and_weights = []
    for style_name, weight in [(style1, weight1), (style2, weight2), (style3, weight3)]:
        if style_name != NONE_LABEL and weight > 0:
            key = DISPLAY_TO_KEY.get(style_name)
            if key:
                styles_and_weights.append((key, weight))

    if not styles_and_weights:
        raise gr.Error("Select at least one style.")

    available = get_available_styles()
    for key, _ in styles_and_weights:
        name = STYLES[key]["display_name"]
        if name not in available:
            raise gr.Error(f"'{name}' LoRA not trained yet. Run train_all.py first.")

    pipeline = ensure_pipeline()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    set_style_mix(pipeline, styles_and_weights)
    prompt = build_prompt(styles_and_weights)

    result = stylize_image(
        pipeline, input_image,
        prompt=prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_steps),
        seed=int(seed) if seed >= 0 else None,
    )

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def build_ui():
    available = get_available_styles()
    status_lines = []
    for key, cfg in STYLES.items():
        name = cfg["display_name"]
        tag = "ready" if name in available else "not trained"
        status_lines.append(f"- **{name}**: {tag}")
    status_md = "\n".join(status_lines)

    with gr.Blocks(
        title="Multi-Style Art Transfer",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Multi-Style Art Transfer\n"
            "Upload any image and transform it using up to **3 blended art styles**.\n\n"
            f"**Available styles:**\n{status_md}"
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                size_info = gr.Markdown("")

                gr.Markdown("### Style Mix")

                with gr.Row():
                    style1 = gr.Dropdown(
                        choices=STYLE_CHOICES, value=STYLE_CHOICES[1] if len(STYLE_CHOICES) > 1 else NONE_LABEL,
                        label="Style 1",
                    )
                    weight1 = gr.Slider(minimum=0, maximum=100, value=100, step=5, label="Weight %")

                with gr.Row():
                    style2 = gr.Dropdown(choices=STYLE_CHOICES, value=NONE_LABEL, label="Style 2")
                    weight2 = gr.Slider(minimum=0, maximum=100, value=0, step=5, label="Weight %")

                with gr.Row():
                    style3 = gr.Dropdown(choices=STYLE_CHOICES, value=NONE_LABEL, label="Style 3")
                    weight3 = gr.Slider(minimum=0, maximum=100, value=0, step=5, label="Weight %")

                mix_info = gr.Markdown("**Mix:** Van Gogh 100% = **100%**")

                gr.Markdown("### Settings")
                strength = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.65, step=0.05,
                    label="Style Strength",
                    info="Higher = more stylized, less original detail",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                    label="Guidance Scale",
                )
                num_steps = gr.Slider(
                    minimum=10, maximum=50, value=30, step=5,
                    label="Inference Steps",
                )
                seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)

                submit_btn = gr.Button("Transform", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Styled Result")

        input_image.change(fn=on_upload, inputs=[input_image], outputs=[size_info])

        mix_inputs = [weight1, weight2, weight3, style1, style2, style3]
        for component in mix_inputs:
            component.change(fn=update_total, inputs=mix_inputs, outputs=[mix_info])

        submit_btn.click(
            fn=process_image,
            inputs=[
                input_image,
                style1, weight1, style2, weight2, style3, weight3,
                strength, guidance_scale, num_steps, seed,
            ],
            outputs=output_image,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)
