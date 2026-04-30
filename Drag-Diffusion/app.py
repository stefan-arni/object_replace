"""
Gradio UI for object relocation via DDPM noise prior shift.
Run:  python app.py
"""

import numpy as np
from PIL import Image
import gradio as gr

from utils.image_utils import get_device, pil_to_tensor
from pipeline.relocation_pipeline import ObjectRelocationPipeline
from eval.perceptual_loss import VGGPerceptualLoss

# ── Load once at startup ──────────────────────────────────────────────────────
device = get_device()
print("Loading pipeline...")
pipe = ObjectRelocationPipeline(device=device)
loss_fn = VGGPerceptualLoss(device=device)
print("Ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_mask(editor_value, fallback_size=(512, 512)):
    """
    Pull a binary PIL mask out of a gr.ImageEditor value.
    The editor returns {"background": PIL, "layers": [PIL RGBA, ...], "composite": PIL}.
    Painted pixels have non-zero alpha in the first layer.
    """
    if editor_value is None:
        return Image.new("L", fallback_size, 0)

    layers = editor_value.get("layers") or []
    if not layers or layers[0] is None:
        return Image.new("L", fallback_size, 0)

    layer = layers[0]
    if layer.mode == "RGBA":
        # Alpha > 0 wherever the user drew
        mask = layer.split()[3]
    else:
        mask = layer.convert("L")

    return mask


def set_image_on_editors(image):
    """When an image is uploaded, load it into both mask editors."""
    if image is None:
        return gr.update(), gr.update()
    blank = {"background": image, "layers": [], "composite": image}
    return blank, blank


def run(image, src_editor, tgt_editor, prompt,
        use_noise_shift, seed, steps, strength, cfg):

    if image is None:
        return None, None, None, "⚠️ Upload an image first."
    if not prompt.strip():
        return None, None, None, "⚠️ Enter a prompt describing the final scene."

    size = image.size
    src_mask = extract_mask(src_editor, size)
    tgt_mask = extract_mask(tgt_editor, size)

    if np.array(src_mask).sum() == 0:
        return None, None, None, "⚠️ Draw a source mask (paint over the object to move)."
    if np.array(tgt_mask).sum() == 0:
        return None, None, None, "⚠️ Draw a target mask (paint where to move the object)."

    result, composite = pipe(
        image, prompt, src_mask, tgt_mask,
        use_noise_shift=use_noise_shift,
        seed=int(seed),
        num_inference_steps=int(steps),
        sdedit_strength=float(strength),
        guidance_scale=float(cfg),
    )

    # Perceptual score on the object region
    src_t = pil_to_tensor(image.resize((256, 256)), device)
    res_t = pil_to_tensor(result.resize((256, 256)), device)
    score = loss_fn(src_t, res_t)
    label = f"Perceptual distance: {score:.4f}  (lower = more texture preserved)"

    return composite, result, label


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Object Relocator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## Object Relocation · DDPM Noise Prior Shift\n"
        "Upload a photo, paint masks, enter a prompt, click **Run**."
    )

    with gr.Row():
        # ── Left: inputs ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            image_input = gr.Image(
                label="Step 1 — Upload image",
                type="pil",
                height=300,
            )

            with gr.Row():
                src_editor = gr.ImageEditor(
                    label="Step 2 — Paint SOURCE mask (the object to move)",
                    type="pil",
                    layers=True,
                    height=300,
                )
                tgt_editor = gr.ImageEditor(
                    label="Step 3 — Paint TARGET mask (where to move it)",
                    type="pil",
                    layers=True,
                    height=300,
                )

            image_input.change(
                fn=set_image_on_editors,
                inputs=image_input,
                outputs=[src_editor, tgt_editor],
            )

        # ── Right: controls + outputs ─────────────────────────────────────────
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Step 4 — Prompt (describe the scene with object at its new location)",
                placeholder='e.g. "a cat sitting on the right side of the grass"',
                lines=2,
            )

            with gr.Row():
                use_shift = gr.Checkbox(
                    label="DDPM noise shift (our contribution)",
                    value=True,
                    info="Uncheck for the SDEdit baseline",
                )
                seed_input = gr.Number(label="Seed", value=42, precision=0, minimum=0)

            with gr.Accordion("Advanced parameters", open=False):
                steps_slider = gr.Slider(
                    10, 50, value=30, step=5,
                    label="Inference steps (30 = fast, 50 = quality)",
                )
                strength_slider = gr.Slider(
                    0.1, 0.9, value=0.3, step=0.05,
                    label="SDEdit strength (0.3 = faithful, 0.7 = creative)",
                )
                cfg_slider = gr.Slider(
                    1.0, 15.0, value=7.5, step=0.5,
                    label="Guidance scale (CFG)",
                )

            run_btn = gr.Button("▶  Run", variant="primary", size="lg")
            score_box = gr.Textbox(label="Score", interactive=False)

            composite_out = gr.Image(label="Composite (copy-paste before harmonization)", type="pil")
            result_out = gr.Image(label="Result", type="pil")

    run_btn.click(
        fn=run,
        inputs=[
            image_input, src_editor, tgt_editor, prompt_input,
            use_shift, seed_input, steps_slider, strength_slider, cfg_slider,
        ],
        outputs=[composite_out, result_out, score_box],
    )

    gr.Markdown(
        "**Tip:** Start with `SDEdit strength ≈ 0.5`. "
        "Lower strength → more faithful to the copy-paste; "
        "higher → smoother harmonization but may lose object texture. "
        "Toggle the noise-shift checkbox to compare methods."
    )


if __name__ == "__main__":
    demo.launch(share=False)  # set share=True to get a public URL on Colab
