"""Unified Gradio app for the 3-person CS5788 generative-models project.

Three operations on a real photograph:
  1. Object Replacement (Stefan)  -- this repo's Editor
  2. Repositioning      (teammate) -- swap in their function below
  3. Style Transfer     (teammate) -- swap in their function below

Run locally:
    cd platform && python app.py

Or deploy to HuggingFace Spaces:
    gradio deploy
"""
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import get_editor  # loads SD lazily on first call

SCHEDULES = {
    "vanilla P2P (baseline)":  "vanilla_p2p",
    "linear decay (recommended)": "linear_decay",
    "cosine":                  "cosine",
    "constant 0.5 (slow blend)": "constant_05",
    "piecewise":               "piecewise",
}


def _build_schedule(name: str):
    from schedules import (
        constant_replaced, cosine_replaced, linear_decay_replaced,
        piecewise_demo, vanilla_p2p,
    )
    return {
        "vanilla_p2p":  vanilla_p2p(0.8),
        "linear_decay": linear_decay_replaced(),
        "cosine":       cosine_replaced(),
        "constant_05":  constant_replaced(0.5),
        "piecewise":    piecewise_demo(),
    }[name]


# ---------------------------------------------------------------------------
# TAB 1: object replacement (your module)
# ---------------------------------------------------------------------------
def run_replace(image, source_prompt, target_prompt, schedule_name, mask_mode,
                composite_mode, background_prompt):
    if image is None:
        return None, "upload an image first"
    if not source_prompt or not target_prompt:
        return None, "fill in source and target prompts"
    editor = get_editor()
    schedule = _build_schedule(SCHEDULES[schedule_name])
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    img = img.convert("RGB").resize((512, 512))
    try:
        result = editor.edit(
            img, source_prompt, target_prompt,
            schedule=schedule,
            mask_mode=mask_mode,
            composite_mode=composite_mode,
            background_prompt=background_prompt or None,
        )
        return result, "done"
    except Exception as e:
        return None, f"error: {e}"


# ---------------------------------------------------------------------------
# TAB 2: repositioning (teammate's module -- stub)
# ---------------------------------------------------------------------------
def run_reposition(image, prompt, dx, dy):
    """STUB. Replace with teammate's function:
        from teammate_repos.reposition import reposition
        return reposition(image, prompt, dx=dx, dy=dy, c=get_components()), "done"
    """
    return None, "Repositioning module not yet wired in. Hand off to teammate."


# ---------------------------------------------------------------------------
# TAB 3: style transfer (teammate's module -- stub)
# ---------------------------------------------------------------------------
def run_style(image, style_prompt, strength):
    """STUB. Replace with teammate's function:
        from teammate_repos.style import transfer
        return transfer(image, style_prompt, strength=strength, c=get_components()), "done"
    """
    return None, "Style transfer module not yet wired in. Hand off to teammate."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="CS5788 — Unified Image Editor") as app:
    gr.Markdown(
        "# CS5788 Generative Models — Unified Image Editor\n"
        "Three operations on a real photograph: replace an object, "
        "reposition an object, or transfer a style. All three share one Stable "
        "Diffusion v1.5 backbone loaded once at startup."
    )

    with gr.Tabs():
        # --- Tab 1: Replace ---
        with gr.Tab("Object Replacement"):
            with gr.Row():
                with gr.Column():
                    r_in = gr.Image(type="pil", label="source image")
                    r_src = gr.Textbox(label="source prompt",
                                       placeholder="a photograph of a cat sitting on a couch")
                    r_tgt = gr.Textbox(label="target prompt",
                                       placeholder="a photograph of a dog sitting on a couch")
                    r_sched = gr.Dropdown(choices=list(SCHEDULES.keys()),
                                          value="linear decay (recommended)",
                                          label="schedule")
                    r_mask = gr.Radio(choices=["attention", "none"],
                                      value="attention",
                                      label="mask mode")
                    r_comp = gr.Radio(choices=["strict", "inpaint"],
                                      value="strict",
                                      label="composite mode (use 'inpaint' for shape-mismatch edits)")
                    r_bg = gr.Textbox(label="background prompt (only used by inpaint mode; auto-derived if empty)",
                                      placeholder="optional")
                    r_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    r_out = gr.Image(type="pil", label="edited image")
                    r_status = gr.Textbox(label="status", interactive=False)
            r_btn.click(run_replace,
                        [r_in, r_src, r_tgt, r_sched, r_mask, r_comp, r_bg],
                        [r_out, r_status])

        # --- Tab 2: Reposition ---
        with gr.Tab("Repositioning"):
            with gr.Row():
                with gr.Column():
                    p_in = gr.Image(type="pil", label="source image")
                    p_prompt = gr.Textbox(label="object to reposition", placeholder="the cat")
                    p_dx = gr.Slider(-1.0, 1.0, value=0.0, label="horizontal shift")
                    p_dy = gr.Slider(-1.0, 1.0, value=0.0, label="vertical shift")
                    p_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    p_out = gr.Image(type="pil", label="repositioned image")
                    p_status = gr.Textbox(label="status", interactive=False)
            p_btn.click(run_reposition, [p_in, p_prompt, p_dx, p_dy], [p_out, p_status])

        # --- Tab 3: Style ---
        with gr.Tab("Style Transfer"):
            with gr.Row():
                with gr.Column():
                    s_in = gr.Image(type="pil", label="source image")
                    s_style = gr.Textbox(label="style prompt", placeholder="oil painting in the style of Van Gogh")
                    s_strength = gr.Slider(0.0, 1.0, value=0.5, label="strength")
                    s_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    s_out = gr.Image(type="pil", label="stylized image")
                    s_status = gr.Textbox(label="status", interactive=False)
            s_btn.click(run_style, [s_in, s_style, s_strength], [s_out, s_status])


if __name__ == "__main__":
    app.launch(share=True)
