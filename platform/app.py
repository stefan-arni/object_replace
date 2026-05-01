"""Unified Gradio app for the CS5788 image-editor project.

Three tabs, one image -> three different operations:
  1. Replace Object  (Stefan)   -- swap one object in a real photo for another
  2. Move Object     (teammate) -- DDPM noise-shift relocation, paint two masks
  3. Apply Style     (teammate) -- LoRA-blended art-style transfer

Each module loads its own SD components LAZILY on first use of that tab.
Models live across tab switches; only freshly-clicked tabs trigger loads.

Run locally:
    python platform/app.py

Public sharable link: starts on launch (share=True).
"""
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Drag-Diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "Picture_Editor_3in1"))


# ============================================================================
# LAZY MODEL LOADERS  (each module loads its own SD on first call)
# ============================================================================
_object_editor = None
_relocation_pipe = None
_style_pipe = None


def _get_object_editor():
    global _object_editor
    if _object_editor is None:
        gr.Info("Loading Stable Diffusion 1.5 for object replacement (one-time, ~30s)...")
        from sd_components import load_sd
        from editor import Editor
        _object_editor = Editor(load_sd())
    return _object_editor


def _get_relocation_pipe():
    global _relocation_pipe
    if _relocation_pipe is None:
        gr.Info("Loading Stable Diffusion 2.1 for object relocation (one-time, ~60s)...")
        from pipeline.relocation_pipeline import ObjectRelocationPipeline
        _relocation_pipe = ObjectRelocationPipeline()
    return _relocation_pipe


def _get_style_pipe():
    global _style_pipe
    if _style_pipe is None:
        gr.Info("Loading Stable Diffusion 1.5 img2img + LoRA adapters (one-time, ~30s)...")
        # Picture_Editor has a stale runwayml/* model ID baked in. Monkey-patch
        # the module-global before loading so we don't have to edit their repo.
        import inference
        inference.MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        # Their styles.py uses relative LoRA paths ("output/lora/van_gogh/final").
        # When the app runs from the project root, those paths don't resolve --
        # the LoRAs actually live under Picture_Editor_3in1/output/lora/...
        # Rewrite each entry's lora_dir to an absolute path.
        import styles
        pe_root = PROJECT_ROOT / "Picture_Editor_3in1"
        for key, cfg in styles.STYLES.items():
            cfg["lora_dir"] = str(pe_root / cfg["lora_dir"])
        _style_pipe = inference.load_pipeline(inference.get_device())
    return _style_pipe


# ============================================================================
# TAB 1 — Object Replacement
# ============================================================================
SCHEDULES = {
    "Linear decay (recommended)": "linear_decay",
    "Vanilla P2P (baseline)": "vanilla_p2p",
    "Cosine": "cosine",
    "Constant 0.5 (slow blend)": "constant_05",
    "Piecewise": "piecewise",
}


def _build_schedule(name_key: str):
    from schedules import (
        constant_replaced, cosine_replaced, linear_decay_replaced,
        piecewise_demo, vanilla_p2p,
    )
    return {
        "linear_decay": linear_decay_replaced(),
        "vanilla_p2p":  vanilla_p2p(0.8),
        "cosine":       cosine_replaced(),
        "constant_05":  constant_replaced(0.5),
        "piecewise":    piecewise_demo(),
    }[name_key]


def run_replace(image, source_prompt, target_prompt, schedule_name,
                mask_mode, composite_mode, background_prompt):
    if image is None:
        return None, "Upload an image first."
    if not source_prompt or not target_prompt:
        return None, "Fill in both source and target prompts."

    editor = _get_object_editor()
    schedule = _build_schedule(SCHEDULES[schedule_name])
    img = (image if isinstance(image, Image.Image) else Image.fromarray(image)).convert("RGB").resize((512, 512))

    try:
        result = editor.edit(
            img, source_prompt, target_prompt,
            schedule=schedule,
            mask_mode=mask_mode,
            composite_mode=composite_mode,
            background_prompt=background_prompt or None,
        )
        return result, "Done."
    except Exception as e:
        return None, f"Error: {e}"


# ============================================================================
# TAB 2 — Object Relocation (Drag-Diffusion)
# ============================================================================
def _extract_mask(editor_value, fallback_size=(512, 512)):
    if editor_value is None:
        return Image.new("L", fallback_size, 0)
    layers = editor_value.get("layers") or []
    if not layers or layers[0] is None:
        return Image.new("L", fallback_size, 0)
    layer = layers[0]
    if layer.mode == "RGBA":
        return layer.split()[3]
    return layer.convert("L")


def _seed_editors(image):
    """When an image is uploaded, push it into both mask editors so the user can paint on it."""
    if image is None:
        return gr.update(), gr.update()
    blank = {"background": image, "layers": [], "composite": image}
    return blank, blank


def run_relocate(image, src_editor, tgt_editor, prompt, use_noise_shift,
                 seed, num_steps, sdedit_strength, guidance_scale):
    if image is None:
        return None, "Upload an image first."
    if not prompt or not prompt.strip():
        return None, "Describe the final scene in the prompt."

    src_mask = _extract_mask(src_editor, image.size)
    tgt_mask = _extract_mask(tgt_editor, image.size)
    if np.array(src_mask).sum() == 0:
        return None, "Paint a source mask (where the object IS now)."
    if np.array(tgt_mask).sum() == 0:
        return None, "Paint a target mask (where the object SHOULD GO)."

    pipe = _get_relocation_pipe()
    try:
        result, _composite = pipe(
            image, prompt, src_mask, tgt_mask,
            use_noise_shift=bool(use_noise_shift),
            seed=int(seed),
            num_inference_steps=int(num_steps),
            sdedit_strength=float(sdedit_strength),
            guidance_scale=float(guidance_scale),
        )
        return result, "Done."
    except Exception as e:
        return None, f"Error: {e}"


# ============================================================================
# TAB 3 — Style Transfer (Picture_Editor_3in1)
# ============================================================================
NONE_LABEL = "(none)"


def _style_choices():
    """Read the live STYLES dict from Picture_Editor_3in1.styles."""
    try:
        from styles import STYLES
        return [NONE_LABEL] + [STYLES[k]["display_name"] for k in STYLES]
    except Exception:
        return [NONE_LABEL]


def _display_to_key(display: str) -> str | None:
    if display == NONE_LABEL:
        return None
    from styles import STYLES
    for k, v in STYLES.items():
        if v["display_name"] == display:
            return k
    return None


def run_style(image, style1, w1, style2, w2, style3, w3,
              strength, num_steps, guidance_scale, seed):
    if image is None:
        return None, "Upload an image first."

    keys = [(_display_to_key(style1), w1),
            (_display_to_key(style2), w2),
            (_display_to_key(style3), w3)]
    keys = [(k, w) for k, w in keys if k is not None and w > 0]
    if not keys:
        return None, "Pick at least one style."

    pipe = _get_style_pipe()
    from inference import set_style_mix, build_prompt, stylize_image
    try:
        set_style_mix(pipe, keys)
        prompt = build_prompt(keys)
        result = stylize_image(
            pipe, image, prompt,
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_steps),
            seed=int(seed) if seed is not None else None,
        )
        return result, f"Applied: {prompt}"
    except Exception as e:
        return None, f"Error: {e}"


# ============================================================================
# UI
# ============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

/* --- base canvas --------------------------------------------------------- */
:root, body, .gradio-container, .dark {
    background: #0a0b10 !important;
    color: #e4e4e7 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-feature-settings: "ss01", "cv11";
}
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; padding: 24px !important; }
footer, .gradio-container > footer { display: none !important; }

/* --- hero ---------------------------------------------------------------- */
#hero {
    position: relative;
    padding: 56px 40px 48px;
    margin-bottom: 28px;
    border-radius: 20px;
    background:
      radial-gradient(circle at 18% 28%, rgba(99,102,241,0.20) 0%, transparent 55%),
      radial-gradient(circle at 82% 72%, rgba(236,72,153,0.18) 0%, transparent 55%),
      linear-gradient(180deg, #11121a 0%, #0c0d14 100%);
    border: 1px solid rgba(255,255,255,0.07);
    overflow: hidden;
}
#hero::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, transparent 30%, rgba(99,102,241,0.04) 100%);
    pointer-events: none;
}
#hero .eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.2em;
    color: #818cf8; text-transform: uppercase; margin: 0 0 14px 0;
}
#hero h1 {
    color: #fafafa !important; margin: 0 0 12px 0;
    font-size: 44px; font-weight: 700; letter-spacing: -0.025em; line-height: 1.05;
    background: linear-gradient(180deg, #ffffff 0%, #a5b4fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
#hero p { color: #a1a1aa !important; margin: 0; font-size: 16px; max-width: 640px; line-height: 1.6; }

/* --- tabs ---------------------------------------------------------------- */
.tab-nav { background: transparent !important; border: none !important; gap: 4px !important; }
.tab-nav button {
    background: transparent !important; color: #71717a !important;
    border: 1px solid transparent !important; border-radius: 10px !important;
    padding: 10px 18px !important; font-weight: 500 !important; font-size: 14px !important;
    transition: all 0.18s ease !important;
}
.tab-nav button:hover { color: #d4d4d8 !important; background: rgba(255,255,255,0.03) !important; }
.tab-nav button.selected {
    color: #ffffff !important;
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(129,140,248,0.30) !important;
}

/* --- cards / blocks ------------------------------------------------------ */
.gr-block, .block, .form, .gr-form {
    background: #11121a !important; border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
}
.tabitem { padding-top: 18px !important; }

/* --- inputs -------------------------------------------------------------- */
input, textarea, select, .gr-input, .gr-textbox textarea, .gr-dropdown {
    background: #16172a !important; color: #f4f4f5 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; font-family: 'Inter', sans-serif !important;
}
input:focus, textarea:focus, .gr-input:focus, .gr-textbox textarea:focus {
    border-color: #818cf8 !important; box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
    outline: none !important;
}
label, .label-wrap span, .gr-form label {
    color: #d4d4d8 !important; font-weight: 500 !important; font-size: 13px !important;
    letter-spacing: 0.005em !important;
}
.label-wrap, .block-label { background: transparent !important; color: #a1a1aa !important; }

/* --- buttons ------------------------------------------------------------- */
button.primary, .gr-button-primary {
    background: linear-gradient(180deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; letter-spacing: 0.005em !important;
    padding: 12px 22px !important; font-size: 14px !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.10) inset, 0 8px 24px -8px rgba(99,102,241,0.45) !important;
    transition: all 0.18s ease !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: linear-gradient(180deg, #7376f6 0%, #5b54f0 100%) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.10) inset, 0 12px 32px -8px rgba(99,102,241,0.65) !important;
    transform: translateY(-1px);
}
button.secondary, .gr-button-secondary {
    background: rgba(255,255,255,0.04) !important; color: #d4d4d8 !important;
    border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 10px !important;
}

/* --- sliders, dropdowns, radios ----------------------------------------- */
.gr-radio label, .gr-checkbox label { color: #d4d4d8 !important; }
input[type="range"]::-webkit-slider-thumb { background: #818cf8 !important; }

/* --- accordions ---------------------------------------------------------- */
.gr-accordion {
    background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}
.gr-accordion summary, .gr-accordion .label-wrap {
    color: #d4d4d8 !important; font-weight: 500 !important;
}

/* --- images -------------------------------------------------------------- */
.gr-image, .image-container, .gr-image-container {
    background: #0c0d14 !important; border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
}

/* --- markdown ------------------------------------------------------------ */
.gr-markdown, .prose {
    color: #d4d4d8 !important;
    background: transparent !important;
}
.gr-markdown p { color: #a1a1aa !important; line-height: 1.65; }
.gr-markdown strong { color: #fafafa !important; font-weight: 600 !important; }

/* --- footer below tabs --------------------------------------------------- */
.app-footer {
    text-align: center; padding: 32px 16px 8px;
    color: #52525b; font-size: 12px; font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em; text-transform: uppercase;
}
"""

THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="zinc",
    font=("Inter", "ui-sans-serif", "system-ui"),
).set(
    body_background_fill="#0a0b10",
    body_text_color="#e4e4e7",
    background_fill_primary="#11121a",
    background_fill_secondary="#16172a",
    border_color_primary="rgba(255,255,255,0.06)",
    block_radius="14px",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
    button_primary_text_color="white",
    block_title_text_color="#d4d4d8",
    block_label_text_color="#a1a1aa",
)


with gr.Blocks(theme=THEME, css=CSS, title="CS5788 — Image Editor 3-in-1") as app:

    gr.HTML(
        """
        <div id="hero">
            <p class="eyebrow">CS5788 · Generative Models · Spring 2026</p>
            <h1>Edit Reality.</h1>
            <p>Replace, relocate, or restyle any object in a real photograph.
            Three complementary diffusion pipelines, one interface.</p>
        </div>
        """
    )

    with gr.Tabs():
        # ---------------- Tab 1 ----------------
        with gr.Tab("Replace Object"):
            gr.Markdown(
                "**Swap one object in your photo for another.** "
                "Describe what's there now and what you want instead. "
                "The background stays bit-exact."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    r_in = gr.Image(type="pil", label="Source image", height=380)
                    r_src = gr.Textbox(
                        label="Source prompt — what's in the photo now",
                        placeholder="a photograph of a cat sitting on a couch",
                    )
                    r_tgt = gr.Textbox(
                        label="Target prompt — what should replace it",
                        placeholder="a photograph of a dog sitting on a couch",
                    )
                    with gr.Accordion("Advanced", open=False):
                        r_sched = gr.Dropdown(
                            choices=list(SCHEDULES.keys()),
                            value="Linear decay (recommended)",
                            label="Attention swap schedule",
                        )
                        r_mask = gr.Radio(
                            choices=["attention", "none"], value="attention",
                            label="Mask mode (attention = preserve background)",
                        )
                        r_comp = gr.Radio(
                            choices=["strict", "inpaint"], value="strict",
                            label="Composite mode (use 'inpaint' for size-mismatch edits)",
                        )
                        r_bg = gr.Textbox(
                            label="Background prompt (only for inpaint mode; auto-derived if blank)",
                            placeholder="optional",
                        )
                    r_btn = gr.Button("Replace object", variant="primary", size="lg")
                with gr.Column(scale=1):
                    r_out = gr.Image(type="pil", label="Edited image", height=380)
                    r_status = gr.Textbox(label="Status", interactive=False)
            r_btn.click(
                run_replace,
                [r_in, r_src, r_tgt, r_sched, r_mask, r_comp, r_bg],
                [r_out, r_status],
            )

        # ---------------- Tab 2 ----------------
        with gr.Tab("Move Object"):
            gr.Markdown(
                "**Move an object within the same photo.** "
                "Paint over the object on the left (where it is now), "
                "paint where you want it on the right, then describe the final scene."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    m_in = gr.Image(type="pil", label="Source image", height=300)
                    with gr.Row():
                        m_src_mask = gr.ImageEditor(
                            label="Source mask (paint over the object)",
                            type="pil",
                            height=280,
                            brush=gr.Brush(default_size=30, colors=["#22c55e"]),
                        )
                        m_tgt_mask = gr.ImageEditor(
                            label="Target mask (paint where to move it)",
                            type="pil",
                            height=280,
                            brush=gr.Brush(default_size=30, colors=["#ef4444"]),
                        )
                    m_prompt = gr.Textbox(
                        label="Final-scene prompt",
                        placeholder="a cat sitting on the rug in a sunlit room",
                    )
                    with gr.Accordion("Advanced", open=False):
                        m_noise_shift = gr.Checkbox(
                            value=True,
                            label="Use noise-prior shift (the novel contribution; uncheck for SDEdit baseline)",
                        )
                        m_seed = gr.Number(value=42, label="Seed", precision=0)
                        m_steps = gr.Slider(20, 80, value=50, step=1, label="Inference steps")
                        m_strength = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="SDEdit strength")
                        m_cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                    m_btn = gr.Button("Move object", variant="primary", size="lg")
                with gr.Column(scale=1):
                    m_out = gr.Image(type="pil", label="Result", height=380)
                    m_status = gr.Textbox(label="Status", interactive=False)
            m_in.change(_seed_editors, [m_in], [m_src_mask, m_tgt_mask])
            m_btn.click(
                run_relocate,
                [m_in, m_src_mask, m_tgt_mask, m_prompt, m_noise_shift,
                 m_seed, m_steps, m_strength, m_cfg],
                [m_out, m_status],
            )

        # ---------------- Tab 3 ----------------
        with gr.Tab("Apply Style"):
            gr.Markdown(
                "**Repaint your photo in the style of one or more famous artists.** "
                "Mix up to three styles with custom weights. Higher strength = stronger stylization."
            )
            style_choices = _style_choices()
            with gr.Row():
                with gr.Column(scale=1):
                    s_in = gr.Image(type="pil", label="Source image", height=380)
                    with gr.Row():
                        s_style1 = gr.Dropdown(style_choices, value=style_choices[1] if len(style_choices) > 1 else NONE_LABEL, label="Style 1")
                        s_w1 = gr.Slider(0, 100, value=100, step=5, label="Weight 1 (%)")
                    with gr.Row():
                        s_style2 = gr.Dropdown(style_choices, value=NONE_LABEL, label="Style 2")
                        s_w2 = gr.Slider(0, 100, value=0, step=5, label="Weight 2 (%)")
                    with gr.Row():
                        s_style3 = gr.Dropdown(style_choices, value=NONE_LABEL, label="Style 3")
                        s_w3 = gr.Slider(0, 100, value=0, step=5, label="Weight 3 (%)")
                    with gr.Accordion("Advanced", open=False):
                        s_strength = gr.Slider(0.2, 1.0, value=0.45, step=0.05, label="Stylization strength (raise for more abstract; lower to preserve subject)")
                        s_steps = gr.Slider(15, 60, value=30, step=1, label="Inference steps")
                        s_cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                        s_seed = gr.Number(value=0, label="Seed", precision=0)
                    s_btn = gr.Button("Apply style", variant="primary", size="lg")
                with gr.Column(scale=1):
                    s_out = gr.Image(type="pil", label="Stylized image", height=380)
                    s_status = gr.Textbox(label="Status", interactive=False)
            s_btn.click(
                run_style,
                [s_in, s_style1, s_w1, s_style2, s_w2, s_style3, s_w3,
                 s_strength, s_steps, s_cfg, s_seed],
                [s_out, s_status],
            )

    gr.HTML(
        """
        <div style="text-align:center; padding: 16px; color: #64748b; font-size: 13px;">
            CS5788 Generative Models · Cornell Tech · Spring 2026<br>
            Object replacement · DDPM-noise-shift relocation · LoRA-adapted style transfer
        </div>
        """
    )


if __name__ == "__main__":
    app.launch(share=True)
