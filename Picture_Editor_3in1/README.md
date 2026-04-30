# Multi-Style Art Transfer (LoRA)

Transform photographs using Stable Diffusion 1.5 img2img with per-style LoRA adapters. Styles (Van Gogh, Monet, Seurat, Ukiyo-e, Picasso, etc.) are trained on WikiArt data and can be blended in the Gradio UI or CLI.

## Architecture

- **Foundation model**: Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`) — ~4 GB in fp16
- **Adapters**: LoRA (rank 8, alpha 16) on UNet attention layers — ~3 MB each
- **Training data**: WikiArt via HuggingFace Datasets (streaming; avoids downloading the full dataset)
- **Inference**: img2img with optional multi-adapter blending (`set_adapters`)

### Disk budget (~15 GB typical)

| Component | Size |
|---|---|
| Python + dependencies | ~8 GB |
| SD 1.5 model (cached) | ~4 GB |
| Training images (per style) | ~50 MB |
| LoRA weights (per style) | ~3 MB |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1 — Download training data

All styles in one pass (recommended):

```bash
python download_all.py
```

Or one style:

```bash
python download_dataset.py --style van_gogh --num-images 100
```

Omit `--style` to download every registered style.

## Step 2 — Train LoRAs

Train every style that has enough images (`train_all.py` skips styles that already have `final/pytorch_lora_weights.safetensors`):

```bash
python train_all.py
```

Single-style manual run (see `train_lora.py --help`):

```bash
python train_lora.py \
  --data-dir data/van_gogh \
  --output-dir output/lora/van_gogh \
  --max-steps 500 \
  --trigger-phrase "a painting in the style of van gogh"
```

## Step 3 — Apply styles

### CLI

```bash
python inference.py photo.jpg --styles van_gogh monet --weights 0.6 0.4 --strength 0.65
```

### Web UI

```bash
python app.py
```

Opens a Gradio app (default `http://127.0.0.1:7860`) to upload an image, mix up to three styles with weights, and adjust strength, guidance, steps, and seed.

## Hardware (approximate)

| Device | Training | Inference |
|---|---|---|
| Apple Silicon (MPS) 16 GB+ | Varies by steps | ~15 s/image |
| NVIDIA GPU 8 GB+ VRAM | Faster than MPS | ~5 s/image |
| CPU only | Not recommended | Much slower |

## Project structure

```
├── styles.py             # Style registry (WikiArt filters, triggers, paths)
├── download_dataset.py   # Per-style WikiArt download
├── download_all.py       # Single-pass download for all styles
├── train_lora.py         # LoRA training for one style
├── train_all.py          # Train all styles sequentially
├── inference.py          # CLI: img2img + LoRA blend
├── app.py                # Gradio UI
├── data/<style>/         # Training images
└── output/lora/<style>/final/   # Saved LoRA weights
```
