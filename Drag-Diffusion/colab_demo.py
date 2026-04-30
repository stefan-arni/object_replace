"""
Colab demo — paste this into a Colab notebook cell by cell.

Runtime: GPU (T4 or better). Runtime > Change runtime type > GPU.
"""

# ── Cell 1: Install & clone ──────────────────────────────────────────────────
# !pip install diffusers transformers accelerate
# !git clone https://github.com/YOUR_GITHUB/Drag-Diffusion.git
# %cd Drag-Diffusion

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

from utils.image_utils import get_device
from pipeline.relocation_pipeline import ObjectRelocationPipeline
from eval.perceptual_loss import VGGPerceptualLoss
from eval.visualize import make_comparison_figure

device = get_device()
print(f"Device: {device}")  # should say cuda

# ── Cell 3: Load pipeline ─────────────────────────────────────────────────────
# On Colab: SD 2.1 is not cached — pass local_files_only=False to download on first run.

from pipeline.relocation_pipeline import ObjectRelocationPipeline

pipe = ObjectRelocationPipeline(device=device, local_files_only=False)
loss_fn = VGGPerceptualLoss(device=device)

# ── Cell 4: Upload your image ─────────────────────────────────────────────────
from google.colab import files
uploaded = files.upload()  # upload your image here
image_path = list(uploaded.keys())[0]
image = Image.open(image_path).convert("RGB")
plt.imshow(image); plt.title("Your image"); plt.axis("off"); plt.show()

# ── Cell 5: Draw masks interactively ─────────────────────────────────────────
# Option A: draw masks in a paint app, upload them as PNG files
# Option B: define masks programmatically (example below for a top-left object)

H, W = image.size[1], image.size[0]

# Example: manually specify bounding boxes for source and target
# Change these to match where your object actually is
src_box = (50, 100, 200, 300)   # (left, top, right, bottom) in pixels
tgt_box = (350, 100, 500, 300)  # where you want it to move

src_arr = np.zeros((H, W), dtype=np.uint8)
tgt_arr = np.zeros((H, W), dtype=np.uint8)
src_arr[src_box[1]:src_box[3], src_box[0]:src_box[2]] = 255
tgt_arr[tgt_box[1]:tgt_box[3], tgt_box[0]:tgt_box[2]] = 255
source_mask = Image.fromarray(src_arr)
target_mask = Image.fromarray(tgt_arr)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image); axes[0].set_title("Image"); axes[0].axis("off")
axes[1].imshow(source_mask, cmap="gray"); axes[1].set_title("Source mask"); axes[1].axis("off")
axes[2].imshow(target_mask, cmap="gray"); axes[2].set_title("Target mask"); axes[2].axis("off")
plt.tight_layout(); plt.show()

# ── Cell 6: Run the pipeline ──────────────────────────────────────────────────
prompt = "describe the scene with the object at its new location"  # EDIT THIS

print("Running baseline (SDEdit, fresh noise)...")
baseline, composite = pipe(
    image, prompt, source_mask, target_mask,
    use_noise_shift=False, seed=42, num_inference_steps=50, sdedit_strength=0.5,
)

print("Running ours (DDPM noise shift)...")
ours, _ = pipe(
    image, prompt, source_mask, target_mask,
    use_noise_shift=True, seed=42, num_inference_steps=50, sdedit_strength=0.5,
)

# ── Cell 7: Results ───────────────────────────────────────────────────────────
from utils.image_utils import pil_to_tensor

# Score: perceptual distance between source texture and result at target
def obj_crop(img, mask, pad=10):
    arr = np.array(mask.convert("L")) > 127
    ys, xs = np.where(arr)
    if len(ys) == 0:
        return img
    y0, y1, x0, x1 = ys.min()-pad, ys.max()+pad, xs.min()-pad, xs.max()+pad
    y0, x0 = max(0, y0), max(0, x0)
    return img.crop((x0, y0, x1, y1)).resize((256, 256))

src_t   = pil_to_tensor(obj_crop(image,    source_mask), device)
bl_t    = pil_to_tensor(obj_crop(baseline, target_mask), device)
ours_t  = pil_to_tensor(obj_crop(ours,     target_mask), device)

bl_score   = loss_fn(src_t, bl_t)
ours_score = loss_fn(src_t, ours_t)

print(f"\nPerceptual dist (lower = better texture preservation):")
print(f"  Baseline:  {bl_score:.4f}")
print(f"  Ours:      {ours_score:.4f}  {'✓ better' if ours_score < bl_score else '✗ worse'}")

make_comparison_figure(image, baseline, ours, "results.png",
                       title=prompt,
                       scores={"baseline": bl_score, "ours": ours_score})

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, t in zip(axes, [image, composite, baseline, ours],
                      ["Original", "Composite (copy-paste)", "Baseline", "Ours"]):
    ax.imshow(img); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.show()
