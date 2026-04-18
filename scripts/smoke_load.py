import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sd_components import encode_prompt, load_sd

c = load_sd()
emb = encode_prompt(c, "a photo of a cat")

print(f"embedding shape: {tuple(emb.shape)}")
print(f"embedding dtype: {emb.dtype}")
print(f"device: {c.device}")
print(f"dtype: {c.dtype}")
print(f"unet on:         {next(c.unet.parameters()).device}")
print(f"vae on:          {next(c.vae.parameters()).device}")
print(f"text_encoder on: {next(c.text_encoder.parameters()).device}")
