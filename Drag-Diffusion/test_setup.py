# test_setup.py
import torch
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline
from transformers import CLIPTextModel, CLIPTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the public SD 2.1 community mirror used by default in the pipeline
model_id = "sd2-community/stable-diffusion-2-1-base"
inpaint_model_id = "sd2-community/stable-diffusion-2-inpainting"

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    inpaint_model_id,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)
print("All components loaded successfully")
