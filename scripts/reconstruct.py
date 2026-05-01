"""Invert a real image with null-text and re-decode. Reports reconstruction LPIPS.

  python scripts/reconstruct.py path/to/image.jpg --prompt "a photo of a cat"

Run this on every new photo before trying to edit it -- if reconstruction LPIPS
is much above 0.05, the inversion is shaky and edits will inherit the drift.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from editor import _to_pil
from null_text_inv import null_text_inversion, sample_with_null
from metrics import reconstruction_lpips
from sd_components import decode_latents, encode_prompt, load_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--inner-steps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("outputs/reconstruct"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    c = load_sd()
    src_pil = Image.open(args.image).convert("RGB").resize((512, 512))

    print(f"Inverting {args.image.name} with prompt: {args.prompt!r}")
    nt = null_text_inversion(
        c, src_pil, args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        inner_steps=args.inner_steps,
    )

    cond = encode_prompt(c, args.prompt)
    z0 = sample_with_null(c, cond, nt.null_embeds, nt.z_T, args.steps, args.guidance)
    recon_pil = _to_pil(decode_latents(c, z0).clamp(-1, 1))

    src_pil.save(args.out / f"{args.image.stem}_orig.png")
    recon_pil.save(args.out / f"{args.image.stem}_recon.png")

    score = reconstruction_lpips(src_pil, recon_pil)
    verdict = "OK" if score < 0.05 else "HIGH (inversion is drifting)"
    print(f"reconstruction LPIPS: {score:.4f}  ({verdict}; target < 0.05)")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
