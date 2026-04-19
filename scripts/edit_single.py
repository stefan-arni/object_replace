"""Edit a single real image. Saves source + edited (+ mask if requested).

  python scripts/edit_single.py path/to/image.jpg \
    --src "a photograph of a cat sitting on a couch" \
    --tgt "a photograph of a dog sitting on a couch" \
    --schedule linear --mask-mode attention
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from editor import Editor
from masks import visualize_mask
from sd_components import load_sd
from schedules import (
    constant_replaced,
    cosine_replaced,
    linear_decay_replaced,
    piecewise_demo,
    vanilla_p2p,
)

SCHEDULES = {
    "vanilla":   lambda: vanilla_p2p(0.8),
    "linear":    linear_decay_replaced,
    "cosine":    cosine_replaced,
    "constant":  lambda: constant_replaced(0.5),
    "piecewise": piecewise_demo,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--schedule", choices=list(SCHEDULES.keys()), default="vanilla")
    ap.add_argument("--mask-mode", choices=["none", "attention"], default="none")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--out", type=Path, default=Path("outputs/edit_single"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    c = load_sd()
    editor = Editor(c)
    src_pil = Image.open(args.image).convert("RGB").resize((512, 512))

    print(f"editing {args.image.name}")
    print(f"  src: {args.src!r}")
    print(f"  tgt: {args.tgt!r}")
    print(f"  schedule={args.schedule}  mask_mode={args.mask_mode}")

    schedule = SCHEDULES[args.schedule]()
    result = editor.edit(
        src_pil, args.src, args.tgt,
        schedule=schedule,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        mask_mode=args.mask_mode,
        return_mask=(args.mask_mode == "attention"),
    )
    if args.mask_mode == "attention":
        edit_pil, mask = result
    else:
        edit_pil = result
        mask = None

    tag = f"{args.image.stem}__{args.schedule}__{args.mask_mode}"
    src_pil.save(args.out / f"{tag}__src.png")
    edit_pil.save(args.out / f"{tag}__edit.png")
    if mask is not None:
        mask_vis = visualize_mask(mask, size=512)
        Image.fromarray((mask_vis.cpu().numpy() * 255).astype("uint8")).save(
            args.out / f"{tag}__mask.png"
        )
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
