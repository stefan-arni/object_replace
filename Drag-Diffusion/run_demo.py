"""
Interactive demo — loads an image and two masks, runs both baseline and ours.
Usage:
    python run_demo.py --image data/test_images/cat.jpg \
                       --src_mask data/test_images/cat_src_mask.png \
                       --tgt_mask data/test_images/cat_tgt_mask.png \
                       --prompt "a cat sitting on the right side of the grass" \
                       --output data/results/demo_comparison.png
"""

import argparse
from PIL import Image
from utils.image_utils import get_device
from pipeline.relocation_pipeline import ObjectRelocationPipeline
from eval.visualize import make_comparison_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--src_mask", required=True)
    parser.add_argument("--tgt_mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="data/results/demo.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    args = parser.parse_args()

    device = get_device()
    pipe = ObjectRelocationPipeline(device=device)

    image = Image.open(args.image).convert("RGB")
    src_mask = Image.open(args.src_mask).convert("L")
    tgt_mask = Image.open(args.tgt_mask).convert("L")

    print("Running SDEdit baseline (use_noise_shift=False)...")
    baseline, composite = pipe(
        image, args.prompt, src_mask, tgt_mask,
        use_noise_shift=False, seed=args.seed,
        num_inference_steps=args.steps, guidance_scale=args.cfg,
    )
    composite.save(args.output.replace(".png", "_composite.png"))

    print("Running ours (use_noise_shift=True)...")
    ours, _ = pipe(
        image, args.prompt, src_mask, tgt_mask,
        use_noise_shift=True, seed=args.seed,
        num_inference_steps=args.steps, guidance_scale=args.cfg,
    )

    import os; os.makedirs(os.path.dirname(args.output), exist_ok=True)
    make_comparison_figure(image, baseline, ours, args.output, title=args.prompt)
    baseline.save(args.output.replace(".png", "_baseline.png"))
    ours.save(args.output.replace(".png", "_ours.png"))
    print("Done.")


if __name__ == "__main__":
    main()
