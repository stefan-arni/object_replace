"""
Evaluate generated images against the provided grass-lawn + small-dog reference.

This deliberately avoids model downloads so it can run while the heavier
diffusion pipeline is unavailable in an offline Cloud Agent environment.
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from eval.reference_profile import ReferenceProfile, score_against_lawn_dog_reference


_GENERATED_REPORT_SUFFIXES = (
    "_report.png",
    "_report.jpg",
    "_report.jpeg",
)
_GENERATED_RESULT_STEMS = {
    "reference_eval",
    "reference_eval_report",
    "quick_test_reference_eval",
    "quick_test_reference_report",
}


def _is_candidate_image(path: Path) -> bool:
    """Exclude evaluator outputs so watch mode never scores its own report."""
    name = path.name.lower()
    if path.stem.lower() in _GENERATED_RESULT_STEMS:
        return False
    if name.endswith(_GENERATED_REPORT_SUFFIXES):
        return False
    if "_mask" in name or name.endswith("_composite.png"):
        return False
    return path.suffix.lower() in {".png", ".jpg", ".jpeg"}


def resolve_image(path: str, result_dir: Path = Path("data/results")) -> Path:
    if path != "latest":
        return Path(path)

    candidate_groups = [
        [p for p in result_dir.glob("*_ours.png") if _is_candidate_image(p)],
        [p for p in result_dir.glob("*_baseline.png") if _is_candidate_image(p)],
        [
            p
            for pattern in ("*.png", "*.jpg", "*.jpeg")
            for p in result_dir.glob(pattern)
            if _is_candidate_image(p)
        ],
    ]
    candidates = next((group for group in candidate_groups if group), [])
    if not candidates:
        raise FileNotFoundError("No images found in data/results; pass --image explicitly.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _save_report(image: Image.Image, metrics: dict, output_path: Path) -> None:
    report = image.convert("RGB").resize((512, 512))
    draw = ImageDraw.Draw(report, "RGBA")

    dog_box = metrics["dog_box"]
    if dog_box is not None:
        analysis_w = metrics["analysis_width"]
        analysis_h = metrics["analysis_height"]
        scale_x = 512 / analysis_w
        scale_y = 512 / analysis_h
        box = [
            int(dog_box[0] * scale_x),
            int(dog_box[1] * scale_y),
            int(dog_box[2] * scale_x),
            int(dog_box[3] * scale_y),
        ]
        draw.rectangle(box, outline=(255, 255, 255, 255), width=4)
        draw.rectangle(box, outline=(255, 180, 0, 255), width=2)

    panel_h = 114
    draw.rectangle([0, 512 - panel_h, 512, 512], fill=(0, 0, 0, 178))
    lines = [
        f"reference score: {metrics['overall_score']:.3f}",
        f"grass coverage: {metrics['grass_coverage']:.3f}",
        f"dog coverage: {metrics['dog_coverage']:.3f}",
        f"dog center: {metrics['dog_center']}",
        f"verdict: {metrics['verdict']}",
    ]
    y = 512 - panel_h + 10
    for line in lines:
        draw.text((12, y), line, fill=(255, 255, 255, 255))
        y += 20

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="latest",
        help='Image to score, or "latest" to use the newest image in data/results.',
    )
    parser.add_argument("--json-output", default="data/results/reference_eval.json")
    parser.add_argument("--report-output", default="data/results/reference_eval_report.png")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=0.0,
        help="Repeat every N seconds; useful with --image latest for background monitoring.",
    )
    args = parser.parse_args()

    last_image_signature = None
    while True:
        image_path = resolve_image(args.image)
        image_signature = (image_path, image_path.stat().st_mtime_ns)
        if args.watch_interval > 0 and image_signature == last_image_signature:
            time.sleep(args.watch_interval)
            continue
        last_image_signature = image_signature

        image = Image.open(image_path).convert("RGB")
        metrics = score_against_lawn_dog_reference(image, ReferenceProfile())
        metrics["image_path"] = str(image_path)
        metrics["passed"] = metrics["overall_score"] >= args.min_score

        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        _save_report(image, metrics, Path(args.report_output))

        print(json.dumps(metrics, indent=2), flush=True)
        if not metrics["passed"] and args.watch_interval <= 0:
            raise SystemExit(
                f"Reference score {metrics['overall_score']:.3f} is below "
                f"threshold {args.min_score:.3f}."
            )
        if args.watch_interval <= 0:
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
