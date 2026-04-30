"""
Download paintings from the WikiArt dataset on HuggingFace.
Uses streaming to avoid downloading the entire ~25GB dataset.
Supports filtering by artist ID or style ID.
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from styles import STYLES


def resolve_filter_id(dataset, filter_key: str, fallback_id: int) -> int:
    """Resolve the filter ID from dataset features at runtime."""
    features = dataset.features
    if features and filter_key in features and hasattr(features[filter_key], "names"):
        return fallback_id
    return fallback_id


def download_style(style_key: str, num_images: int = 100):
    cfg = STYLES[style_key]
    output_path = Path(cfg["data_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob("*.jpg")) + list(output_path.glob("*.png"))
    if len(existing) >= num_images:
        print(f"[{cfg['display_name']}] Already have {len(existing)} images, skipping.")
        return

    print(f"[{cfg['display_name']}] Loading WikiArt dataset (streaming)...")
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)

    filter_key = cfg["filter_key"]
    target_id = cfg["filter_id"]
    print(f"[{cfg['display_name']}] Filtering by {filter_key}={target_id}")

    count = len(existing)
    needed = num_images - count

    for sample in tqdm(dataset, desc=f"Scanning for {cfg['display_name']}", unit="img"):
        if count >= num_images:
            break

        if sample.get(filter_key) != target_id:
            continue

        image: Image.Image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"{style_key}_{count:04d}.jpg"
        filepath = output_path / filename

        if not filepath.exists():
            image.save(filepath, "JPEG", quality=95)

        count += 1
        if count % 10 == 0:
            print(f"  [{cfg['display_name']}] Downloaded {count}/{num_images}")

    print(f"[{cfg['display_name']}] Done! {count} images in {cfg['data_dir']}")
    if count < num_images:
        print(f"  Warning: only found {count}/{num_images} images in the dataset.")


def main():
    parser = argparse.ArgumentParser(description="Download art style training data from WikiArt")
    parser.add_argument(
        "--style", type=str, default=None,
        choices=list(STYLES.keys()),
        help="Which style to download (default: all)",
    )
    parser.add_argument("--num-images", type=int, default=100)
    args = parser.parse_args()

    if args.style:
        download_style(args.style, args.num_images)
    else:
        for key in STYLES:
            download_style(key, args.num_images)


if __name__ == "__main__":
    main()
