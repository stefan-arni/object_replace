"""
Download training images for ALL styles in a single pass through the WikiArt dataset.
Much faster than downloading each style separately.
"""

from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from styles import STYLES

NUM_IMAGES = 100


def main():
    artist_targets = {}
    style_targets = {}

    for key, cfg in STYLES.items():
        out_dir = Path(cfg["data_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")))

        if existing >= NUM_IMAGES:
            print(f"[{cfg['display_name']}] Already have {existing} images — skipping.")
            continue

        entry = {"key": key, "cfg": cfg, "count": existing, "need": NUM_IMAGES}

        if cfg["filter_key"] == "artist":
            artist_targets[cfg["filter_id"]] = entry
        else:
            style_targets[cfg["filter_id"]] = entry

    if not artist_targets and not style_targets:
        print("All styles already downloaded!")
        return

    remaining = []
    for t in list(artist_targets.values()) + list(style_targets.values()):
        remaining.append(f"{t['cfg']['display_name']} (need {t['need'] - t['count']} more)")
    print(f"Downloading: {', '.join(remaining)}")
    print("Streaming WikiArt dataset (single pass)...\n")

    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)

    for sample in tqdm(dataset, desc="Scanning WikiArt", unit="img"):
        if not artist_targets and not style_targets:
            break

        artist_id = sample.get("artist")
        style_id = sample.get("style")

        matched = None
        if artist_id in artist_targets:
            matched = artist_targets[artist_id]
        elif style_id in style_targets:
            matched = style_targets[style_id]

        if matched is None:
            continue

        key = matched["key"]
        cfg = matched["cfg"]
        count = matched["count"]

        image: Image.Image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"{key}_{count:04d}.jpg"
        filepath = Path(cfg["data_dir"]) / filename

        if not filepath.exists():
            image.save(filepath, "JPEG", quality=95)

        matched["count"] = count + 1

        if matched["count"] % 20 == 0:
            tqdm.write(f"  [{cfg['display_name']}] {matched['count']}/{matched['need']}")

        if matched["count"] >= matched["need"]:
            tqdm.write(f"  [{cfg['display_name']}] Done! ({matched['count']} images)")
            if cfg["filter_key"] == "artist":
                del artist_targets[cfg["filter_id"]]
            else:
                del style_targets[cfg["filter_id"]]

    print("\nDownload summary:")
    for key, cfg in STYLES.items():
        out_dir = Path(cfg["data_dir"])
        n = len(list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")))
        print(f"  {cfg['display_name']}: {n} images")


if __name__ == "__main__":
    main()
