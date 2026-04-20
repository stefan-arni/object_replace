"""Auto-curate ~20 real photos from COCO val2017 and write data/prompts.json.

Picks images where ONE object of a chosen category dominates the frame
(>= MIN_AREA_RATIO of the image), so the swap has a clear target. The
swap pairs cover both structural (similar silhouette: cat<->dog) and
geometric (different silhouette: apple<->banana) edits, which is what
the schedule ablation is designed to differentiate.

Run once. Idempotent (won't re-download if files exist).

  python scripts/fetch_real_photos.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REAL_DIR = DATA_DIR / "real"
PROMPTS_PATH = DATA_DIR / "prompts.json"
CACHE_DIR = PROJECT_ROOT / "outputs" / "_cache"
ANNOTATIONS_CACHE = CACHE_DIR / "coco_val2017_instances.json"

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL_PATTERN = "http://images.cocodataset.org/val2017/{file_name}"

# (source_category_name [must match COCO], target_word, edit_type)
SWAP_PAIRS = [
    # structural -- similar silhouette
    ("cat",     "dog",      "structural"),
    ("dog",     "cat",      "structural"),
    ("horse",   "cow",      "structural"),
    ("zebra",   "horse",    "structural"),
    ("apple",   "orange",   "structural"),
    # geometric -- different silhouette, layout has to adapt
    ("apple",   "banana",   "geometric"),
    ("banana",  "apple",    "geometric"),
    ("cup",     "bottle",   "geometric"),
    ("bowl",    "vase",     "geometric"),
    ("pizza",   "donut",    "geometric"),
]
IMAGES_PER_PAIR = 2     # 10 pairs * 2 = 20 images
MIN_AREA_RATIO = 0.10   # bbox must cover >= 10% of frame


def download_annotations():
    if ANNOTATIONS_CACHE.exists():
        print(f"using cached annotations at {ANNOTATIONS_CACHE.relative_to(PROJECT_ROOT)}")
        return json.loads(ANNOTATIONS_CACHE.read_text())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("downloading COCO val2017 annotations (~26MB)...")
    with urlopen(ANNOTATIONS_URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open("annotations/instances_val2017.json") as f:
            ann = json.load(f)
    ANNOTATIONS_CACHE.write_text(json.dumps(ann))
    print(f"cached to {ANNOTATIONS_CACHE.relative_to(PROJECT_ROOT)}")
    return ann


def select_images(ann):
    cat_name_to_id = {c["name"]: c["id"] for c in ann["categories"]}
    img_by_id = {img["id"]: img for img in ann["images"]}

    anns_by_img = {}
    for a in ann["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    selected, used = [], set()
    for source, target, edit_type in SWAP_PAIRS:
        if source not in cat_name_to_id:
            print(f"  SKIP {source!r} -> {target!r}: not a COCO category")
            continue
        src_cat = cat_name_to_id[source]

        candidates = []
        for img_id, anns in anns_by_img.items():
            if img_id in used:
                continue
            cat_anns = [a for a in anns if a["category_id"] == src_cat]
            if len(cat_anns) != 1:
                continue
            img = img_by_id.get(img_id)
            if img is None:
                continue
            if cat_anns[0]["area"] / (img["width"] * img["height"]) < MIN_AREA_RATIO:
                continue
            candidates.append(img)

        candidates.sort(key=lambda i: i["id"])  # deterministic
        picked = candidates[:IMAGES_PER_PAIR]
        for img in picked:
            used.add(img["id"])
            selected.append({"image": img, "source": source, "target": target, "edit_type": edit_type})
        print(f"  {source:>7} -> {target:<8}  picked {len(picked)} / {len(candidates)} candidates")
    return selected


def download_images(selected):
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(selected, 1):
        file_name = s["image"]["file_name"]
        out_path = REAL_DIR / file_name
        if out_path.exists():
            continue
        url = IMAGE_URL_PATTERN.format(file_name=file_name)
        print(f"  [{i:>2}/{len(selected)}] {file_name}")
        urlretrieve(url, out_path)


def write_prompts_json(selected):
    entries = [
        {
            "image": s["image"]["file_name"],
            "source": f"a photograph of a {s['source']}",
            "target": f"a photograph of a {s['target']}",
            "edit_type": s["edit_type"],
        }
        for s in selected
    ]
    PROMPTS_PATH.write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} entries to {PROMPTS_PATH.relative_to(PROJECT_ROOT)}")


def main():
    ann = download_annotations()
    print(f"\nselecting images (>=1 dominant annotation, area >= {int(MIN_AREA_RATIO * 100)}% of frame)")
    selected = select_images(ann)
    print(f"\ntotal selected: {len(selected)} images")

    print("\ndownloading images to data/real/")
    download_images(selected)

    print()
    write_prompts_json(selected)

    print()
    print("done.")
    print(f"  images:  {REAL_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  prompts: {PROMPTS_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print("next steps:")
    print("  1. (optional) edit data/prompts.json to refine prompts with scene context")
    print("     e.g. 'a photograph of a cat' -> 'a photograph of a cat sitting on a couch'")
    print("  2. python scripts/reconstruct.py data/real/<one_image>.jpg --prompt '...'")
    print("     pilot test: confirm reconstruction LPIPS < 0.05 on a real photo")
    print("  3. python scripts/run_ablation.py")


if __name__ == "__main__":
    main()
