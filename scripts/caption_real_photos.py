"""Re-write data/prompts.json with BLIP-generated source prompts that actually
describe each photo. The auto-curation step picks images by COCO category but
COCO labels are sometimes wrong (a burger labeled 'pizza') and bare prompts
like 'a photograph of a banana' lose all scene context, which kills null-text
inversion on real photos.

For each entry:
  - caption the image with BLIP
  - source prompt = the caption
  - target prompt = caption with source word swapped to target word
  - if the source word isn't in the caption, flag the entry as "needs manual
    review" -- the COCO label was probably wrong
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = PROJECT_ROOT / "data" / "prompts.json"
REAL_DIR = PROJECT_ROOT / "data" / "real"


def _strip_bare(s: str) -> str:
    return s.replace("a photograph of a ", "").replace("a photograph of an ", "").strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading BLIP image-captioning model (~500MB on first run)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device).eval()

    entries = json.loads(PROMPTS_PATH.read_text())
    new_entries = []
    flagged = []

    print(f"\ncaptioning {len(entries)} images")
    for entry in entries:
        img_path = REAL_DIR / entry["image"]
        if not img_path.exists():
            print(f"  SKIP {entry['image']}: not found")
            continue

        img = Image.open(img_path).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()

        src_word = _strip_bare(entry["source"])
        tgt_word = _strip_bare(entry["target"])

        if src_word.lower() in caption.lower():
            # rebuild target by case-insensitive substitution of the source word
            import re
            new_source = caption
            new_target = re.sub(rf"\b{re.escape(src_word)}\b", tgt_word, caption, flags=re.IGNORECASE)
            note = ""
        else:
            # COCO label is suspicious for this image. Keep it but flag it; the
            # user can manually edit prompts.json or skip this entry.
            new_source = caption
            new_target = f"a photograph of a {tgt_word}"
            note = f"  [FLAG: '{src_word}' not in caption, COCO label likely wrong]"
            flagged.append(entry["image"])

        new_entries.append({
            "image": entry["image"],
            "source": new_source,
            "target": new_target,
            "edit_type": entry.get("edit_type", "structural"),
        })
        print(f"  {entry['image']}: {caption!r}{note}")
        print(f"     src='{new_source}'")
        print(f"     tgt='{new_target}'")

    PROMPTS_PATH.write_text(json.dumps(new_entries, indent=2))
    print(f"\nupdated {PROMPTS_PATH}")
    if flagged:
        print(f"\n{len(flagged)} entries FLAGGED -- caption did not contain the source word.")
        print("   The COCO label is probably wrong for these images. Either:")
        print("   - manually fix the source/target in data/prompts.json, or")
        print("   - just remove these entries from the JSON before running the ablation.")
        for f in flagged:
            print(f"     {f}")


if __name__ == "__main__":
    main()
