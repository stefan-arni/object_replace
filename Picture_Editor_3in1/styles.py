"""
Central registry of all supported art styles.
Each style maps to a WikiArt dataset filter and a LoRA trigger phrase.
"""

STYLES = {
    "van_gogh": {
        "display_name": "Van Gogh",
        "filter_key": "artist",
        "filter_id": 22,
        "trigger": "a painting in the style of van gogh",
        "data_dir": "data/van_gogh",
        "lora_dir": "output/lora/van_gogh/final",
    },
    "seurat": {
        "display_name": "Seurat",
        "filter_key": "artist",
        "filter_id": 47,
        "trigger": "a painting in the style of seurat pointillism",
        "data_dir": "data/seurat",
        "lora_dir": "output/lora/seurat/final",
    },
    "monet": {
        "display_name": "Monet",
        "filter_key": "artist",
        "filter_id": 4,
        "trigger": "a painting in the style of claude monet",
        "data_dir": "data/monet",
        "lora_dir": "output/lora/monet/final",
    },
    "ukiyoe": {
        "display_name": "Ukiyo-e",
        "filter_key": "style",
        "filter_id": 26,
        "trigger": "a painting in the style of ukiyo-e japanese woodblock print",
        "data_dir": "data/ukiyoe",
        "lora_dir": "output/lora/ukiyoe/final",
    },
    "picasso": {
        "display_name": "Picasso",
        "filter_key": "artist",
        "filter_id": 15,
        "trigger": "a painting in the style of pablo picasso",
        "data_dir": "data/picasso",
        "lora_dir": "output/lora/picasso/final",
    },
}

STYLE_NAMES = list(STYLES.keys())
DISPLAY_NAMES = {k: v["display_name"] for k, v in STYLES.items()}
