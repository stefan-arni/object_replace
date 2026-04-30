"""
Train LoRA adapters for all registered styles sequentially.
Skips styles that already have a trained 'final' checkpoint.
"""

import subprocess
import sys
from pathlib import Path

from styles import STYLES


def train_style(key: str, cfg: dict):
    lora_dir = Path(cfg["lora_dir"])
    weights_file = lora_dir / "pytorch_lora_weights.safetensors"

    if weights_file.exists():
        print(f"\n[{cfg['display_name']}] Already trained — skipping ({weights_file})")
        return

    print(f"\n{'='*60}")
    print(f"  Training: {cfg['display_name']}")
    print(f"  Data:     {cfg['data_dir']}")
    print(f"  Output:   {lora_dir.parent}")
    print(f"  Trigger:  {cfg['trigger']}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "train_lora.py",
        "--data-dir", cfg["data_dir"],
        "--output-dir", str(lora_dir.parent),
        "--max-steps", "500",
        "--lora-rank", "8",
        "--batch-size", "1",
        "--gradient-accumulation", "4",
        "--learning-rate", "1e-4",
        "--save-every", "500",
        "--log-every", "25",
        "--trigger-phrase", cfg["trigger"],
    ]

    result = subprocess.run(cmd, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})

    if result.returncode != 0:
        print(f"[{cfg['display_name']}] Training FAILED (exit code {result.returncode})")
    else:
        print(f"[{cfg['display_name']}] Training complete!")


def main():
    for key, cfg in STYLES.items():
        data_path = Path(cfg["data_dir"])
        images = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
        if len(images) < 10:
            print(f"[{cfg['display_name']}] Skipping — only {len(images)} images in {cfg['data_dir']} (need at least 10). Run download_dataset.py first.")
            continue
        train_style(key, cfg)

    print("\n" + "=" * 60)
    print("All training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
