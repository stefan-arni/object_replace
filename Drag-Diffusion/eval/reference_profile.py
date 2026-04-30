"""
Lightweight image-profile scoring for the lawn/dog reference scene.

The uploaded target is visually dominated by bright green grass with a small
cream dog lying near the lower-middle of the lawn. These checks are intentionally
model-free so they can run even when Stable Diffusion weights are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class LawnDogProfile:
    grass_coverage: Tuple[float, float] = (0.55, 0.99)
    dog_coverage: Tuple[float, float] = (0.006, 0.075)
    dog_center_x: Tuple[float, float] = (0.35, 0.68)
    dog_center_y: Tuple[float, float] = (0.42, 0.74)
    dog_min_lightness: float = 0.58


ReferenceProfile = LawnDogProfile


@dataclass(frozen=True)
class ReferenceScore:
    overall: float
    grass: float
    dog_presence: float
    dog_position: float
    dog_lightness: float
    metrics: Dict[str, float]


def _range_score(value: float, target: Tuple[float, float]) -> float:
    lo, hi = target
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return max(0.0, value / lo) if lo > 0 else 0.0
    return max(0.0, 1.0 - (value - hi) / max(1.0 - hi, 1e-6))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component in a boolean mask."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best_pixels = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue

            pixels = []
            queue = deque([(y, x)])
            seen[y, x] = True

            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and not seen[ny, nx]
                            and mask[ny, nx]
                        ):
                            seen[ny, nx] = True
                            queue.append((ny, nx))

            if len(pixels) > len(best_pixels):
                best_pixels = pixels

    component = np.zeros_like(mask, dtype=bool)
    if best_pixels:
        ys, xs = zip(*best_pixels)
        component[np.array(ys), np.array(xs)] = True
    return component


def _resize_for_analysis(image: Image.Image, max_side: int = 384) -> Image.Image:
    width, height = image.size
    scale = max(width, height) / max_side
    if scale <= 1:
        return image.convert("RGB")
    return image.convert("RGB").resize(
        (round(width / scale), round(height / scale)),
        Image.Resampling.BILINEAR,
    )


def analyze_lawn_dog_image(
    image: Image.Image,
    profile: LawnDogProfile | None = None,
) -> Dict[str, float]:
    """
    Score how closely an image matches the grass-and-small-cream-dog reference.

    Returns metrics plus an overall score in [0, 1]. Higher is better.
    """
    profile = profile or LawnDogProfile()
    small = _resize_for_analysis(image)
    arr = np.asarray(small).astype(np.float32) / 255.0
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]
    max_ch = arr.max(axis=2)
    min_ch = arr.min(axis=2)
    saturation = (max_ch - min_ch) / np.maximum(max_ch, 1e-6)
    lightness = arr.mean(axis=2)

    grass = (
        (green > 0.28)
        & (green > red * 1.10)
        & (green > blue * 1.10)
        & (saturation > 0.16)
    )
    grass_coverage = float(grass.mean())

    cream_candidates = (
        (lightness > profile.dog_min_lightness)
        & (saturation < 0.42)
        & (red > 0.48)
        & (green > 0.42)
        & (blue > 0.32)
        & ~grass
    )
    dog = _largest_component(cream_candidates)
    dog_coverage = float(dog.mean())

    if dog.any():
        ys, xs = np.where(dog)
        dog_center_x = float(xs.mean() / max(dog.shape[1] - 1, 1))
        dog_center_y = float(ys.mean() / max(dog.shape[0] - 1, 1))
        dog_lightness = float(lightness[dog].mean())
        dog_box = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )
    else:
        dog_center_x = 0.0
        dog_center_y = 0.0
        dog_lightness = 0.0
        dog_box = None

    grass_score = _range_score(grass_coverage, profile.grass_coverage)
    dog_size_score = _range_score(dog_coverage, profile.dog_coverage)
    dog_x_score = _range_score(dog_center_x, profile.dog_center_x)
    dog_y_score = _range_score(dog_center_y, profile.dog_center_y)
    dog_lightness_score = min(1.0, dog_lightness / max(profile.dog_min_lightness, 1e-6))

    overall = float(
        0.35 * grass_score
        + 0.25 * dog_size_score
        + 0.15 * dog_x_score
        + 0.15 * dog_y_score
        + 0.10 * dog_lightness_score
    )

    return {
        "overall_score": overall,
        "analysis_width": dog.shape[1],
        "analysis_height": dog.shape[0],
        "grass_coverage": grass_coverage,
        "dog_coverage": dog_coverage,
        "dog_center_x": dog_center_x,
        "dog_center_y": dog_center_y,
        "dog_lightness": dog_lightness,
        "dog_box": dog_box,
        "dog_center": (round(dog_center_x, 3), round(dog_center_y, 3)),
        "grass_score": grass_score,
        "dog_size_score": dog_size_score,
        "dog_x_score": dog_x_score,
        "dog_y_score": dog_y_score,
        "dog_lightness_score": dog_lightness_score,
        "dog_position_score": (dog_x_score + dog_y_score) / 2.0,
        "verdict": "pass" if overall >= 0.7 else "review",
        "profile": asdict(profile),
    }


def score_image_against_reference(
    image: Image.Image,
    profile: LawnDogProfile | None = None,
) -> ReferenceScore:
    """Return a compact score object for tests and iteration loops."""
    metrics = analyze_lawn_dog_image(image, profile)
    return ReferenceScore(
        overall=metrics["overall_score"],
        grass=metrics["grass_score"],
        dog_presence=metrics["dog_size_score"],
        dog_position=metrics["dog_position_score"],
        dog_lightness=metrics["dog_lightness_score"],
        metrics=metrics,
    )


def score_against_lawn_dog_reference(
    image: Image.Image,
    profile: LawnDogProfile | None = None,
) -> Dict[str, float]:
    """Return JSON/report-friendly metrics for the reference scene."""
    return analyze_lawn_dog_image(image, profile)


def create_lawn_dog_fixture(size: int = 512) -> Image.Image:
    """Create a deterministic synthetic image matching the reference profile."""
    rng = np.random.default_rng(27)
    grass = np.zeros((size, size, 3), dtype=np.float32)
    grass[..., 0] = 55 + rng.normal(0, 5, (size, size))
    grass[..., 1] = 165 + rng.normal(0, 18, (size, size))
    grass[..., 2] = 38 + rng.normal(0, 6, (size, size))

    img = Image.fromarray(np.clip(grass, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)

    # Lawn borders and shrub hints keep the fixture close to the uploaded photo.
    draw.rectangle((0, 0, size, int(size * 0.07)), fill=(42, 92, 38))
    draw.rectangle((0, 0, int(size * 0.16), int(size * 0.35)), fill=(72, 120, 48))
    draw.rectangle((int(size * 0.84), 0, size, size), fill=(72, 95, 56))
    draw.rectangle((0, int(size * 0.78), size, size), fill=(60, 172, 38))

    dog_cx, dog_cy = int(size * 0.51), int(size * 0.58)
    dog_w, dog_h = int(size * 0.20), int(size * 0.075)
    draw.ellipse(
        (dog_cx - dog_w // 2, dog_cy - dog_h // 2, dog_cx + dog_w // 2, dog_cy + dog_h // 2),
        fill=(239, 229, 197),
    )
    draw.ellipse(
        (dog_cx - dog_w // 2 - 10, dog_cy - 18, dog_cx - dog_w // 2 + 35, dog_cy + 22),
        fill=(247, 238, 212),
    )
    draw.ellipse((dog_cx - 48, dog_cy - 12, dog_cx - 42, dog_cy - 6), fill=(35, 30, 25))
    draw.ellipse((dog_cx - 29, dog_cy - 13, dog_cx - 23, dog_cy - 7), fill=(35, 30, 25))
    draw.rectangle((dog_cx + dog_w // 2 - 5, dog_cy + 5, dog_cx + dog_w // 2 + 45, dog_cy + 13), fill=(244, 237, 215))

    return img
