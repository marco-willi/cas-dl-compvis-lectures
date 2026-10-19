"""Prepare the sample photographs for the augmentation stability demo.

Resizes course images to the demo's input limits and writes them as PNG (lossless, so
that the browser and Python decode exactly the same pixels) together with `samples.json`.

Usage (from the repository root):
    uv run python demos/cnns/augmentation_stability/tools/prepare_samples.py
"""

import json
from pathlib import Path

from PIL import Image

DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_DIR.parents[2]
IMAGES = REPO_ROOT / "assets" / "images"
OUT_DIR = DEMO_DIR / "samples"

# Must match MAX_SHORT_SIDE / MAX_LONG_SIDE in app.js: images within these limits are
# used by the demo as they are, without any further resampling.
MAX_SHORT_SIDE = 320
MAX_LONG_SIDE = 640

SAMPLES = [
    (
        "elephant",
        "Elephant",
        IMAGES / "representation_learning/examples/elephant_day_01.png",
    ),
    ("cat", "Cat", IMAGES / "classification/cats_and_dogs/cat_01.png"),
    ("shepherd", "German shepherd", IMAGES / "classification/cats_and_dogs/dog_02.png"),
    ("beagle", "Dog with ball", IMAGES / "classification/cats_and_dogs/dog_03.png"),
    ("car", "Car", IMAGES / "representation_learning/examples/car_day_01.png"),
]


def fit(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = min(1.0, MAX_SHORT_SIDE / min(w, h), MAX_LONG_SIDE / max(w, h))
    size = (round(w * scale), round(h * scale))
    return img.resize(size, Image.Resampling.LANCZOS) if size != (w, h) else img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for key, label, path in SAMPLES:
        img = fit(Image.open(path).convert("RGB"))
        img.save(
            OUT_DIR / f"{key}.png", optimize=True
        )  # RGB only: no alpha, no ICC profile
        manifest.append({"key": key, "label": label, "file": f"samples/{key}.png"})
        print(f"{key}: {img.size[0]}x{img.size[1]}")
    (DEMO_DIR / "samples" / "samples.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
