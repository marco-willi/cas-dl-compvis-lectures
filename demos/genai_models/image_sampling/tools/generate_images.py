"""Generates the elephant photos of the demo with the Replicate API (FLUX 1.1 Pro).

Run from the repository root (needs REPLICATE_API_KEY in _environment.local):
    .venv/bin/python demos/genai_models/image_sampling/tools/generate_images.py

Writes assets/images/genai_models/image_sampling/elephant_NN.jpg and prompts.json. Existing
images are skipped, so delete a file to regenerate it. Afterwards, run build_data.py.
"""

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "assets" / "images" / "genai_models" / "image_sampling"
MODEL = "black-forest-labs/flux-1.1-pro"
STYLE = "Wildlife photograph, natural light, realistic, sharp focus, the elephant is clearly visible and roughly centered."

PROMPTS = [
    "An African elephant walking through golden savanna grass at sunrise, warm orange light, acacia trees in the background.",
    "An African elephant standing in a shallow river, splashing water with its trunk, green riverbank behind.",
    "A mother elephant with a small calf walking on a dusty dirt track, dry bush on both sides.",
    "An African elephant in front of snow-capped Mount Kilimanjaro, clear blue sky, short green grass.",
    "An Asian elephant in a dense green tropical forest, soft light falling through the trees.",
    "An African elephant under dark dramatic storm clouds, light rain, savanna landscape.",
    "A silhouette of an African elephant against a red-orange sunset sky, flat horizon.",
    "An African elephant covered in reddish dust, standing on red earth, bright midday sun.",
    "A small herd of African elephants crossing a wide open plain, distant mountains, late afternoon light.",
]


def load_token() -> str:
    for line in (ROOT / "_environment.local").read_text().splitlines():
        if line.startswith("REPLICATE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("REPLICATE_API_KEY not found in _environment.local")


def generate(prompt: str, seed: int, token: str) -> bytes:
    response = requests.post(
        f"https://api.replicate.com/v1/models/{MODEL}/predictions",
        json={
            "input": {
                "prompt": f"{prompt} {STYLE}",
                "aspect_ratio": "16:9",
                "output_format": "jpg",
                "output_quality": 90,
                "seed": seed,
                "safety_tolerance": 2,
            }
        },
        headers={"Authorization": f"Bearer {token}", "Prefer": "wait"},
        timeout=120,
    )
    response.raise_for_status()
    prediction = response.json()
    if prediction.get("status") != "succeeded":
        raise RuntimeError(
            f"prediction {prediction.get('id')}: {prediction.get('status')} {prediction.get('error')}"
        )
    output = prediction["output"]
    image = requests.get(output if isinstance(output, str) else output[0], timeout=120)
    image.raise_for_status()
    return image.content


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    token = load_token()
    manifest = []
    for k, prompt in enumerate(PROMPTS, start=1):
        path = OUT / f"elephant_{k:02d}.jpg"
        manifest.append(
            {
                "file": path.name,
                "model": MODEL,
                "seed": k,
                "prompt": f"{prompt} {STYLE}",
            }
        )
        if path.exists():
            print(f"skip {path.name}")
            continue
        path.write_bytes(generate(prompt, k, token))
        print(f"wrote {path.name}")
    (OUT / "prompts.json").write_text(json.dumps(manifest, indent=2) + "\n")
