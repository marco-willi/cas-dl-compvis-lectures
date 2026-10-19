"""Export torchvision's ImageNet-pretrained EfficientNet-B0 to ONNX for the browser demo.

Writes into `model/`:
    efficientnet_b0.onnx   the network in evaluation mode (input 1x3x224x224, output logits)
    meta.json              class labels, preprocessing of the selected weights, model size
    reference.json         Python results that the demo's self-check compares against

Usage (from the repository root; onnx packages are only needed for this script):
    uv run --with onnx --with onnxscript --with onnxruntime \
        python demos/cnns/augmentation_stability/tools/export_model.py
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

DEMO_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = DEMO_DIR / "model"
ONNX_PATH = MODEL_DIR / "efficientnet_b0.onnx"

WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1
TOP_K = 5


def synthetic_input(seed: int = 1) -> np.ndarray:
    """Deterministic pseudo-random input in [-1, 1); app.js generates the identical tensor."""
    state = seed
    values = np.empty(3 * 224 * 224, dtype=np.float32)
    for i in range(values.size):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        values[i] = state / 2**32 * 2 - 1
    return values.reshape(1, 3, 224, 224)


def top_k(probs: np.ndarray) -> list[dict]:
    order = np.argsort(-probs)[:TOP_K]
    return [{"index": int(i), "prob": round(float(probs[i]), 6)} for i in order]


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = efficientnet_b0(weights=WEIGHTS).eval()
    preprocess = WEIGHTS.transforms()
    categories = WEIGHTS.meta["categories"]

    # --- Export ---------------------------------------------------------------------
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy,),
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        external_data=False,  # single file, easier to host
    )
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    def run_onnx(x: np.ndarray) -> np.ndarray:
        return session.run(["logits"], {"input": x.astype(np.float32)})[0][0]

    def run_torch(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()[0]

    # --- Verify ONNX against PyTorch, and record references for the browser ---------
    x = synthetic_input()
    logits_torch, logits_onnx = run_torch(x), run_onnx(x)
    max_diff = float(np.abs(logits_torch - logits_onnx).max())
    print(
        f"synthetic input: max |logit| difference PyTorch vs ONNX Runtime = {max_diff:.2e}"
    )
    assert max_diff < 1e-3, "ONNX export does not reproduce the PyTorch model"

    reference = {
        "synthetic": {"seed": 1, "logits": [round(float(v), 5) for v in logits_torch]},
        "samples": {},
    }
    for sample in json.loads((DEMO_DIR / "samples" / "samples.json").read_text()):
        img = Image.open(DEMO_DIR / sample["file"]).convert("RGB")
        x = (
            preprocess(img).unsqueeze(0).numpy()
        )  # the weights' own torchvision transform
        probs_torch, probs_onnx = softmax(run_torch(x)), softmax(run_onnx(x))
        assert np.abs(probs_torch - probs_onnx).max() < 1e-3
        reference["samples"][sample["key"]] = {"top": top_k(probs_torch)}
        best = int(probs_torch.argmax())
        print(f"{sample['key']:>10}: {categories[best]} ({probs_torch[best]:.1%})")

    meta = {
        "model": "torchvision efficientnet_b0",
        "weights": str(WEIGHTS),
        "file": ONNX_PATH.name,
        "bytes": ONNX_PATH.stat().st_size,
        "preprocessing": {
            "resize": preprocess.resize_size[0],
            "crop": preprocess.crop_size[0],
            "interpolation": preprocess.interpolation.value,
            "mean": list(preprocess.mean),
            "std": list(preprocess.std),
        },
        "categories": categories,
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta) + "\n")
    (MODEL_DIR / "reference.json").write_text(json.dumps(reference) + "\n")
    print(
        f"wrote {ONNX_PATH.relative_to(DEMO_DIR)} ({meta['bytes'] / 1e6:.1f} MB), meta.json, reference.json"
    )
    print("preprocessing:", meta["preprocessing"])


if __name__ == "__main__":
    main()
