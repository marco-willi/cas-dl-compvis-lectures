"""Acceptance checks 3 and 4: a standalone NumPy forward pass (no torch) that reads the
exported .js files, executes each manifest `layers` list literally and compares with
reference.js; plus the size budget. Writes the results into export/README.md.

Usage (from experiments/cnn_on_mnist/):
    python check_export.py [--export-dir export]
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CANVAS, DIGIT = 56, 28
TOLERANCE = 1e-3
MAX_WEIGHTS_BYTES = 6 * 1024 * 1024
MAX_EXPORT_BYTES = 8 * 1024 * 1024
README_CHECK_BEGIN = "<!-- CHECK_EXPORT:BEGIN -->"
README_CHECK_END = "<!-- CHECK_EXPORT:END -->"


def read_js(path: Path) -> dict:
    """Parse the JSON object assigned in a `window.X = {...};` file (last assignment)."""
    text = path.read_text()
    start = text.index("= {") + 2
    return json.loads(text[start : text.rindex("}") + 1])


def decode(entry: dict, dtype: str) -> np.ndarray:
    raw = base64.b64decode(entry["data"])
    arr = np.frombuffer(raw, dtype="<f2" if dtype == "float16" else "<f4")
    return arr.reshape(entry["shape"]).astype(np.float32)


def place(digit: np.ndarray, dx: int, dy: int) -> np.ndarray:
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    canvas[dy : dy + DIGIT, dx : dx + DIGIT] = digit
    return canvas


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int) -> np.ndarray:
    k = w.shape[-1]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (k, k), axis=(1, 2))  # (C, H, W, k, k)
    return np.einsum("chwij,ocij->ohw", win, w, optimize=True) + b[:, None, None]


def forward(layers: list[dict], params: dict, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Execute the manifest op list; returns logits and conv outputs after their ReLU."""
    inter, last_conv = {}, None
    for op in layers:
        kind = op["type"]
        if kind == "conv2d":
            x = x[None] if x.ndim == 2 else x
            x = conv2d(
                x,
                params[f"{op['name']}.weight"],
                params[f"{op['name']}.bias"],
                op["pad"],
            )
            last_conv = op["name"]
            continue
        if kind == "relu":
            x = np.maximum(x, 0)
            if last_conv:
                inter[last_conv] = x
        elif kind == "maxpool2d":
            k = op["k"]
            c, h, w = x.shape
            x = x.reshape(c, h // k, k, w // k, k).max(axis=(2, 4))
        elif kind == "gap":
            x = x.mean(axis=(1, 2), keepdims=True)
        elif kind == "flatten":
            x = x.reshape(-1)
        elif kind == "linear":
            x = params[f"{op['name']}.weight"] @ x + params[f"{op['name']}.bias"]
        else:
            raise ValueError(f"unknown op {kind!r}")
        last_conv = None
    return x, inter


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--export-dir", type=Path, default=Path(__file__).resolve().parent / "export"
    )
    args = parser.parse_args()
    export = args.export_dir

    manifest = read_js(export / "manifest.js")
    samples = read_js(export / "samples.js")
    reference = read_js(export / "reference.js")
    pixels = [
        np.frombuffer(base64.b64decode(it["pixels"]), dtype=np.uint8)
        .reshape(DIGIT, DIGIT)
        .astype(np.float32)
        / np.float32(255)
        for it in samples["items"]
    ]

    lines, ok_all = [], True
    for mid, spec in manifest["models"].items():
        w = read_js(export / "weights" / f"{mid}.js")
        params = {name: decode(t, w["dtype"]) for name, t in w["tensors"].items()}
        max_err = 0.0
        for ref in reference["models"][mid]:
            logits, inter = forward(
                spec["layers"],
                params,
                place(pixels[ref["sample"]], ref["dx"], ref["dy"]),
            )
            max_err = max(
                max_err, float(np.abs(logits - np.array(ref["logits"])).max())
            )
            for layer in ("conv1", "conv3"):
                key = f"{layer}_first20"
                if key in ref:
                    got = inter[layer].reshape(-1)[:20]
                    max_err = max(
                        max_err, float(np.abs(got - np.array(ref[key])).max())
                    )
        n_params = sum(p.size for p in params.values())
        ok = max_err <= TOLERANCE and n_params == spec["params"]
        line = f"- `{mid}`: max |numpy - reference| = {max_err:.2e}, params {n_params:,} → {'PASS' if ok else 'FAIL'}"
        if mid.endswith("_tl"):
            correct = sum(
                int(
                    np.argmax(forward(spec["layers"], params, place(p, 0, 0))[0])
                    == it["label"]
                )
                for p, it in zip(pixels, samples["items"], strict=True)
            )
            line += f"; samples correct at (0,0): {correct}/{len(pixels)}"
            ok = ok and correct == len(pixels)
        ok_all &= ok
        lines.append(line)

    labels = [it["label"] for it in samples["items"]]
    ok_samples = len(labels) == 50 and all(labels.count(c) == 5 for c in range(10))
    lines.append(
        f"- samples.js: {len(labels)} digits, 5 per class → {'PASS' if ok_samples else 'FAIL'}"
    )
    ok_all &= ok_samples

    weights_bytes = dir_size(export / "weights")
    export_bytes = dir_size(export)
    ok_size = weights_bytes <= MAX_WEIGHTS_BYTES and export_bytes <= MAX_EXPORT_BYTES
    size_lines = [
        f"- `weights/`: {weights_bytes / 1e6:.2f} MB (≤ 6 MB) → {'PASS' if weights_bytes <= MAX_WEIGHTS_BYTES else 'FAIL'}",
        f"- `export/` total: {export_bytes / 1e6:.2f} MB (≤ 8 MB) → {'PASS' if export_bytes <= MAX_EXPORT_BYTES else 'FAIL'}",
    ]
    ok_all &= ok_size

    section = "\n".join(
        [
            README_CHECK_BEGIN,
            "### 3. NumPy forward pass reproduces `reference.js` (tolerance 1e-3)",
            "",
            *lines,
            "",
            "### 4. Sizes",
            "",
            *size_lines,
            README_CHECK_END,
        ]
    )
    readme = export / "README.md"
    if readme.exists():
        text = readme.read_text()
        head, rest = text.split(README_CHECK_BEGIN, 1)
        readme.write_text(head + section + rest.split(README_CHECK_END, 1)[1])

    print("\n".join(lines + size_lines))
    print("ALL CHECKS PASS" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
