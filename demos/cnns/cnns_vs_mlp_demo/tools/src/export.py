"""Write the deliverables for the browser demo into export/ (see SPEC.md, "Deliverables").

Usage (from experiments/cnn_on_mnist/):
    python export.py      # after train.py --all and evaluate.py
"""

import base64
import json
import platform
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from common import (
    ARCHS,
    CANVAS,
    CKPT_DIR,
    DIGIT,
    EVAL_DIR,
    EXP_DIR,
    EXPORT_DIR,
    EXPORT_SEED,
    MAX_OFFSET,
    METRICS_DIR,
    N_OFFSETS,
    REGIMES,
    REPO_ROOT,
    SEEDS,
    read_json,
    run_id,
    write_json,
)
from models import layer_spec, load_checkpoint, n_params, predict, round_to_fp16_

from data import eval_subset_indices, load_mnist, offset_range, place, to_float

MODEL_IDS = [f"{arch}_{regime}" for regime in REGIMES for arch in ARCHS]
SAMPLES_PER_CLASS = 5
REFERENCE_OFFSETS = [(0, 0), (20, 20), (28, 0)]  # (dx, dy), applied to sample 0
REFERENCE_N_VALUES = 20
SPEC_CHANNELS = [8, 16, 32]
SPEC_LR = 1e-3
SPEC_EPOCHS = {"tl": 10, "all": 20}
README_CHECK_BEGIN = "<!-- CHECK_EXPORT:BEGIN -->"
README_CHECK_END = "<!-- CHECK_EXPORT:END -->"
COMMANDS = """\
cd experiments/cnn_on_mnist
../../.venv/bin/python train.py --all        # 3 archs x 2 regimes x seeds 0-2
../../.venv/bin/python evaluate.py           # position maps, fp16 check, equivariance
../../.venv/bin/python export.py             # writes export/
../../.venv/bin/python check_export.py       # acceptance checks 3 + 4
# or: bash run_all.sh"""


def write_js(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n")


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def weights_entry(model, dtype: str) -> dict:
    np_dtype = "<f2" if dtype == "float16" else "<f4"
    tensors = {}
    for name, t in model.state_dict().items():
        arr = t.detach().cpu().numpy().astype(np_dtype)  # C order, little-endian
        tensors[name] = {"shape": list(arr.shape), "data": b64(arr)}
    return {"dtype": dtype, "tensors": tensors}


def deviation_notes(cfg: dict, summary: dict) -> list[str]:
    notes = [
        "Produced in the dl-cv-exercises repo (experiments/cnn_on_mnist/, flat scripts instead "
        "of a separate repo with a src/mnist_shift package); the code is in export/src/.",
        "The mlp layer list starts with an explicit {type: flatten} op (canvas -> 3136 vector).",
    ]
    if cfg["channels"] != SPEC_CHANNELS:
        notes.append(
            f"CNN trunk widened from {SPEC_CHANNELS} to {cfg['channels']} channels for both "
            "cnn_fc and cnn_gap: with 8/16/32, cnn_gap_tl reached only 91.6 % val accuracy."
        )
    if cfg["lr"] != SPEC_LR or cfg["epochs"] != SPEC_EPOCHS[cfg["regime"]]:
        notes.append(
            f"Training changed for all six models: AdamW lr {cfg['lr']:g} (spec {SPEC_LR:g}), "
            f"tl epochs {cfg['epochs']} (spec 10); all epochs 20 as in the spec. With the widened "
            "trunk, lr 1e-3 and 10 epochs, cnn_gap_tl was still underfitting (94.5 % val)."
        )
    notes.append(
        f"Position maps use the {summary['map_size']}-image "
        f"{'stratified test subset' if summary['map_set'] == 'subset' else 'full test set'}."
    )
    notes.append(
        "test_acc_own_regime: full 10k test set, one fixed offset per image drawn from the "
        "model's regime; test_acc_all_positions: mean of the position map."
    )
    notes.append(
        f"Weights exported as {summary['export_dtype']} (max |fp16 - fp32| accuracy difference "
        f"{summary['fp16_max_abs_diff_pp']} pp); all reported numbers use the exported weights."
    )
    return notes


def pick_samples(models: dict, test_x, test_y, subset: list[int]) -> list[int]:
    """5 per class from the subset, correct for all tl models at (0, 0); classes interleaved
    (sample i has label i % 10)."""
    idx = torch.tensor(subset, device=test_x.device)
    canvas = place(test_x[idx], 0, 0)
    ok = torch.ones(len(idx), dtype=torch.bool, device=test_x.device)
    for arch in ARCHS:
        pred, _ = predict(models[f"{arch}_tl"], canvas)
        ok &= pred == test_y[idx]
    per_class = {c: [] for c in range(10)}
    for i, good in zip(subset, ok.tolist(), strict=True):
        label = int(test_y[i])
        if good and len(per_class[label]) < SAMPLES_PER_CLASS:
            per_class[label].append(i)
    return [per_class[c][k] for k in range(SAMPLES_PER_CLASS) for c in range(10)]


@torch.no_grad()
def reference_entry(model, digit: torch.Tensor, is_cnn: bool) -> list[dict]:
    out = []
    for dx, dy in REFERENCE_OFFSETS:
        logits, inter = model.forward_with_intermediates(place(digit, dx, dy)[None])
        entry = {
            "sample": 0,
            "dx": dx,
            "dy": dy,
            "logits": [float(v) for v in logits[0].cpu()],
        }
        if is_cnn:
            for layer in ("conv1", "conv3"):
                values = inter[layer][0].flatten()[:REFERENCE_N_VALUES]
                entry[f"{layer}_first{REFERENCE_N_VALUES}"] = [
                    float(v) for v in values.cpu()
                ]
        out.append(entry)
    return out


def check_verdicts(
    manifest_models: dict, patterns: dict
) -> tuple[list[str], list[str]]:
    """Acceptance checks 1 and 2 as markdown lines (check 2 is reported, never tuned)."""
    c1 = []
    for arch in ARCHS:
        acc = manifest_models[f"{arch}_tl"]["test_acc_own_regime"]
        c1.append(
            f"- `{arch}_tl`: {100 * acc:.2f} % → {'PASS' if acc >= 0.97 else 'FAIL'}"
        )

    def line(mid, expected, ok):
        p = patterns[mid]
        return (
            f"| `{mid}` | {expected} | {p['at_origin']:.3f} | {p['near_top_left']:.3f} | "
            f"{p['far_bottom_right']:.3f} | {p['mean']:.3f} | {p['min']:.3f} | "
            f"{p['dx_dy_multiple_of_4']:.3f} | {p['other_offsets']:.3f} | {'yes' if ok else '**no**'} |"
        )

    def fails_away(mid):  # high at the training position, ~chance far away
        return p[mid]["at_origin"] >= 0.95 and p[mid]["far_bottom_right"] <= 0.25

    def high_everywhere(mid):
        return p[mid]["mean"] >= 0.85 and p[mid]["min"] >= 0.75

    p = patterns
    all_means = {a: p[f"{a}_all"]["mean"] for a in ARCHS}
    period4 = p["cnn_gap_tl"]["dx_dy_multiple_of_4"] - p["cnn_gap_tl"]["other_offsets"]
    c2 = [
        "Criteria: *fails away* = (0,0) ≥ 0.95 and far BR ≤ 0.25; *high everywhere* = "
        "mean ≥ 0.85 and min ≥ 0.75.",
        "",
        "| model | expected | (0,0) | near TL (dx,dy ≤ 3) | far BR (dx,dy ≥ 21) | mean | min | "
        "dx,dy mult. of 4 | other | matches |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|---|",
        line("mlp_tl", "high near TL, ~chance far away", fails_away("mlp_tl")),
        line("cnn_fc_tl", "same failure as mlp_tl", fails_away("cnn_fc_tl")),
        line(
            "cnn_gap_tl",
            "high over most of the map, period-4 pattern likely",
            p["cnn_gap_tl"]["mean"] >= 0.8,
        ),
        line(
            "mlp_all",
            "high everywhere, lower than CNNs",
            high_everywhere("mlp_all")
            and all_means["mlp"] < min(all_means["cnn_fc"], all_means["cnn_gap"]),
        ),
        line("cnn_fc_all", "high everywhere", high_everywhere("cnn_fc_all")),
        line("cnn_gap_all", "high everywhere", high_everywhere("cnn_gap_all")),
        "",
        "Observations:",
        "",
        f"- Period-4 pattern of `cnn_gap_tl`: {'**not visible**' if abs(period4) < 0.01 else 'visible'} "
        f"(offsets with dx, dy multiples of 4: {p['cnn_gap_tl']['dx_dy_multiple_of_4']:.3f}, "
        f"others: {p['cnn_gap_tl']['other_offsets']:.3f}; min over the map "
        f"{p['cnn_gap_tl']['min']:.3f}). Reported as is, nothing was changed to provoke it.",
        f"- `mlp_tl` and `cnn_fc_tl` already break down within a few pixels: (0,0) "
        f"{p['mlp_tl']['at_origin']:.3f} / {p['cnn_fc_tl']['at_origin']:.3f}, mean over "
        f"dx, dy ≤ 3 {p['mlp_tl']['near_top_left']:.3f} / {p['cnn_fc_tl']['near_top_left']:.3f}.",
        f"- `mlp_all` is weakest near the canvas corners: (0,0) {p['mlp_all']['at_origin']:.3f}, "
        f"min {p['mlp_all']['min']:.3f}, mean {p['mlp_all']['mean']:.3f}.",
    ]
    return c1, c2


def git_commit() -> str:
    def run(*cmd):
        return subprocess.run(  # nosec B603
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    head = run("git", "rev-parse", "HEAD")
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    return f"{branch}@{head}"


def write_readme(
    manifest: dict, log: dict, summary: dict, c1: list[str], c2: list[str]
) -> None:
    rows = []
    for mid in MODEL_IDS:
        m, s = manifest["models"][mid], log["models"][mid]["seed_spread"]
        own, allp = s["test_acc_own_regime"], s["test_acc_all_positions"]
        rows.append(
            f"| `{mid}` | {m['params']:,} | {m['best_epoch']} | {100 * m['test_acc_own_regime']:.2f} | "
            f"{100 * m['test_acc_all_positions']:.2f} | {100 * own['mean']:.2f} ± {100 * own['std']:.2f} | "
            f"{100 * allp['mean']:.2f} ± {100 * allp['std']:.2f} |"
        )
    eq_rows = [
        f"| {r['layer']} | {r['t_px']} | {r['feature_shift']} | {'yes' if r['exact_shift'] else 'no (nearest)'} | "
        f"{r['rel_err_mean']:.4f} | {r['rel_err_max']:.4f} |"
        for r in summary.get("equivariance_cnn_gap_tl", {}).get("rows", [])
    ]
    notes = "\n".join(f"- {n}" for n in manifest["notes"])
    text = f"""# Shifted MNIST: MLP vs. CNN+FC vs. CNN+GAP — export

Deliverables for the browser demo `demos/cnns/cnns_vs_mlp_demo/` of the course repo
(`cas-dl-compvis-lectures-hs2026`), produced as specified in `SPEC.md` (next to this
experiment's code, copied into `src/`).

## Provenance

- Repository: `dl-cv-exercises`, `{git_commit()}` (the experiment folder is gitignored and
  not committed; `src/` holds the exact scripts that produced this export)
- Python {platform.python_version()}, torch {torch.__version__}, torchvision {torchvision.__version__}, numpy {np.__version__}
- Hardware: {log["hardware"]}; total training wall-clock {log["total_train_time_s"] / 60:.1f} min (18 runs; the seed-0 runs of
  `cnn_fc_all` and `cnn_gap_tl` shared the GPU with other jobs and took 3-5x longer than the
  ~220 s of the other CNN runs)
- Weights dtype: `{manifest["dtype"]}`

```bash
{COMMANDS}
```

## Deviations from the spec

{notes}

## Results (seed 0 = exported, exported weights)

| model | params | best epoch | test acc own regime (%) | test acc all positions (%) | own, seeds 0-2 (%) | all pos., seeds 0-2 (%) |
|---|--:|--:|--:|--:|--:|--:|
{chr(10).join(rows)}

## Equivariance of `cnn_gap_tl` (100 test digits, shift right, after ReLU)

Relative error ||f(shift_t x) - shift_s f(x)|| / ||shift_s f(x)||, border of s px cropped.
Rows with "no (nearest)" are outside the spec (t not a multiple of 4 for conv3) and compare
with the nearest integer feature shift.

| layer | t (px) | feature shift s | exact | mean | max |
|---|--:|--:|---|--:|--:|
{chr(10).join(eq_rows)}

conv1 is exactly equivariant (error at float precision). For conv3 the exact shifts
(t = 4, 8) are not exactly 0 either: the background after conv1 is relu(bias), not 0, while the
zero padding of conv2/conv3 is 0, so border effects reach further in than the s-px crop. The
non-multiples of 4 are clearly worse (pooling breaks exact equivariance).

## Acceptance checks

### 1. ≥ 97 % test accuracy in the tl regime

{chr(10).join(c1)}

### 2. Expected patterns on the position maps (reported, not tuned)

{chr(10).join(c2)}

{README_CHECK_BEGIN}
### 3 + 4. Run `check_export.py` to fill in this section.
{README_CHECK_END}
"""
    (EXPORT_DIR / "README.md").write_text(text)


def main() -> None:
    summary = read_json(EVAL_DIR / "summary.json")
    dtype = summary["export_dtype"]
    device = torch.device("cpu")  # export from CPU for platform-independent references
    images, labels = load_mnist(train=False)
    test_x, test_y = to_float(images), labels
    subset = eval_subset_indices()

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    (EXPORT_DIR / "weights").mkdir(parents=True)

    models, configs = {}, {}
    for mid in MODEL_IDS:
        arch, regime = mid.rsplit("_", 1)
        model, cfg = load_checkpoint(
            CKPT_DIR / f"{run_id(arch, regime, EXPORT_SEED)}.pt", device
        )
        if dtype == "float16":
            round_to_fp16_(model)
        models[mid], configs[mid] = model, cfg
        write_js(
            EXPORT_DIR / "weights" / f"{mid}.js",
            "window.MNIST_SHIFT_WEIGHTS = window.MNIST_SHIFT_WEIGHTS || {};\n"
            f'window.MNIST_SHIFT_WEIGHTS["{mid}"] = {dumps(weights_entry(model, dtype))};',
        )

    # Numbers and maps of the exported weights (fp16-rounded if exported as float16).
    evals = {}
    for mid in MODEL_IDS:
        arch, regime = mid.rsplit("_", 1)
        ev = read_json(EVAL_DIR / f"{run_id(arch, regime, EXPORT_SEED)}.json")
        evals[mid] = ev["fp16"] if dtype == "float16" else ev

    tl_cfg = configs["mlp_tl"]
    manifest = {
        "version": 1,
        "canvas": CANVAS,
        "digit": DIGIT,
        "max_offset": MAX_OFFSET,
        "value_range": [0, 1],
        "dtype": dtype,
        "regimes": {
            r: {"offsets": list(offset_range(r, tl_cfg["tl_max_offset"]))}
            for r in REGIMES
        },
        "position_maps": {
            "subset_size": summary["map_size"],
            "set": summary["map_set"],
        },
        "models": {},
        "notes": deviation_notes(configs["cnn_gap_tl"], summary),
    }
    for mid in MODEL_IDS:
        cfg, model = configs[mid], models[mid]
        metrics = read_json(
            METRICS_DIR / f"{run_id(cfg['arch'], cfg['regime'], EXPORT_SEED)}.json"
        )
        manifest["models"][mid] = {
            "arch": cfg["arch"],
            "regime": cfg["regime"],
            "seed": EXPORT_SEED,
            "params": n_params(model),
            "best_epoch": metrics["best_epoch"],
            "test_acc_own_regime": round(evals[mid]["test_acc_own_regime"], 4),
            "test_acc_all_positions": round(evals[mid]["test_acc_all_positions"], 4),
            "layers": layer_spec(cfg["arch"], tuple(cfg["channels"])),
        }
    write_js(
        EXPORT_DIR / "manifest.js", f"window.MNIST_SHIFT_MANIFEST = {dumps(manifest)};"
    )

    sample_idx = pick_samples(models, test_x, test_y, subset)
    samples = {
        "size": DIGIT,
        "items": [
            {
                "label": int(labels[i]),
                "test_index": i,
                "pixels": b64(images[i].numpy().astype(np.uint8)),
            }
            for i in sample_idx
        ],
    }
    write_js(
        EXPORT_DIR / "samples.js", f"window.MNIST_SHIFT_SAMPLES = {dumps(samples)};"
    )

    maps = {
        "offsets": N_OFFSETS,
        "subset_size": summary["map_size"],
        "maps": {
            mid: {
                k: np.round(np.array(evals[mid][k]), 3).tolist()
                for k in ("acc", "conf")
            }
            for mid in MODEL_IDS
        },
    }
    write_js(
        EXPORT_DIR / "position_maps.js", f"window.MNIST_SHIFT_MAPS = {dumps(maps)};"
    )

    digit0 = to_float(images[sample_idx[0]])
    reference = {
        "tolerance": 1e-3,
        "weights_dtype": dtype,
        "sample_test_index": sample_idx[0],
        "models": {
            mid: reference_entry(models[mid], digit0, mid.startswith("cnn"))
            for mid in MODEL_IDS
        },
    }
    write_js(
        EXPORT_DIR / "reference.js",
        f"window.MNIST_SHIFT_REFERENCE = {dumps(reference)};",
    )

    log = {"models": {}, "fp16_check": {}, "notes": manifest["notes"]}
    total_time, hw = 0.0, None
    for mid in MODEL_IDS:
        arch, regime = mid.rsplit("_", 1)
        runs = {
            s: read_json(METRICS_DIR / f"{run_id(arch, regime, s)}.json") for s in SEEDS
        }
        total_time += sum(r["wall_clock_s"] for r in runs.values())
        hw = runs[EXPORT_SEED]["hardware"]
        s0 = runs[EXPORT_SEED]
        log["models"][mid] = {
            "config": s0["config"],
            "params": s0["params"],
            "best_epoch": s0["best_epoch"],
            "best_val_acc": s0["best_val_acc"],
            "history": s0["history"],
            "final_exported": {
                k: manifest["models"][mid][k]
                for k in ("test_acc_own_regime", "test_acc_all_positions")
            },
            "pattern_exported": evals[mid]["pattern"],
            "seed_spread": summary["models"][mid]["seed_spread"],
            "other_seeds": {
                s: {k: r[k] for k in ("best_epoch", "best_val_acc", "wall_clock_s")}
                for s, r in runs.items()
                if s != EXPORT_SEED
            },
            "wall_clock_s": s0["wall_clock_s"],
            "hardware": hw,
        }
        log["fp16_check"][mid] = summary["models"][mid]["fp16_diff_pp"]
    log["fp16_check"]["max_abs_diff_pp"] = summary["fp16_max_abs_diff_pp"]
    log["fp16_check"]["export_dtype"] = dtype
    log["equivariance_cnn_gap_tl"] = summary.get("equivariance_cnn_gap_tl")
    log["total_train_time_s"] = round(total_time, 1)
    log["hardware"] = hw
    write_json(EXPORT_DIR / "training_log.json", log)

    c1, c2 = check_verdicts(
        manifest["models"], {mid: evals[mid]["pattern"] for mid in MODEL_IDS}
    )
    write_readme(manifest, log, summary, c1, c2)

    src = EXPORT_DIR / "src"
    src.mkdir()
    for f in [
        *sorted(EXP_DIR.glob("*.py")),
        EXP_DIR / "run_all.sh",
        EXP_DIR / "SPEC.md",
    ]:
        if f.exists():
            shutil.copy2(f, src / f.name)

    print(f"wrote {EXPORT_DIR} (dtype {dtype})")
    print("\n".join(c1))
    print("\n".join(c2))


if __name__ == "__main__":
    sys.exit(main())
