# Spec: Train MLPs and CNNs on Shifted MNIST — Handoff for the Training Agent

> **Created:** 2026-10-22
> **Status:** Final (decisions of 2026-10-22: exact top-left position, 3 architectures, no hand-drawn input, separate repo)
> **For:** a coding agent working in a new, separate repository (GPU optional, see "Compute")
> **Consumer:** browser demo `demos/cnns/cnns_vs_mlp_demo/` in the course repository
> (`cas-dl-compvis-lectures-hs2026`), embedded in `pages/lectures/cnns.qmd`
> **Parent plan:** `.claude/plans/PLAN_MNIST_TRANSLATION_DEMO.md` (course repo)

This file is self-contained: copy it into the training repository and follow it. Everything
under "Deliverables" is a hard contract — the browser code is written against it.

## Repository setup

Create a new, standalone repository (suggested name `cnn-translation-mnist`); it must not
depend on the course repository or on the `dl_cv_lectures` package. Reimplement the few
helpers needed (placement, training loop) — they are short.

```
cnn-translation-mnist/
  pyproject.toml          uv project; Python ≥ 3.11; deps: torch, torchvision, numpy
  README.md               setup, commands, link back to this spec
  SPEC.md                 copy of this file (unchanged)
  src/mnist_shift/
    data.py               place(), ShiftedMNIST dataset (regime, fixed eval subset)
    models.py             MLP, CNN trunk, CNN+FC head, CNN+GAP head
    train.py              CLI: --model {mlp,cnn_fc,cnn_gap} --regime {tl,all} --seed N
    evaluate.py           position maps, equivariance table
    export.py             writes export/ (float16, base64, .js wrappers)
    check_export.py       NumPy forward pass vs. reference.js
  checkpoints/            (gitignored)
  data/                   (gitignored; MNIST download)
  export/                 deliverables, committed
```

Commands (put them in a `Makefile` or `README.md`):

```bash
uv sync
uv run python -m mnist_shift.train --all            # 6 models × seeds 0–2
uv run python -m mnist_shift.evaluate               # position maps + equivariance
uv run python -m mnist_shift.export                 # writes export/
uv run python -m mnist_shift.check_export           # acceptance check 3
```

Use the GPU if available (`cuda` → `mps` → `cpu`), but the code must also run on CPU. Fix all
seeds (`torch`, `numpy`, `random`, DataLoader generator) so reruns reproduce the export.
Commit `export/` and record the commit hash in `export/README.md`.

## Purpose

Students should *see* that

1. an **MLP** trained on digits in one position fails when the digit moves (no translation
   invariance, no equivariance),
2. a **CNN with a fully connected head** has translation-*equivariant* feature maps, but its
   flatten + FC head is position-dependent, so it fails too,
3. a **CNN with global average pooling (GAP)** — same convolutional trunk, different head —
   is (approximately) translation-*invariant* and keeps working,
4. the MLP *can* learn all positions if it sees them during training (augmentation), at the cost
   of parameters and data.

This extends `archive/03_cnns/cnns.ipynb` of the course repo (MNIST in the top-left quadrant of a
56×56 canvas, tested on the bottom-right quadrant).

## Data

- Source: `torchvision.datasets.MNIST` (train 60k, test 10k). Pixel values scaled to `[0, 1]`
  (`x / 255`). **No mean/std normalization** — the browser feeds raw `[0, 1]` values.
- Canvas: `56 × 56`, background 0. A `28 × 28` digit is pasted with its top-left corner at
  offset `(dx, dy)`, `dx, dy ∈ {0, …, 28}` (column, row). Integer offsets only.
- Implement one function `place(digit_28x28, dx, dy) -> canvas_56x56` and use it everywhere
  (training, evaluation, references) so the browser can replicate it exactly.
- Training regimes:
  - `tl` ("top-left only"): `dx = dy = 0` exactly, as in the archive notebook (no jitter). Keep the
    range as a CLI parameter (`--tl-max-offset`, default `0`) and report it in the manifest.
  - `all` ("everywhere"): `dx, dy` uniform in `{0, …, 28}`.
- Hold out 5k of the 60k training images as validation (fixed seed); test set only for final
  evaluation.

## Models

Only these layer types (the browser implements exactly these):
`Conv2d` (stride 1, `padding = k // 2`, with bias), `ReLU`, `MaxPool2d(2, 2)`,
`AdaptiveAvgPool2d(1)`, `Flatten`, `Linear`. **No BatchNorm, no Dropout at export, no
other layers.** All feature maps keep the input coordinate frame thanks to "same" padding,
which matters for visualizing equivariance.

| id        | Architecture                                                                    |
| --------- | ------------------------------------------------------------------------------- |
| `mlp`     | Flatten(3136) → Linear(3136, 128) → ReLU → Linear(128, 10)                      |
| `cnn_fc`  | **trunk** → Flatten(32·14·14 = 6272) → Linear(6272, 32) → ReLU → Linear(32, 10) |
| `cnn_gap` | **trunk** → AdaptiveAvgPool2d(1) → Flatten(32) → Linear(32, 10)                 |

**trunk** (identical for both CNNs):
`Conv2d(1, 8, 5) → ReLU → MaxPool(2) → Conv2d(8, 16, 5) → ReLU → MaxPool(2) → Conv2d(16, 32, 3) → ReLU`
→ shape `32 × 14 × 14`.

Name the modules so the exported layer names are stable: `conv1, conv2, conv3, fc1, fc2`
(`cnn_gap` has `conv1–3, fc1`; `mlp` has `fc1, fc2`).

Train all 3 models × 2 regimes = **6 models**: `mlp_tl, cnn_fc_tl, cnn_gap_tl, mlp_all, cnn_fc_all, cnn_gap_all`.

If `cnn_gap_tl` reaches < 95 % test accuracy in its own regime, it is acceptable to widen the
trunk (e.g. 16/32/64 channels) — but then change it for **both** CNNs and report it. Do not
change anything else without noting it in `manifest.json` → `notes`.

## Training

- Optimizer AdamW (lr 1e-3, weight decay 1e-4 default), batch size 256, cross-entropy.
- Epochs: up to 10 for `tl`, up to 20 for `all`; keep the checkpoint with best validation
  accuracy *in the model's own regime*. Positions are re-sampled every epoch (on-the-fly
  augmentation).
- Fixed seed per model (`seed = 0`); additionally train seeds 1 and 2 and report the spread of
  the headline numbers in `training_log.json` (only seed 0 is exported).
- Log per epoch: train loss/acc, val acc.

### Compute

All six models are small (≤ 0.5 M parameters, 56×56 grayscale). A GPU makes this take minutes;
a CPU also works (expect roughly tens of minutes in total). Do not optimize for speed.

## Evaluation: position-accuracy maps

For every model, evaluate test accuracy at **every** offset `(dx, dy) ∈ {0..28}²` (29 × 29 = 841
positions) on the same fixed subset of **2000 test images** (stratified, 200 per class, fixed
seed; store the indices). Use the full 10k if it fits the time budget, and say which in the
manifest.

Also record, per model, the **mean max-softmax confidence** per position (shows "confidently
wrong" for the MLP).

## Equivariance check (for the lecture text)

For `cnn_gap_tl`, compute for 100 test digits and shifts `t ∈ {1, 2, 3, 4, 8}` px (right):

`err(t) = ‖conv1(shift_t(x)) − shift_t(conv1(x))‖ / ‖conv1(x)‖`, same for `conv3` (after the two
poolings shifts must be divided by 4 → only use `t ∈ {4, 8}` for conv3), cropping a border of
`t` px to exclude padding effects. Report as a small table in `training_log.json`. Expected:
≈ 0 for conv1; ≈ 0 for conv3 when `t` is a multiple of 4, clearly > 0 otherwise (pooling
breaks exact equivariance — useful teaching point, see Zhang 2019, arXiv:1904.11486).

## Deliverables (hard contract)

Write everything into one folder `export/` with this layout. **All files consumed by the
browser are `.js` files that assign to a global**, so the demo also works when the page is
opened via `file://` (no `fetch`).

```
export/
  manifest.js            window.MNIST_SHIFT_MANIFEST = {...}
  weights/
    mlp_tl.js            window.MNIST_SHIFT_WEIGHTS = window.MNIST_SHIFT_WEIGHTS || {};
    ...                  window.MNIST_SHIFT_WEIGHTS["mlp_tl"] = {...};
  samples.js             window.MNIST_SHIFT_SAMPLES = {...}
  position_maps.js       window.MNIST_SHIFT_MAPS = {...}
  reference.js           window.MNIST_SHIFT_REFERENCE = {...}
  training_log.json      (for humans; not loaded by the demo)
  README.md              how it was produced, exact commands, library versions, git commit
  src/                   copy of the training/export scripts (for reproducibility)
```

### `manifest.js`

```json
{
  "version": 1,
  "canvas": 56, "digit": 28, "max_offset": 28,
  "value_range": [0, 1],
  "regimes": {"tl": {"offsets": [0, 0]}, "all": {"offsets": [0, 28]}},
  "models": {
    "mlp_tl": {
      "arch": "mlp", "regime": "tl", "params": 402826,
      "test_acc_own_regime": 0.98, "test_acc_all_positions": 0.21,
      "layers": [
        {"name": "fc1", "type": "linear", "in": 3136, "out": 128},
        {"type": "relu"},
        {"name": "fc2", "type": "linear", "in": 128, "out": 10}
      ]
    },
    "cnn_gap_tl": {
      "arch": "cnn_gap", "regime": "tl", "params": 0,
      "layers": [
        {"name": "conv1", "type": "conv2d", "in": 1, "out": 8, "k": 5, "pad": 2},
        {"type": "relu"}, {"type": "maxpool2d", "k": 2},
        {"name": "conv2", "type": "conv2d", "in": 8, "out": 16, "k": 5, "pad": 2},
        {"type": "relu"}, {"type": "maxpool2d", "k": 2},
        {"name": "conv3", "type": "conv2d", "in": 16, "out": 32, "k": 3, "pad": 1},
        {"type": "relu"}, {"type": "gap"}, {"type": "flatten"},
        {"name": "fc1", "type": "linear", "in": 32, "out": 10}
      ]
    }
  },
  "notes": []
}
```

(Values above are placeholders; fill in the real ones. `layers` must list every op in forward
order — the browser executes this list literally.)

### `weights/<id>.js`

```json
{"dtype": "float16", "tensors": {
  "conv1.weight": {"shape": [8, 1, 5, 5], "data": "<base64>"},
  "conv1.bias":   {"shape": [8],          "data": "<base64>"}
}}
```

- PyTorch layouts: conv weight `[out, in, kH, kW]`, linear weight `[out, in]`, row-major
  (C order), little-endian.
- `float16` (IEEE half) for all tensors. Check that the float16 model's test accuracy differs
  from float32 by < 0.2 pp; if not, export `float32` and set `"dtype": "float32"`.
- Size budget: total of all `weights/*.js` ≤ 6 MB.

### `samples.js`

50 test digits, 5 per class, chosen from the 2000-image subset, each correctly classified by
`cnn_gap_tl`, `cnn_fc_tl` **and** `mlp_tl` at offset `(0, 0)` (so the starting state is
"everything works").

```json
{"size": 28, "items": [{"label": 7, "test_index": 0, "pixels": "<base64 of 784 uint8>"}]}
```

### `position_maps.js`

```json
{"offsets": 29, "subset_size": 2000,
 "maps": {"mlp_tl": {"acc": [[...29 floats...], ...29 rows], "conf": [[...]]}, ...}}
```

Row index = `dy`, column index = `dx`. Round to 3 decimals.

### `reference.js` (for the browser self-check)

For each model and for 3 fixed inputs (sample 0 at offsets `(0,0)`, `(20,20)`, `(28,0)`):
the logits (float32, as plain numbers) and, for CNNs, the first 20 values (flattened) of the
outputs of `conv1` (after ReLU) and `conv3` (after ReLU). The demo recomputes these in
JavaScript and must match to `1e-3` (float16 weights are decoded to float32 on both sides —
**compute the references with the float16-rounded weights**).

### `training_log.json`

Per model: per-epoch metrics, final numbers, parameter count, the seed spread (seeds 0–2) of
`test_acc_own_regime` and `test_acc_all_positions`, the equivariance table, wall-clock time,
hardware.

## Acceptance checks (run them, report results in `export/README.md`)

1. `cnn_gap_tl` and `cnn_fc_tl`, `mlp_tl`: ≥ 97 % test accuracy in the `tl` regime.
2. Expected pattern on the position maps (if it does not show, **report it, do not tune it
   away**):
   - `mlp_tl`: high only near the top-left, near chance (~10–20 %) far away.
   - `cnn_fc_tl`: same qualitative failure as `mlp_tl`.
   - `cnn_gap_tl`: high over most of the map. Because training uses one exact position, a
     periodic pattern with period 4 px (the two stride-2 poolings) is likely and may be strong
     (offsets that are multiples of 4 good, others worse). This is a teaching point: report it,
     do not smooth it out, do not add blur-pooling or augmentation.
   - `*_all`: high everywhere; `mlp_all` lower than the CNNs.
3. A standalone NumPy re-implementation of the forward pass (reading the exported `.js`
   files, i.e. parsing the JSON after the `=`) reproduces `reference.js` to `1e-3`. Put it in
   `src/check_export.py`.
4. Sizes: `weights/` ≤ 6 MB total, everything in `export/` ≤ 8 MB.

## Out of scope

No anti-aliasing / blur-pooling variant, no hand-drawn input support, no web code, no figures for the lecture, no edits in the course repository. Deliver `export/`
(e.g. as a zip or a commit in the training repo) and the README.
