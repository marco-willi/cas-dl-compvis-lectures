# Shifted MNIST: MLP vs. CNN+FC vs. CNN+GAP — export

Deliverables for the browser demo `demos/cnns/cnns_vs_mlp_demo/` of the course repo
(`cas-dl-compvis-lectures-hs2026`), produced as specified in `SPEC.md` (next to this
experiment's code, copied into `src/`).

## Provenance

- Repository: `dl-cv-exercises`, `dev@0c2897e570282530c58f5d61f289552ef21e77c3` (the experiment folder is gitignored and
  not committed; `src/` holds the exact scripts that produced this export)
- Python 3.12.3, torch 2.14.1+cu130, torchvision 0.29.1+cu130, numpy 2.5.4
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU; total training wall-clock 68.8 min (18 runs; the seed-0 runs of
  `cnn_fc_all` and `cnn_gap_tl` shared the GPU with other jobs and took 3-5x longer than the
  ~220 s of the other CNN runs)
- Weights dtype: `float16`

```bash
cd experiments/cnn_on_mnist
../../.venv/bin/python train.py --all        # 3 archs x 2 regimes x seeds 0-2
../../.venv/bin/python evaluate.py           # position maps, fp16 check, equivariance
../../.venv/bin/python export.py             # writes export/
../../.venv/bin/python check_export.py       # acceptance checks 3 + 4
# or: bash run_all.sh
```

## Deviations from the spec

- Produced in the dl-cv-exercises repo (experiments/cnn_on_mnist/, flat scripts instead of a separate repo with a src/mnist_shift package); the code is in export/src/.
- The mlp layer list starts with an explicit {type: flatten} op (canvas -> 3136 vector).
- CNN trunk widened from [8, 16, 32] to [16, 32, 64] channels for both cnn_fc and cnn_gap: with 8/16/32, cnn_gap_tl reached only 91.6 % val accuracy.
- Training changed for all six models: AdamW lr 0.003 (spec 0.001), tl epochs 20 (spec 10); all epochs 20 as in the spec. With the widened trunk, lr 1e-3 and 10 epochs, cnn_gap_tl was still underfitting (94.5 % val).
- Position maps use the 2000-image stratified test subset.
- test_acc_own_regime: full 10k test set, one fixed offset per image drawn from the model's regime; test_acc_all_positions: mean of the position map.
- Weights exported as float16 (max |fp16 - fp32| accuracy difference 0.01 pp); all reported numbers use the exported weights.

## Results (seed 0 = exported, exported weights)

| model         |  params | best epoch | test acc own regime (%) | test acc all positions (%) | own, seeds 0-2 (%) | all pos., seeds 0-2 (%) |
| ------------- | ------: | ---------: | ----------------------: | -------------------------: | -----------------: | ----------------------: |
| `mlp_tl`      | 402,826 |         13 |                   97.66 |                      11.01 |       97.70 ± 0.04 |            11.15 ± 0.15 |
| `cnn_fc_tl`   | 433,514 |         12 |                   99.07 |                       9.34 |       99.10 ± 0.03 |            10.55 ± 1.16 |
| `cnn_gap_tl`  |  32,394 |         19 |                   98.71 |                      98.58 |       98.48 ± 0.20 |            97.76 ± 0.75 |
| `mlp_all`     | 402,826 |         17 |                   90.25 |                      89.87 |       90.48 ± 0.33 |            89.97 ± 0.12 |
| `cnn_fc_all`  | 433,514 |         19 |                   98.75 |                      98.63 |       98.90 ± 0.14 |            98.86 ± 0.20 |
| `cnn_gap_all` |  32,394 |         18 |                   98.11 |                      98.05 |       98.12 ± 0.02 |            98.10 ± 0.06 |

## Equivariance of `cnn_gap_tl` (100 test digits, shift right, after ReLU)

Relative error ||f(shift_t x) - shift_s f(x)|| / ||shift_s f(x)||, border of s px cropped.
Rows with "no (nearest)" are outside the spec (t not a multiple of 4 for conv3) and compare
with the nearest integer feature shift.

| layer | t (px) | feature shift s | exact        |   mean |    max |
| ----- | -----: | --------------: | ------------ | -----: | -----: |
| conv1 |      1 |               1 | yes          | 0.0000 | 0.0000 |
| conv1 |      2 |               2 | yes          | 0.0000 | 0.0000 |
| conv1 |      3 |               3 | yes          | 0.0000 | 0.0000 |
| conv1 |      4 |               4 | yes          | 0.0000 | 0.0000 |
| conv1 |      8 |               8 | yes          | 0.0000 | 0.0000 |
| conv3 |      1 |               0 | no (nearest) | 0.2229 | 0.2978 |
| conv3 |      2 |               0 | no (nearest) | 0.4103 | 0.5062 |
| conv3 |      3 |               1 | no (nearest) | 0.2261 | 0.2919 |
| conv3 |      4 |               1 | yes          | 0.0175 | 0.0420 |
| conv3 |      8 |               2 | yes          | 0.0184 | 0.0486 |

conv1 is exactly equivariant (error at float precision). For conv3 the exact shifts
(t = 4, 8) are not exactly 0 either: the background after conv1 is relu(bias), not 0, while the
zero padding of conv2/conv3 is 0, so border effects reach further in than the s-px crop. The
non-multiples of 4 are clearly worse (pooling breaks exact equivariance).

## Acceptance checks

### 1. ≥ 97 % test accuracy in the tl regime

- `mlp_tl`: 97.66 % → PASS
- `cnn_fc_tl`: 99.07 % → PASS
- `cnn_gap_tl`: 98.71 % → PASS

### 2. Expected patterns on the position maps (reported, not tuned)

Criteria: *fails away* = (0,0) ≥ 0.95 and far BR ≤ 0.25; *high everywhere* = mean ≥ 0.85 and min ≥ 0.75.

| model         | expected                                           | (0,0) | near TL (dx,dy ≤ 3) | far BR (dx,dy ≥ 21) |  mean |   min | dx,dy mult. of 4 | other | matches |
| ------------- | -------------------------------------------------- | ----: | ------------------: | ------------------: | ----: | ----: | ---------------: | ----: | ------- |
| `mlp_tl`      | high near TL, ~chance far away                     | 0.973 |               0.649 |               0.099 | 0.110 | 0.018 |            0.119 | 0.109 | yes     |
| `cnn_fc_tl`   | same failure as mlp_tl                             | 0.991 |               0.861 |               0.101 | 0.093 | 0.007 |            0.105 | 0.092 | yes     |
| `cnn_gap_tl`  | high over most of the map, period-4 pattern likely | 0.985 |               0.986 |               0.985 | 0.986 | 0.976 |            0.986 | 0.986 | yes     |
| `mlp_all`     | high everywhere, lower than CNNs                   | 0.840 |               0.898 |               0.903 | 0.899 | 0.783 |            0.889 | 0.899 | yes     |
| `cnn_fc_all`  | high everywhere                                    | 0.974 |               0.981 |               0.985 | 0.986 | 0.974 |            0.985 | 0.986 | yes     |
| `cnn_gap_all` | high everywhere                                    | 0.981 |               0.980 |               0.980 | 0.981 | 0.978 |            0.982 | 0.980 | yes     |

Observations:

- Period-4 pattern of `cnn_gap_tl`: **not visible** (offsets with dx, dy multiples of 4: 0.986, others: 0.986; min over the map 0.976). Reported as is, nothing was changed to provoke it.
- `mlp_tl` and `cnn_fc_tl` already break down within a few pixels: (0,0) 0.973 / 0.991, mean over dx, dy ≤ 3 0.649 / 0.861.
- `mlp_all` is weakest near the canvas corners: (0,0) 0.840, min 0.783, mean 0.899.

<!-- CHECK_EXPORT:BEGIN -->

### 3. NumPy forward pass reproduces `reference.js` (tolerance 1e-3)

- `mlp_tl`: max |numpy - reference| = 3.81e-06, params 402,826 → PASS; samples correct at (0,0): 50/50
- `cnn_fc_tl`: max |numpy - reference| = 2.38e-06, params 433,514 → PASS; samples correct at (0,0): 50/50
- `cnn_gap_tl`: max |numpy - reference| = 7.63e-06, params 32,394 → PASS; samples correct at (0,0): 50/50
- `mlp_all`: max |numpy - reference| = 1.91e-06, params 402,826 → PASS
- `cnn_fc_all`: max |numpy - reference| = 3.81e-06, params 433,514 → PASS
- `cnn_gap_all`: max |numpy - reference| = 3.62e-05, params 32,394 → PASS
- samples.js: 50 digits, 5 per class → PASS

### 4. Sizes

- `weights/`: 4.64 MB (≤ 6 MB) → PASS
- `export/` total: 4.87 MB (≤ 8 MB) → PASS

<!-- CHECK_EXPORT:END -->
