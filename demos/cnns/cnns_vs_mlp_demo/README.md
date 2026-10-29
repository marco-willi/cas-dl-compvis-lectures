# MLP vs CNN on Shifted Digits

Interactive demo for the CNN lecture (recap section "Recap: Translation in MLPs and CNNs" at
the end of `pages/lectures/cnns.qmd`): an MLP, a CNN with a fully connected head (CNN + FC) and a CNN with
the same convolutional layers followed by global average pooling (CNN + GAP), trained on MNIST
digits placed on a 56 × 56 canvas. Students move a digit and compare the predictions, the
position-accuracy maps and, under "Look inside the CNNs", the activation maps.

Plain HTML, CSS and JavaScript without a backend. The networks run in `nn.js` (no ONNX
Runtime), and all data is loaded with `<script>` tags, so the demo works from `file://` and on
GitHub Pages.

## Files

| Path                     | Purpose                                                                           |
| ------------------------ | --------------------------------------------------------------------------------- |
| `index.html`             | Page and styles                                                                   |
| `app.js`                 | Interface: input canvas, model cards, look inside, self-check, events             |
| `nn.js`                  | Weight decoding (float16) and forward pass over the manifest's layer list         |
| `model/manifest.js`      | Architectures (layer lists), regimes, parameter counts, headline accuracies       |
| `model/weights/*.js`     | Weights of the 6 networks (float16, base64), 4.6 MB in total                      |
| `model/samples.js`       | 50 MNIST test digits (5 per class)                                                |
| `model/position_maps.js` | Test accuracy for each of the 29 × 29 positions (2,000 test images)               |
| `model/reference.js`     | PyTorch outputs for the self-check                                                |
| `tools/`                 | Training and export code and its report (`EXPORT_README.md`, `training_log.json`) |
| `tests/nn_test.ts`       | Checks for `nn.js` against `model/reference.js`                                   |
| `archive/`               | Old exercise notebooks the experiment is based on (not published)                 |

## Networks

| Network | Architecture                                                          | Parameters |
| ------- | --------------------------------------------------------------------- | ---------: |
| MLP     | Flatten → Linear(3136, 128) → ReLU → Linear(128, 10)                  |    402,826 |
| CNN+FC  | trunk → Flatten(64·14·14) → Linear(12544, 32) → ReLU → Linear(32, 10) |    433,514 |
| CNN+GAP | trunk → global average pooling → Linear(64, 10)                       |     32,394 |

trunk: `Conv(1→16, 5×5) → ReLU → MaxPool 2 → Conv(16→32, 5×5) → ReLU → MaxPool 2 → Conv(32→64, 3×3) → ReLU`, "same" padding, so all maps keep the input's coordinate frame.

Each network is trained twice: "only top-left" (digit exactly at offset (0, 0)) and "anywhere"
(offset uniform in 0–28 along both axes). The "anywhere" weights are only loaded when that
setting is selected.

## Usage

The page has six collapsible sections: "Training data" (5 examples per type of training data,
open), "Architectures" (layer tables, closed), "Move the digit" (open), "Look inside the CNNs",
"Questions" and "Verification against Python" (closed). The layer tables are static HTML: update
them when the networks change. "Questions" repeats the exercise of the lecture, each answer collapsed; keep
both in sync.

- **Move the digit**: drag it, click elsewhere on the canvas, use the arrow keys (Shift: 4 px), or
  click on an accuracy map.
- **URL parameters**: `?regime=all` starts with the "anywhere" networks, `?inside` opens "Look
  inside the CNNs", `?selftest` runs the self-check on load.
- **Look inside**: activation maps of conv1 and conv3 (channel selectable; the default is the
  channel with the largest total activation for the current digit at (0, 0)); for CNN + GAP the 64
  GAP outputs now vs. at (0, 0), for CNN + FC the sum of |fc1 weights| per position of the
  conv3 map. "Change" is ‖a − a₀‖ / ‖a₀‖ of the classifier input, relative to offset (0, 0).

## Checks

```bash
quarto run demos/cnns/cnns_vs_mlp_demo/tests/nn_test.ts
```

In the browser: open the demo with `?selftest` or press "Run self-check" under "Verification
against Python".

## Regenerate the networks

The networks were trained outside this repository with the scripts in `tools/src/` (spec:
`tools/src/SPEC.md`; results and deviations from the spec: `tools/EXPORT_README.md`). Copy a
new export's `manifest.js`, `samples.js`, `position_maps.js`, `reference.js` and `weights/` to
`model/` and run the checks.

## Embed in Quarto

````markdown
```{=html}
<iframe name="cnns-vs-mlp" loading="lazy" data-auto-height
        src="../../demos/cnns/cnns_vs_mlp_demo/index.html"
        width="100%" height="1080" style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: MLP vs CNN on shifted MNIST digits"></iframe>
```
````

On Reveal.js slides use `data-src` instead of `src` and leave out `data-auto-height`. The
folder is listed under `project.resources` in `_quarto.yml` (without `archive/` and `tools/`).
