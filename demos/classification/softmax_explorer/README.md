# Softmax Explorer Demo

Interactive demo for the classification lecture: drag the logits of five classes (camera-trap
example: empty, deer, fox, badger, bird) and observe the probabilities after the softmax.

Plain HTML, CSS and JavaScript without dependencies or a backend; works on GitHub Pages and
from `file://`.

## Files

| Path                    | Purpose                                           |
| ----------------------- | ------------------------------------------------- |
| `index.html`            | Page and styles                                   |
| `config.js`             | Class names, logit range, presets, default preset |
| `softmax.js`            | Softmax, entropy, argmax, rounding; no DOM access |
| `app.js`                | Interface: bars, dragging, inputs, presets        |
| `tests/softmax_test.ts` | Checks for `softmax.js` and `config.js`           |

## Usage

- **Logits**: press on a column of the logit chart and drag up or down; the logit follows the
  pointer (rounded to 0.1, limited to −10 … 10). Alternatively type a value into the field
  below the column.
- **Preset**: the dropdown always shows the current state. After an edit it switches to
  "Custom (edited)".
- **Add 1 to all**: probabilities do not change (softmax is invariant to adding a constant),
  as long as no logit hits the axis limit of 10.
- **All × 2 / All ÷ 2**: scaling the logits makes the distribution sharper or flatter
  (the effect of a softmax temperature).
- The class with the largest probability is highlighted; the facts below the charts show its
  softmax calculation, the sum of the probabilities and the entropy.

## Embedding

````markdown
```{=html}
<iframe name="softmax-explorer" loading="lazy" data-auto-height
        src="../../demos/classification/softmax_explorer/index.html" width="100%" height="520"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: from logits to probabilities with the softmax"></iframe>
```
````

The folder is listed under `project.resources` in `_quarto.yml`.

## Tests

```bash
quarto run demos/classification/softmax_explorer/tests/softmax_test.ts
```
