# Receptive Field Demo

Interactive demo for the CNN lecture (section "Receptive Field"): choose the number of layers
(3 or 4) and the kernel size and stride of each layer, then select an activation in the last
layer. The demo marks, in every activation map down to the input, the activations the selected
one depends on, and shows the closed-form receptive field $r_0$ with the current values.

Plain HTML, CSS and JavaScript without a backend. KaTeX for the formulas is vendored in
`../vendor/katex/` (shared with the convolution-parameter demo; without it the formulas are
shown as TeX source). Works on GitHub Pages and from `file://`.

## Files

| Path               | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `index.html`       | Page and styles                                                         |
| `config.js`        | Input size, offered layer counts, kernel sizes, strides, defaults       |
| `rf.js`            | Computation (map sizes, receptive fields, ranges, paths), no DOM access |
| `app.js`           | Interface: controls, canvas drawing, events, layout                     |
| `tests/rf_test.ts` | Checks for `rf.js` and `config.js`                                      |

## Usage

- **Layers**: 3 or 4 layers; each layer has its own kernel size and stride.
- **Activation maps**: drawn with the same side length, from the input (left) to the last layer
  (right; stacked vertically on narrow screens). A map with fewer activations therefore has
  larger cells, and the marked regions line up spatially across layers.
- **Selection**: hover over an activation of the last map (mouse) or tap/drag (touch); a click
  keeps it selected, the arrow keys move it.
- **Dashed frame**: the receptive field extends beyond the map into the zero-padding. The text
  below the table says how much of it lies inside the input.
- **Shade by number of paths**: instead of a uniform color, each activation is shaded by the
  number of paths through the network that connect it to the selected activation. Central
  activations contribute through many more paths than those at the border of the receptive field.

## Computation

One channel, no dilation, padding $p_l = \lfloor k_l / 2 \rfloor$ in every layer (keeps the
size for stride 1):

```
o_l = floor((o_{l-1} + 2 p_l - k_l) / s_l) + 1                (o_0 = i, the input size)
r_0 = sum_{l=1..L} (k_l - 1) * prod_{i=1..l-1} s_i + 1        (receptive field, see lecture)
range in map l-1 of the range [lo, hi] in map l: [lo*s_l - p_l, hi*s_l - p_l + k_l - 1]
```

In 2-D the receptive field is the product of the row range and the column range; the number
of paths to a cell (r, c) is the product of the 1-D path counts of row r and column c.

## Embedding

Quarto page (`pages/lectures/*.qmd`):

````markdown
```{=html}
<iframe name="receptive-field" loading="lazy" data-auto-height
        src="../../demos/cnns/receptive_field/index.html" width="100%" height="700"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: receptive field of stacked convolutions"></iframe>
```
````

With `data-auto-height` the iframe follows the height of the demo when page and demo are
served from the same site; otherwise the given `height` is used and the demo scrolls inside
the frame.

The folders `demos/cnns/receptive_field/**` and `demos/cnns/vendor/**` are listed under
`project.resources` in `_quarto.yml` so that Quarto copies them into the site.

## Tests

```bash
quarto run demos/cnns/receptive_field/tests/rf_test.ts
```

The checks cover hand-calculated examples (including the quiz in the lecture), and compare
`rf.js` for every network the demo offers (3 and 4 layers, all kernel sizes and strides,
three positions each) with a brute-force search that follows every connection backwards.
