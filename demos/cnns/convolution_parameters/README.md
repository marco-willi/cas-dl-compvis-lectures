# Convolution Parameters Demo

Interactive demo for the CNN lecture: change kernel size, zero-padding, stride and dilation of
a convolution and observe how the output feature map changes. Hovering over (or tapping) an
output cell highlights the input cells it reads and shows the calculation.

Plain HTML, CSS and JavaScript without a backend. The only dependency is KaTeX for the
formulas, vendored in `../vendor/katex/` (shared with the receptive-field demo; without it the
formulas are shown as TeX source). All files are loaded with relative paths, so the demo works
on GitHub Pages under any sub-path and also when `index.html` is opened directly from disk
(`file://`).

Notation follows the lecture: input size $i$, output size $o$, kernel size $k$, padding $p$,
stride $s$, dilation $d$; output cell $(r, c)$, kernel entry $(u, v)$.

## Files

| Path                 | Purpose                                                                        |
| -------------------- | ------------------------------------------------------------------------------ |
| `index.html`         | Page and styles                                                                |
| `config.js`          | Input patterns, filter presets, offered parameter values, defaults             |
| `conv.js`            | Computation (cross-correlation, output size, sampled positions), no DOM access |
| `exercises.js`       | Exercises with automatic checks, no DOM access                                 |
| `app.js`             | Interface: rendering, highlighting, events, layout                             |
| `tests/conv_test.ts` | Checks for `conv.js`, `config.js` and `exercises.js`                           |

## Usage

- **Kernel size, Padding, Stride, Dilation**: the output is recomputed immediately. Padding,
  stride and dilation keep the kernel weights. A new kernel size needs a different number of
  weights: they are regenerated from the selected preset, and a notice below the kernel says so.
- **Kernel**: choose a preset or type into the weight fields (negative and decimal values are
  allowed). "Restore preset" discards the edits.
- **Output**: hover over a cell (mouse) or tap/drag (touch) to inspect it; a click or tap
  keeps the cell selected, the arrow keys move the selection. On the input, solid orange
  frames mark the cells that are read, the dashed frame marks the effective kernel extent.
- **Colors**: the input is shown in grayscale (0 = black, 1 = white), hatched cells are
  zero-padding. The output uses a diverging scale centered on zero with a fixed range of ±1,
  so that outputs of different settings can be compared; a note appears when values are
  clipped. "Rescale" fits the range to the current output, "Reset" restores everything.
- **Apply ReLU**: replaces negative output values by zero.
- **Exercises** (collapsed below the calculation): five tasks, e.g. "copy the input" or
  "detect top edges only". After every change the demo checks which tasks the current settings
  solve and ticks them; each task has a hint. Solutions are in the lecture (`cnns.qmd`).
- If the effective kernel is larger than the padded input, the output panel shows a message
  instead of a feature map.

Cells of input and output always have the same size; values are printed inside the cells when
the cells are at least 22 px wide (larger frames).

## Computation

One input channel, one filter, zero bias, cross-correlation as in CNN frameworks
(`torch.nn.Conv2d`):

```
o = floor((i + 2p - d*(k - 1) - 1) / s) + 1
output(r, c) = sum over (u, v) of kernel(u, v) * input(r*s - p + u*d, c*s - p + v*d)
```

Input positions outside the input count as zero.

## Embedding

Quarto page (`pages/lectures/*.qmd`):

````markdown
```{=html}
<iframe name="convolution-parameters" loading="lazy" data-auto-height
        src="../../demos/cnns/convolution_parameters/index.html" width="100%" height="720"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: kernel size, padding, stride and dilation of a convolution"></iframe>
```
````

With `data-auto-height` the iframe follows the height of the demo when page and demo are
served from the same site. Without it, or when the page is opened via `file://`, the given
`height` is used and the demo shrinks its cells to fit.

Reveal.js slide (`pages/slides/*.qmd`), fixed height:

````markdown
## Convolution Parameters

```{=html}
<iframe src="../../demos/cnns/convolution_parameters/index.html" width="100%" height="640"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: convolution parameters"></iframe>
```
````

The folder has to be listed under `project.resources` in `_quarto.yml`
(`demos/cnns/convolution_parameters/**`) so that Quarto copies it into the site.

## Configuration

Presets and defaults are in `config.js`, the exercises in `exercises.js` (`TASKS`: title,
text, hint and a `check(ctx)` function that receives the current parameters, input and output).
`config.js` contains:

- `INPUT_PRESETS`: named square matrices (default: the letter T, which is asymmetric, so that
  edge filters give different responses at its top, bottom and sides) with values in [0, 1]. All must have the same size;
  the size of the first one defines the input size.
- `FILTER_PRESETS`: named filters. `weights(k)` returns the `k × k` matrix for kernel size
  `k`, or `null` if the filter is not defined for that size. In that case the demo switches to
  `FALLBACK_FILTER` (the box average, which exists for every size) and says so.
- `KERNEL_SIZES`, `PADDINGS`, `STRIDES`, `DILATIONS`: the values offered by the controls.
- `DEFAULTS`: state after loading and after "Reset", including the color range.

The presets are scaled so that an input in [0, 1] gives an output in [-1, 1], which matches
the default color range.

## Tests

```bash
quarto run demos/cnns/convolution_parameters/tests/conv_test.ts
```

The checks cover a hand-calculated example (5 × 5 input, 3 × 3 kernel with negative and
decimal weights; padding, stride, dilation, ReLU), the sampled input positions, output sizes,
configurations without output, a comparison of every configuration offered by the demo with
an explicit pad-then-slide implementation, the presets (the box filter sums to 1 for
every kernel size), and the exercises (each intended solution passes, near misses and the
default state do not).
