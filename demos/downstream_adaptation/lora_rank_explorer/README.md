# LoRA Rank Explorer

Interactive demo for the lecture on adapting models: choose the rank `r` of a LoRA update
`W' = W + (α / r) · B A` for a 64 x 64 weight matrix `W` and compare the number of parameters
of the full matrix with those of the factors `B` and `A`.

Plain HTML, CSS and JavaScript without dependencies, backend or network requests; nothing is
stored. All files are loaded with relative paths, so the demo works on GitHub Pages under any
sub-path and also when `index.html` is opened directly from disk (`file://`).

## Files

| Path                 | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| `index.html`         | Page and styles                                |
| `lora.js`            | Parameter counts, no DOM access                |
| `app.js`             | Interface: controls, diagram, bars, statistics |
| `tests/lora_test.ts` | Checks for `lora.js`                           |

## Usage

- **Shape of W**: fixed to `d_in = d_out = 64` (`D_IN`, `D_OUT` in `app.js`), so that every
  rank fits on the slider and the break-even rank 32 is easy to reach.
- **Rank r**: slider (arrow keys, Home, End) or the number field next to it, from 1 to
  `min(d_in, d_out)` = 64.
- **α**: only displayed as the scale `α / r`; it adds no parameters.
- **Target matrices**: multiplies both counts, e.g. 24 for the query and value matrices in
  12 transformer blocks. The classifier head is never included.
- The diagram draws `W`, `B` and `A` to scale relative to the larger dimension of `W`, with a
  minimum thickness of 3 px. Frozen weights are blue and solid, trainable factors orange and
  hatched, and both are labeled in text.

## Computation

```
full       = d_out * d_in * matrices
LoRA       = r * (d_in + d_out) * matrices
ratio      = LoRA / full            (exceeds 100% for r > d_in * d_out / (d_in + d_out))
difference = full - LoRA
```

## Embedding

````markdown
```{=html}
<iframe name="lora-rank-explorer" loading="lazy" data-auto-height
        src="../../demos/downstream_adaptation/lora_rank_explorer/index.html" width="100%" height="760"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: LoRA rank explorer"></iframe>
```
````

With `data-auto-height` the iframe follows the height of the demo when page and demo are
served from the same site; when the page is opened via `file://`, the given `height` is used.
The folder is listed under `project.resources` in `_quarto.yml` (without `tests/`).

## Tests

```bash
quarto run demos/downstream_adaptation/lora_rank_explorer/tests/lora_test.ts
```
