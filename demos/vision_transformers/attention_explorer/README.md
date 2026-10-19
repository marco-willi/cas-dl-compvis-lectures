# ViT Attention Explorer

Interactive demo for the lecture on Vision Transformers: select a query patch on an image and
see how strongly it attends to every other patch, per layer and per attention head of a
pretrained ViT (`vit_little_patch16_reg4_gap_256.sbb_in1k`: 14 layers, 5 heads, 16 x 16
patches, 4 register tokens, no [CLS] token).

Plain HTML, CSS and JavaScript without dependencies, backend or network requests. The attention
weights are shipped as a script (`data/attention.js`), so the demo also works when `index.html`
is opened directly from disk (`file://`).

## Files

| Path                        | Purpose                                   |
| --------------------------- | ----------------------------------------- |
| `index.html`                | Page and styles                           |
| `app.js`                    | Decoding of the data and interface        |
| `data/attention.js`         | Precomputed attention weights (generated) |
| `data/image.jpg`            | Model input, 256 x 256 (generated)        |
| `tools/export_attention.py` | Creates the two files in `data/`          |

## Usage

- **Query**: hover over the image to preview, click (or tap) to select a patch; arrow keys move
  the selected patch. The "Register 1–4" buttons use a register token as query instead.
  In this model the 4 registers are identical (same learned values, hence the same attention):
  they have no positional embedding and timm initialized them with zeros until July 2024, so
  training never breaks the symmetry ([timm PR #2229](https://github.com/huggingface/pytorch-image-models/pull/2229)).
  The export script stores this as `registersIdentical`; the demo then shows a note.
- **Reset**: back to the initial view (layer 8, mean over heads, patch on the right cow).
- **Statistics**: attention on the query patch itself, total attention on the register tokens
  (not shown in the overlay), and the mean attention distance in pixels (distance between the
  query patch and every patch, weighted by attention; as in Fig. 7 of Dosovitskiy et al., 2020).

## Data

`tools/export_attention.py` recomputes the softmax attention of every block with forward hooks
(same model and image as `demos/vision_transformers/create_vit_illustrations.ipynb`). Each row
(one query token) is stored as uint8 relative to its maximum plus a float32 scale, then
gzip-compressed and base64-encoded (about 2.8 MB). The browser decompresses it with
`DecompressionStream`.

```bash
python demos/vision_transformers/attention_explorer/tools/export_attention.py
```

## Embedding

````markdown
```{=html}
<iframe name="vit-attention-explorer" loading="lazy" data-auto-height
        src="../../demos/vision_transformers/attention_explorer/index.html" width="100%" height="940"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: ViT attention explorer"></iframe>
```
````

With `data-auto-height` the iframe follows the height of the demo when page and demo are
served from the same site; when the page is opened via `file://`, the given `height` is used.
The folder is listed under `project.resources` in `_quarto.yml` (without `tools/`).
