# DINOv3 Patch Similarity

Interactive demo for the representation learning lecture (end of section "Global vs Local
Representations" in `pages/lectures/representation_learning.qmd`): click a point on any of the
8 example images and see the cosine similarity between the DINOv3 patch embedding at that point
and every patch of all 8 images, as in the DINOv3 figure above the demo, but on the lecture's
own examples.

Plain HTML, CSS and JavaScript without dependencies, backend or network requests. The features
are shipped as a script (`data/features.js`), so the demo also works when `index.html` is
opened directly from disk (`file://`).

## Files

| Path                       | Purpose                                         |
| -------------------------- | ----------------------------------------------- |
| `index.html`               | Page and styles                                 |
| `app.js`                   | Decoding of the features, similarity, interface |
| `data/features.js`         | Compressed patch features (generated)           |
| `data/collage.jpg`         | The 8 images as one 4 x 2 collage (generated)   |
| `tools/export_features.py` | Creates the two files in `data/`                |

## Usage

- **Query**: hover over an image to preview, click (or tap) to select a patch; arrow keys move
  the selected patch within its image.
- **Color scale**: "0 to 1" maps cosine similarity 0–1 to the color map (negative values are
  shown as 0); "Min to max" stretches it to the range of the current query, which helps for
  queries with low contrast (e.g. sky).
- **Overlay**: each image shows its highest similarity ("max") and a white ring at that patch;
  the query image shows "query" instead.
- **Reset**: back to the initial view (ear of the first elephant).

## Data

`tools/export_features.py` encodes the images with `vit_base_patch16_dinov3.lvd1689m` (timm,
same model as `demos/representation_learning/representation_comparisons.ipynb`) at 640 x 448 px,
i.e. 40 x 28 patches per image, and takes the l2-normalized patch tokens of the last block
(without [CLS] and register tokens). To keep the download small (1.5 MB instead of about 14 MB
in float16), the 768-dimensional features are projected onto their first 128 uncentered
principal directions and stored as int8 per row; the browser normalizes each row again. The
cosine similarities then differ from the full features by 0.009 on average (max 0.12; both are
stored in `PATCH_META`). Runs on CPU in about 15 seconds; no GPU needed.

```bash
python demos/representation_learning/patch_similarity/tools/export_features.py
```

## Embedding

````markdown
```{=html}
<iframe name="patch-similarity" loading="lazy" data-auto-height
        src="../../demos/representation_learning/patch_similarity/index.html" width="100%" height="500"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: DINOv3 patch similarity"></iframe>
```
````

With `data-auto-height` the iframe follows the height of the demo when page and demo are
served from the same site; when the page is opened via `file://`, the given `height` is used.
The folder is listed under `project.resources` in `_quarto.yml` (without `tools/`).
