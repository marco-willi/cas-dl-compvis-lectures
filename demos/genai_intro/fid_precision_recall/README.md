# FID, Precision and Recall

Interactive demo for the lecture "Generative AI for Computer Vision": a generator produces 2D
stand-ins for image features, which are compared with fixed real features by the Fréchet
Inception Distance (FID) and by precision and recall for generative models. Students shift the
generator, change its spread, switch clusters on and off, and see which metric reacts to which
failure.

Plain HTML, CSS and JavaScript without dependencies, backend or network requests; nothing is
stored. All files are loaded with relative paths, so the demo works on GitHub Pages under any
sub-path and also when `index.html` is opened directly from disk (`file://`).

## Files

| Path                | Purpose                                                      |
| ------------------- | ------------------------------------------------------------ |
| `index.html`        | Page and styles                                              |
| `fid.js`            | Sampling, FID, precision and recall, ellipses; no DOM access |
| `app.js`            | Interface: controls, plot, statistics                        |
| `tests/fid_test.ts` | Checks for `fid.js`                                          |

## Model

- **Real features** (`Fid.REAL`, frozen): three isotropic Gaussian clusters with centers
  (−1.6, −0.9), (1.6, −0.9), (0, 1.5) and standard deviation 0.55, the same as in the lecture
  figure `assets/images/genai/intro_figures/fid_pipeline.svg`. The real set consists of the first
  `n` samples of a fixed seed (2026).
- **Generator**: `x = center_k + shift + spread · 0.55 · e`, where the cluster `k` is chosen
  uniformly among the active clusters and `e` is 2D standard normal. The noise is drawn once per
  seed and sample count, so the points move smoothly while the sliders change.
- Real and generated sets always have the same number of points (50 to 1,000).

## Measures

```
FID       = |mu_r - mu_g|^2  +  tr(S_r + S_g - 2 (S_r S_g)^(1/2))
            (mean term)         (covariance term)
precision = share of generated points inside at least one ball around a real point
recall    = share of real points inside at least one ball around a generated point
```

Each ball reaches to the 3rd nearest neighbour of its point within the same set
(Kynkäänniemi et al., 2019). For 2x2 matrices, `tr((S_r S_g)^(1/2)) = sqrt(tr(S_r S_g) + 2 sqrt(det(S_r S_g)))`.

Values of the presets (300 points per set, seed 1):

| Preset                             | FID (mean + cov.)  | Precision | Recall |
| ---------------------------------- | ------------------ | --------- | ------ |
| Good model                         | 0.00               | 94%       | 95%    |
| Shifted (1.5, 1.1)                 | 3.52 (3.52 + 0.00) | 43%       | 61%    |
| Mode collapse (right cluster only) | 4.40 (3.07 + 1.33) | 97%       | 38%    |
| Too broad (spread 2)               | 0.18 (0.01 + 0.17) | 78%       | 99%    |
| Too narrow (spread 0.4)            | 0.02 (0.00 + 0.02) | 99%       | 56%    |

"Too narrow" shows a limitation of FID: a single Gaussian fitted to three clusters is dominated
by the distances between the clusters and hardly changes when the clusters become narrower.

## Embedding

````markdown
```{=html}
<iframe name="fid-precision-recall" loading="lazy" data-auto-height
        src="../../demos/genai_intro/fid_precision_recall/index.html" width="100%" height="900"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: FID, precision and recall"></iframe>
```
````

With `data-auto-height` the iframe follows the height of the demo when page and demo are
served from the same site; when the page is opened via `file://`, the given `height` is used.
The folder is listed under `project.resources` in `_quarto.yml` (without `tests/`).

## Tests

```bash
quarto run demos/genai_intro/fid_precision_recall/tests/fid_test.ts
```
