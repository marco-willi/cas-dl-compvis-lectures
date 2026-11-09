# How Generative Models Sample an Image

Illustrative demo for the slides of "GenAI Model Families": five model families produce an image
of an elephant. Like Figure 1 of the lecture (`sampling_mechanisms.svg`), the steps are computed
from fixed images (revealing patches, adding noise, interpolating with noise); they are **not**
produced by the models shown.

**Images:** image 1 is the canonical example of the lecture (`assets/images/genai/elephant.jpg`);
images 2 to 10 were generated with FLUX 1.1 Pro via the Replicate API (`tools/generate_images.py`,
prompts and seeds in `assets/images/genai_models/image_sampling/prompts.json`).

Every family uses the same layout: inputs | network | output (large), a strip below, one action
row (mode, progress slider, Play, New sample, Reset; autoregressive also Back / Next token) and a
counter of network evaluations. Reset returns to the first image (the canonical example) and step 0.

| `?family=`  | Stage                                                         | Strip                   | Slider            |
| ----------- | ------------------------------------------------------------- | ----------------------- | ----------------- |
| `ar`        | tokens so far → network → distribution over 24 tokens         | –                       | token 0 … 144     |
| `gan`       | z (2-D, prior N(0, I)) → generator → image                    | previous samples        | –                 |
| `vae`       | Generate: z → decoder; Reconstruct: x → encoder → z → decoder | previous samples        | –                 |
| `diffusion` | Reverse: noise → denoiser → x_t; Forward: data → + noise      | x_t at 6 noise levels   | step 0 … 50       |
| `fm`        | noise → velocity → x_t                                        | x_t at t = 0, 0.2, …, 1 | Euler step 0 … 50 |

`?paths=1` turns on *Compare paths* (diffusion, flow matching; off by default).
`?mode=` selects `generate` / `reconstruct` (VAE) or `reverse` / `forward` (diffusion).
`?embed=1` hides the title and the tabs (for slides) and shows a link that opens the demo in a new tab.
An embedding iframe with `data-auto-height` follows the height of the demo.

Plain HTML, CSS and JavaScript without dependencies, backend or network requests. The images are
embedded as data URIs, so their pixels can be read from `file://` as well.

## Files

| Path                       | Purpose                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| `index.html`               | Page and styles                                                    |
| `config.js`                | Texts per family and mode, play durations                          |
| `sampling.js`              | Noise, schedules, paths, token distributions, latents; no DOM      |
| `app.js`                   | Interface: stage, strip, controls, playback                        |
| `data/images.js`           | Ten images (256x144, base64 JPEG) and a token codebook (generated) |
| `tools/build_data.py`      | Writes `data/images.js` from the course images                     |
| `tools/generate_images.py` | Generates images 2 to 10 with the Replicate API (FLUX 1.1 Pro)     |
| `tests/sampling_test.ts`   | Checks for `sampling.js`                                           |

## What is computed

- **Autoregressive:** 16x9 tokens of 16x16 pixels in raster order. The bar chart is an illustrative
  `p(x_i | x_<i)`: a softmax over the negative distances of the image's patch to 24 codebook
  entries (k-means of all patches, temperature 0.04). Filled tokens show the image's own patch.
- **GAN:** `z ~ N(0, I)` in 2-D, plotted with earlier samples (numbered as in the strip); each new
  z shows the next image.
- **VAE:** illustrative encoder output (μ on a circle, small σ), `z = μ + σ ⊙ ε`; the output is the
  image.
- **Diffusion:** cosine schedule, 50 steps, `x_t = sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε_t`. The noise ε_t
  is partly renewed in every step (correlation 0.9): in the reverse process like the fresh noise of
  DDPM, in the forward process like a Markov chain.
- **Flow matching:** the straight line `x_t = (1 − t) ε + t x_1`; the same noise always gives the
  same path.

**How diffusion and flow matching differ** (optional, checkbox *Compare paths*): next to the output, a plot traces the value (green
channel) of three marked pixels over the steps: jagged for diffusion, straight for flow matching.
*Run again (same noise)* keeps the starting noise and shows the previous run dashed: diffusion takes
a different path (new fresh noise), flow matching repeats it exactly.

"New sample" draws new noise and moves on to the next image.

## Tests and data

```bash
quarto run demos/genai_models/image_sampling/tests/sampling_test.ts
.venv/bin/python demos/genai_models/image_sampling/tools/generate_images.py  # needs REPLICATE_API_KEY
.venv/bin/python demos/genai_models/image_sampling/tools/build_data.py
```
