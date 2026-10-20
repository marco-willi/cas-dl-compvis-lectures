# Generative Processes in 1D

Interactive demo for the lecture "GenAI Model Families": five generative model families produce
samples of the same bimodal 1-D target. Every method uses the same page template:

1. **Method head:** name, the question the method answers, and a badge (ideal model or trained network).
2. **One plot:** start (noise / prior / random draws) | mechanism | result (target on the right),
   with the value x on the vertical axis. Exceptions: the GAN plots G(z) with z horizontal, and the
   2-D autoregressive view plots x₁ horizontally.
3. **Main action row:** at most a few controls. The slider only ever means *generation progress*
   (diffusion, flow matching) or, in the GAN's separate training view, the
   training step. GAN, VAE and autoregressive use buttons.
4. **Show distribution:** by default one highlighted sample is followed among 20 faded ones; the
   toggle shows 120 samples, the histogram and the exact distribution at the slider.
5. **Observe / Explain / Try:** fixed texts per view (`content.js`).
6. **Advanced** (collapsed): method-specific options, each with a one-line caption.
7. **Diagnostics** (collapsed): W₁ of finished samples, error against an accurate reference for
   step-count experiments, and the sources of randomness.

`?full=1` opens Advanced and Diagnostics and offers all step counts and samplers (playground).
`?embed=1` shows one method without the title and the tabs, with a link that opens the whole demo
in a new tab; the lecture `genai_models.qmd` embeds one method per section this way.

| Tab            | Default view and main action                               | Advanced                                                        | Model                  |
| -------------- | ---------------------------------------------------------- | --------------------------------------------------------------- | ---------------------- |
| Autoregressive | 2D (default), Sequence: “Next value”; 1D: “Next token”     | temperature, random-number gauges, Sample 100 (2D)              | ideal (exact formulas) |
| GAN            | final G(z); drag z or “Generate another sample”            | Training dynamics view (snapshots, discriminator, unstable run) | trained network        |
| VAE            | one z, its decoder distribution; “Sample x again”, “New z” | decoder noise off, inference (encoder) view                     | trained network        |
| Diffusion      | reverse process (DDPM, 100 steps); Play + progress         | DDIM, steps, re-run sampler, forward noising, heat map          | ideal (exact score)    |
| Flow matching  | Training (lines + velocity probe) / Sampling switch; Play  | Euler steps, resample pairs, velocity-field arrows              | ideal (exact velocity) |

**Compare final distributions** (link below the tabs, `?view=compare`): three panels side by side,
flow matching with an accurate solver (RK4, 200 steps), flow matching with 4 Euler steps and the trained GAN: three sources of
error (none, numerical, learning).

Plain HTML, CSS and JavaScript without dependencies, backend or network requests. Works from
`file://` and on GitHub Pages.

## Files

| Path                    | Purpose                                                                   |
| ----------------------- | ------------------------------------------------------------------------- |
| `index.html`            | Page and styles                                                           |
| `content.js`            | Question and Observe / Explain / Try texts per method and view            |
| `toy.js`                | Target, exact models, samplers, trained-model lookup, comparisons; no DOM |
| `app.js`                | Interface: template, plots, controls, tracing, diagnostics                |
| `data/models.js`        | Trained GAN and VAE as lookup tables (generated, ~60 KB)                  |
| `tools/train_models.py` | Trains GAN and VAE, writes `data/models.js` and `tests/fixtures.json`     |
| `tests/toy_test.ts`     | Checks for `toy.js` and the trained models                                |
| `tests/fixtures.json`   | Reference values of the ideal models, computed independently in Python    |

**Autoregressive sequence:** eight values x₁ … x₈ (like a row of pixels), drawn one after the other.
At each position the plot shows the distribution of the next value given all values so far (blue)
and, dashed, the distribution of that position without context. The data come from two patterns
with the target's weights: all sequences start near 4.5 and move towards 2 or 7; each pattern is a
Gaussian with correlation 0.7^|k−l| between positions, so every conditional is a two-component
Gaussian mixture in closed form, and the last value follows the target exactly. The model's
certainty about the direction grows over several values (on average 64 %, 77 %, 87 %, 94 % after
2–5 values).

## Models

**Target:** 0.4 · N(2, 0.8²) + 0.6 · N(7, 1.0²), the same as in
`demos/genai_intro/distribution_fitting` (checked by the tests). Noise: N(0, 1).

**Ideal models** know the target and are computed in closed form, so every error they show comes
from the sampling procedure (too few steps), never from a trained network:

- **Autoregressive:** a value written as two tokens: coarse c (unit interval in \[−1, 11)) and fine
  f (tenth). Exact tables p(c), p(f | c) from the target cdf; temperature τ applied to every
  categorical. Generated values are bin centres (resolution 0.1). The 2-D companion uses
  x₁ ~ target and x₂ | x₁ ~ N(1 + 0.25 (x₁ − 4.5)², (0.5 + 0.06 x₁)²).
- **Normalizing flow** (no tab; used by the tests only): layers f_k = F_k⁻¹ ∘ F\_{k−1} between intermediate mixtures p_k. The progress
  λ = k/K is split into thirds: separate the modes (centred at 0), move them to the data, sharpen
  them; with K = 3 every layer has one job. Exactly invertible, exact log-derivatives.
- **Flow matching:** linear paths x_t = (1 − t)·x₀ + t·x₁ (t = 0 noise); the marginal velocity
  u(x, t) = E[x₁ − x₀ | x_t = x] is closed form. The probe averages the directions of 40,000 random
  training pairs passing near a chosen (x, t) and compares the mean with u(x, t).
- **Diffusion:** one VP schedule for all samplers: β(t) linear from 0.1 to 20 on t ∈ [0, 1]
  (continuous version of DDPM's 1e-4 … 0.02 over 1000 steps). DDPM (DDIM update with η = 1),
  DDIM (η = 0), probability-flow ODE (Heun). Samplers start from the exact p_T, which is N(0, 1)
  shifted by √ᾱ_T · 5 ≈ 0.03; otherwise the deterministic samplers are visibly off between the
  modes. Initial noise and sampler noise use separate random streams ("re-run sampler" keeps x_T).

In 1-D, the normalizing flow, the flow-matching ODE and the diffusion ODE realize the same unique
order-preserving map from N(0, 1) to the target: same z, same x. The tests check this.

**Trained networks** (`tools/train_models.py`, PyTorch, CPU, about a minute, fixed seeds): trained on
1,000 samples of the target (W₁ of these samples themselves: 0.12), in standardized coordinates.

- **GAN:** MLPs (2 × 64, LeakyReLU), non-saturating loss, Adam (β₁ = 0.5). Two runs: *standard*
  (lr 1e-3, batch 256) and *unstable* (lr 1e-2, batch 32). 13 snapshots between step 0 and 6000;
  for each, G on a z-grid and D on an x-grid. The standard run finds both modes with the right
  weights but squeezes each into narrow spikes (W₁ ≈ 0.37); the unstable run collapses onto the
  right mode (step 25–50), onto a single value (step 75–100) and then oscillates. Nothing is forced.
- **VAE:** 1-D latent, Gaussian encoder and Gaussian decoder with learned σθ(z) (≥ 0.03), β = 1.
  Decoder mean and σ on a z-grid, encoder on an x-grid. The decoder mean alone already places the
  samples in both modes (W₁ 0.13 vs. 0.12 with decoder noise).

The page interpolates the lookup tables linearly; the tests check that sampling them in JS gives the
W₁ measured in Python.

## Embedding

````markdown
```{=html}
<iframe name="generative-paths" loading="lazy" data-auto-height
        src="../../demos/genai_models/generative_paths/index.html?family=fm" width="100%" height="1000"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: generative processes in 1D"></iframe>
```
````

URL parameters: `family` = `ar` | `gan` | `vae` | `diffusion` | `fm`; `view=compare` for the
comparison; `embed=1` for lecture sections; `full=1` for the playground; optional `mode` = `2d` |
`seq` | `tokens` (ar), `generate` | `infer` (vae), `reverse` | `forward` (diffusion), `flow` | `pairs`
(fm); `run` = `standard` | `unstable` (gan). `family=ar2d` is short for `family=ar&mode=2d`. The
folder is listed under `project.resources` in `_quarto.yml` (without `tests/`, `tools/` and the
README).

## Tests and regeneration

```bash
quarto run demos/genai_models/generative_paths/tests/toy_test.ts
uv run python demos/genai_models/generative_paths/tools/train_models.py   # only after changing the models
```
