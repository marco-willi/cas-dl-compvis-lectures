# Learning a 1D Distribution

Interactive demo for the lecture "Generative AI for Computer Vision": a generator (a single
Gaussian or a mixture of two Gaussians) tries to reproduce a fixed bimodal target distribution.
Students change the generator's parameters, the number of samples and the seed, fit the
generator by maximum likelihood, and compare fidelity, coverage and log-likelihood.

Plain HTML, CSS and JavaScript without dependencies, backend or network requests; nothing is
stored. All files are loaded with relative paths, so the demo works on GitHub Pages under any
sub-path and also when `index.html` is opened directly from disk (`file://`).

## Files

| Path                 | Purpose                                                  |
| -------------------- | -------------------------------------------------------- |
| `index.html`         | Page and styles                                          |
| `dist.js`            | Densities, sampling, fitting, comparisons; no DOM access |
| `app.js`             | Interface: controls, plot, statistics                    |
| `tests/dist_test.ts` | Checks for `dist.js`                                     |

## Model

- **Target** (`Dist.TARGET`, frozen): mixture of N(2, 0.8²) and N(7, 1.0²) with weights 0.4 and
  0.6, the same distribution as in the lecture figures. Generator parameters are kept separately
  in `app.js`.
- **Generator**: `x = μ_k + σ_k · e`, where the component `k` is chosen with a uniform number `u`
  and `e` is standard normal. The noise `(u, e)` is drawn once per seed and sample count, so the
  samples move smoothly while the sliders change.
- **Training data**: 1,000 samples from the target with a fixed seed (2026).
- **Fit by maximum likelihood** uses only the training data: sample mean and standard deviation
  for a single Gaussian, 200 EM iterations (initialized at the 25% and 75% quantiles) for the
  mixture. The fitted values are rounded to the slider steps.

## Measures

```
fidelity       = share of generated samples x with p_data(x)  >= t_data   (target's 95% region)
coverage       = share of training samples  x with p_theta(x) >= t_theta  (generator's 95% region)
log-likelihood = (1/N) sum_i log p_theta(x_i) over the training samples
```

The thresholds `t` are computed numerically so that `{x : p(x) >= t}` contains 95% of the
probability (highest-density region). Fidelity and coverage are toy versions of precision and
recall for generative models (Kynkäänniemi et al., 2019); a perfect generator reaches about 95%
on both.

Values with the default training data (10,000 generated samples):

| Generator                 | Fidelity | Coverage | Log-likelihood |
| ------------------------- | -------- | -------- | -------------- |
| Mode collapse, N(7, 0.6²) | 100%     | 44%      | −16.1          |
| ML single Gaussian        | 69%      | 100%     | −2.39          |
| Too broad, N(4.5, 4²)     | 58%      | 100%     | −2.53          |
| ML mixture                | 95%      | 94%      | −2.02          |
| Target itself             |          |          | −2.02          |

## Embedding

````markdown
```{=html}
<iframe name="distribution-fitting" loading="lazy" data-auto-height
        src="../../demos/genai_intro/distribution_fitting/index.html" width="100%" height="900"
        style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: learning a 1D distribution"></iframe>
```
````

With `data-auto-height` the iframe follows the height of the demo when page and demo are
served from the same site; when the page is opened via `file://`, the given `height` is used.
The folder is listed under `project.resources` in `_quarto.yml` (without `tests/`).

## Tests

```bash
quarto run demos/genai_intro/distribution_fitting/tests/dist_test.ts
```
