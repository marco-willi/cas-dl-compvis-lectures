// One-dimensional Gaussian mixtures: densities, sampling, maximum-likelihood fitting and
// simple distribution comparisons. A model is { means: [...], stds: [...], weights: [...] }.
// Pure functions without DOM access, shared by the page (app.js) and the tests.
globalThis.Dist = (() => {
  "use strict";

  // The true data distribution. It is never changed by the interface; generator parameters
  // are kept separately in app.js.
  const TARGET = Object.freeze({
    means: Object.freeze([2, 7]),
    stds: Object.freeze([0.8, 1.0]),
    weights: Object.freeze([0.4, 0.6]),
  });

  // Range used for plotting and for numerical integration.
  const X_MIN = -3;
  const X_MAX = 12;

  // Seeded pseudo-random numbers in [0, 1) (mulberry32), so that a seed always gives the
  // same samples.
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Random input of a generator: for each sample one uniform number (selects the mixture
  // component) and one standard-normal number (Box-Muller). The same noise is reused when
  // the model parameters change, so samples move smoothly with the sliders.
  function noise(n, seed) {
    const next = rng(seed);
    const u = new Float64Array(n);
    const e = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      u[i] = next();
      const r = Math.sqrt(-2 * Math.log(1 - next()));
      e[i] = r * Math.cos(2 * Math.PI * next());
    }
    return { u, e };
  }

  function normalPdf(x, mean, std) {
    const z = (x - mean) / std;
    return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
  }

  function pdf(x, model) {
    let p = 0;
    for (let k = 0; k < model.means.length; k++) p += model.weights[k] * normalPdf(x, model.means[k], model.stds[k]);
    return p;
  }

  // Generator: x = mean_k + std_k * e, where the component k is chosen with u.
  function sample(model, { u, e }) {
    const out = new Float64Array(e.length);
    for (let i = 0; i < e.length; i++) {
      let k = 0;
      let acc = model.weights[0];
      while (u[i] >= acc && k < model.means.length - 1) acc += model.weights[++k];
      out[i] = model.means[k] + model.stds[k] * e[i];
    }
    return out;
  }

  // Training data: n samples from the target with their own seed.
  const trainingData = (n = 1000, seed = 2026) => sample(TARGET, noise(n, seed));

  // Histogram normalized as a density (area 1 for the samples inside [lo, hi]), so it can be
  // compared with probability density functions.
  function histogram(samples, lo = X_MIN, hi = X_MAX, bins = 40) {
    const width = (hi - lo) / bins;
    const counts = new Array(bins).fill(0);
    for (const x of samples) {
      if (x < lo || x >= hi) continue;
      counts[Math.floor((x - lo) / width)]++;
    }
    const n = samples.length || 1;
    return counts.map((c, i) => ({ x0: lo + i * width, x1: lo + (i + 1) * width, density: c / (n * width) }));
  }

  // Average log-likelihood (1/N) sum log p(x_i). Densities are floored to avoid -Infinity.
  function logLikelihood(data, model) {
    let sum = 0;
    for (const x of data) sum += Math.log(Math.max(pdf(x, model), 1e-300));
    return sum / data.length;
  }

  // Maximum-likelihood fit of a single Gaussian: sample mean and (biased) standard deviation.
  function fitGaussian(data) {
    const n = data.length;
    let mean = 0;
    for (const x of data) mean += x;
    mean /= n;
    let v = 0;
    for (const x of data) v += (x - mean) ** 2;
    return { means: [mean], stds: [Math.sqrt(v / n)], weights: [1] };
  }

  // Maximum-likelihood fit of a mixture of two Gaussians with the EM algorithm. Deterministic
  // initialization at the 25% and 75% quantiles of the data.
  function fitMixture(data, iterations = 200) {
    const sorted = Array.from(data).sort((a, b) => a - b);
    const quantile = (q) => sorted[Math.floor(q * (sorted.length - 1))];
    const spread = fitGaussian(data).stds[0];
    let model = { means: [quantile(0.25), quantile(0.75)], stds: [spread / 2, spread / 2], weights: [0.5, 0.5] };
    const n = data.length;
    const resp = new Float64Array(n);
    for (let it = 0; it < iterations; it++) {
      // E-step: responsibility of component 0 for each data point.
      for (let i = 0; i < n; i++) {
        const p0 = model.weights[0] * normalPdf(data[i], model.means[0], model.stds[0]);
        const p1 = model.weights[1] * normalPdf(data[i], model.means[1], model.stds[1]);
        resp[i] = p0 + p1 > 0 ? p0 / (p0 + p1) : 0.5;
      }
      // M-step: weighted means, standard deviations and mixture weights.
      const next = { means: [0, 0], stds: [0, 0], weights: [0, 0] };
      for (let k = 0; k < 2; k++) {
        let w = 0;
        let m = 0;
        for (let i = 0; i < n; i++) {
          const r = k === 0 ? resp[i] : 1 - resp[i];
          w += r;
          m += r * data[i];
        }
        m /= w;
        let v = 0;
        for (let i = 0; i < n; i++) v += (k === 0 ? resp[i] : 1 - resp[i]) * (data[i] - m) ** 2;
        next.means[k] = m;
        next.stds[k] = Math.max(Math.sqrt(v / w), 1e-3);
        next.weights[k] = w / n;
      }
      model = next;
    }
    // Report the left component first.
    if (model.means[0] > model.means[1]) {
      for (const key of ["means", "stds", "weights"]) model[key].reverse();
    }
    return model;
  }

  // Density threshold of the highest-density region containing `mass` of the probability:
  // the region { x : p(x) >= threshold } has probability `mass` (numerically on a grid).
  function hdrThreshold(model, mass = 0.95, lo = -15, hi = 25, step = 0.005) {
    const values = [];
    for (let x = lo; x <= hi; x += step) values.push(pdf(x, model));
    values.sort((a, b) => b - a);
    let acc = 0;
    for (const v of values) {
      acc += v * step;
      if (acc >= mass) return v;
    }
    return 0;
  }

  // Share of the points that lie in the highest-density region (`mass`) of `model`.
  function shareInRegion(points, model, mass = 0.95) {
    const threshold = hdrThreshold(model, mass);
    let inside = 0;
    for (const x of points) if (pdf(x, model) >= threshold) inside++;
    return points.length ? inside / points.length : 0;
  }

  // Toy versions of precision and recall for generative models:
  //   fidelity: share of generated samples inside the 95% region of the target,
  //   coverage: share of real (training) samples inside the 95% region of the model.
  // A perfect model reaches about 95% on both.
  function compare(generated, real, model, mass = 0.95) {
    return {
      fidelity: shareInRegion(generated, TARGET, mass),
      coverage: shareInRegion(real, model, mass),
    };
  }

  return {
    TARGET, X_MIN, X_MAX,
    rng, noise, normalPdf, pdf, sample, trainingData, histogram,
    logLikelihood, fitGaussian, fitMixture, hdrThreshold, shareInRegion, compare,
  };
})();
