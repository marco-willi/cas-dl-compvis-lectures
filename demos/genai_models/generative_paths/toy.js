// Ideal (exact-formula) generative models for a 1-D toy problem: noise z ~ N(0, 1) is turned
// into samples of a fixed bimodal Gaussian mixture by a normalizing flow, flow matching and a
// diffusion model. "Ideal" means the models know the true target: the velocity field, the score
// and the flow layers are computed in closed form, so the only errors are numerical (too few
// steps), never errors of a trained network.
//
// Pure functions without DOM access, shared by the page (app.js) and the tests.
// A mixture is { weights: [...], means: [...], stds: [...] }.
globalThis.Toy = (() => {
  "use strict";

  // Same target as the intro lecture (demos/genai_intro/distribution_fitting/dist.js).
  const TARGET = Object.freeze({
    weights: Object.freeze([0.4, 0.6]),
    means: Object.freeze([2, 7]),
    stds: Object.freeze([0.8, 1.0]),
  });
  const NOISE = Object.freeze({ weights: [1], means: [0], stds: [1] });

  // Value range of all plots (noise and data share one axis).
  const Y_MIN = -4.5;
  const Y_MAX = 11.5;

  // ---- random numbers ---------------------------------------------------------------------

  // Seeded pseudo-random numbers in [0, 1) (mulberry32).
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

  // Standard-normal numbers (Box-Muller) from a uniform generator.
  function normals(n, next) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const r = Math.sqrt(-2 * Math.log(1 - next()));
      out[i] = r * Math.cos(2 * Math.PI * next());
    }
    return out;
  }

  // Independent random streams derived from one user-facing seed. Changing the sampler seed
  // never changes the initial noise, and vice versa.
  const STREAM = Object.freeze({ noise: 0x9e3779b1, sampler: 0x85ebca77, data: 0xc2b2ae3d });
  const stream = (seed, name) => rng(Math.imul(seed >>> 0, 2654435761) ^ STREAM[name]);

  // Initial noise of the generative process: n draws from N(0, 1).
  const initialNoise = (n, seed) => normals(n, stream(seed, "noise"));

  // n samples of a mixture from the given uniform generator.
  function sampleMixture(mix, n, next) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const u = next();
      let k = 0;
      let acc = mix.weights[0];
      while (u >= acc && k < mix.weights.length - 1) acc += mix.weights[++k];
      const r = Math.sqrt(-2 * Math.log(1 - next()));
      out[i] = mix.means[k] + mix.stds[k] * r * Math.cos(2 * Math.PI * next());
    }
    return out;
  }

  // ---- Gaussian mixtures --------------------------------------------------------------------

  const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

  // erfc with close to double precision: a series with positive terms for |x| < 2, a continued
  // fraction (modified Lentz) in the tails.
  function erfc(x) {
    if (x < 0) return 2 - erfc(-x);
    if (x < 2) {
      // erf(x) = 2/sqrt(pi) exp(-x^2) sum_n 2^n x^(2n+1) / (1*3*...*(2n+1))
      let term = x;
      let sum = x;
      for (let n = 1; n < 200; n++) {
        term *= (2 * x * x) / (2 * n + 1);
        sum += term;
        if (term < sum * 1e-17) break;
      }
      return 1 - (2 / Math.sqrt(Math.PI)) * Math.exp(-x * x) * sum;
    }
    // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
    const tiny = 1e-300;
    let f = x;
    let c = x;
    let d = 0;
    for (let n = 1; n < 1000; n++) {
      const a = n / 2;
      d = x + a * d;
      d = Math.abs(d) < tiny ? tiny : d;
      c = x + a / c;
      c = Math.abs(c) < tiny ? tiny : c;
      d = 1 / d;
      const delta = c * d;
      f *= delta;
      if (Math.abs(delta - 1) < 1e-16) break;
    }
    return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
  }

  const normalCdf = (x) => 0.5 * erfc(-x / Math.SQRT2);
  const normalSf = (x) => 0.5 * erfc(x / Math.SQRT2);

  // log of the weighted component densities, and their log-sum-exp.
  function logComponents(x, mix) {
    const out = new Array(mix.weights.length);
    for (let k = 0; k < out.length; k++) {
      const z = (x - mix.means[k]) / mix.stds[k];
      out[k] = Math.log(mix.weights[k]) - Math.log(mix.stds[k]) - LOG_SQRT_2PI - 0.5 * z * z;
    }
    return out;
  }
  function logSumExp(values) {
    const m = Math.max(...values);
    if (m === -Infinity) return -Infinity;
    let s = 0;
    for (const v of values) s += Math.exp(v - m);
    return m + Math.log(s);
  }
  // Posterior probabilities of the components given x (stable for every x).
  function responsibilities(x, mix) {
    const lc = logComponents(x, mix);
    const l = logSumExp(lc);
    return lc.map((v) => Math.exp(v - l));
  }

  // Fast evaluation for many x with the same mixture: precomputed constants and responsibilities
  // written into a reusable array (the inner loops of the samplers call this millions of times).
  function prepare(mix) {
    const K = mix.weights.length;
    const c = new Float64Array(K);
    const iv = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      c[k] = Math.log(mix.weights[k]) - Math.log(mix.stds[k]) - LOG_SQRT_2PI;
      iv[k] = 1 / (mix.stds[k] * mix.stds[k]);
    }
    return { K, c, iv, mu: Float64Array.from(mix.means), r: new Float64Array(K) };
  }
  function fillResponsibilities(x, P) {
    let m = -Infinity;
    for (let k = 0; k < P.K; k++) {
      const d = x - P.mu[k];
      P.r[k] = P.c[k] - 0.5 * d * d * P.iv[k];
      if (P.r[k] > m) m = P.r[k];
    }
    let s = 0;
    for (let k = 0; k < P.K; k++) s += (P.r[k] = Math.exp(P.r[k] - m));
    for (let k = 0; k < P.K; k++) P.r[k] /= s;
    return P.r;
  }

  const logPdf = (x, mix) => logSumExp(logComponents(x, mix));
  const pdf = (x, mix) => Math.exp(logPdf(x, mix));
  function cdf(x, mix) {
    let s = 0;
    for (let k = 0; k < mix.weights.length; k++) s += mix.weights[k] * normalCdf((x - mix.means[k]) / mix.stds[k]);
    return s;
  }
  // Survival function 1 - cdf, computed directly so that the upper tail keeps its precision.
  function sf(x, mix) {
    let s = 0;
    for (let k = 0; k < mix.weights.length; k++) s += mix.weights[k] * normalSf((x - mix.means[k]) / mix.stds[k]);
    return s;
  }

  // Quantile: the x with cdf(x) = p (lower = true) or sf(x) = p (lower = false). Newton steps
  // safeguarded by bisection; accurate to about 1e-12.
  function quantile(p, mix, lower = true) {
    let lo = -60;
    let hi = 60;
    let x = 0;
    for (let k = 0; k < mix.weights.length; k++) x += mix.weights[k] * mix.means[k];
    for (let it = 0; it < 200; it++) {
      const g = lower ? cdf(x, mix) - p : p - sf(x, mix); // increasing in x
      if (g > 0) hi = x;
      else lo = x;
      const d = pdf(x, mix);
      let next = d > 0 ? x - g / d : NaN;
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      if (Math.abs(next - x) < 1e-13 * (1 + Math.abs(x)) || hi - lo < 1e-13) return next;
      x = next;
    }
    return x;
  }

  // Score d/dx log p(x) of a mixture.
  function score(x, mix) {
    const r = responsibilities(x, mix);
    let s = 0;
    for (let k = 0; k < r.length; k++) s -= (r[k] * (x - mix.means[k])) / (mix.stds[k] * mix.stds[k]);
    return s;
  }

  // ---- normalizing flow ---------------------------------------------------------------------
  // K exact invertible layers. Intermediate distributions p_k (k = 0..K) turn N(0, 1) into the
  // target in three stages, each on one third of the progress lambda = k / K:
  //   1. separate: split N(0, 1) into the two modes, centred at 0 (offsets m_j - mean),
  //   2. move:     shift both modes to the position of the data,
  //   3. sharpen:  give each mode its own width s_j.
  // With K = 3, every layer has exactly one job. So p_0 = N(0, 1) and p_K = target. Layer k is
  // the monotone map
  //   f_k = F_k^-1 o F_{k-1}   (F = cdf),
  // which turns p_{k-1} into p_k exactly. Its derivative is f_k'(h) = p_{k-1}(h) / p_k(f_k(h)).
  const NF_JOBS = Object.freeze(["separate the modes", "move them to the data", "sharpen the modes"]);
  const TARGET_MEAN = TARGET.weights.reduce((s, w, j) => s + w * TARGET.means[j], 0);
  const ramp = (lambda, stage) => Math.min(Math.max(3 * lambda - stage, 0), 1);

  function nfStage(lambda) {
    return {
      weights: TARGET.weights.slice(),
      means: TARGET.means.map((m) => ramp(lambda, 0) * (m - TARGET_MEAN) + ramp(lambda, 1) * TARGET_MEAN),
      stds: TARGET.stds.map((s) => 1 + ramp(lambda, 2) * (s - 1)),
    };
  }
  const nfLayerDist = (k, K) => nfStage(k / K);
  // Jobs of layer k (the stages whose ramps change between lambda = (k-1)/K and k/K).
  function nfLayerJobs(k, K) {
    const a = (k - 1) / K;
    const b = k / K;
    return NF_JOBS.filter((_, stage) => ramp(b, stage) > ramp(a, stage));
  }

  // Apply F_to^-1 o F_from to h, using the lower or upper tail for precision.
  function transport(h, from, to) {
    const p = cdf(h, from);
    return p < 0.5 ? quantile(p, to, true) : quantile(sf(h, from), to, false);
  }
  const nfForward = (h, k, K) => transport(h, nfLayerDist(k - 1, K), nfLayerDist(k, K));
  const nfInverse = (x, k, K) => transport(x, nfLayerDist(k, K), nfLayerDist(k - 1, K));
  // log |f_k'(h)|
  const nfLogDet = (h, k, K) => logPdf(h, nfLayerDist(k - 1, K)) - logPdf(nfForward(h, k, K), nfLayerDist(k, K));

  // States h_0 = z, ..., h_K = x for each starting value. Layers are applied one after the other.
  function nfStates(z, K) {
    const n = z.length;
    const states = new Float32Array((K + 1) * n);
    const h = Float64Array.from(z);
    states.set(h, 0);
    for (let k = 1; k <= K; k++) {
      const from = nfLayerDist(k - 1, K);
      const to = nfLayerDist(k, K);
      for (let i = 0; i < n; i++) h[i] = transport(h[i], from, to);
      states.set(h, k * n);
    }
    return states;
  }

  // ---- flow matching ------------------------------------------------------------------------
  // Linear paths x_t = (1 - t) x_0 + t x_1 between noise x_0 ~ N(0, 1) (t = 0) and data
  // x_1 ~ target (t = 1). The marginal p_t is again a mixture, and the marginal velocity
  //   u(x, t) = E[x_1 - x_0 | x_t = x]
  // is the average direction of all straight training lines through (x, t). Per component k,
  // (x_0, x_1, x_t) are jointly Gaussian, which gives u in closed form.

  function fmMarginal(t) {
    return {
      weights: TARGET.weights.slice(),
      means: TARGET.means.map((m) => t * m),
      stds: TARGET.stds.map((s) => Math.sqrt((1 - t) ** 2 + t * t * s * s)),
    };
  }

  // Velocity field u(., t) for one time t.
  function fmField(t) {
    const mix = fmMarginal(t);
    const P = prepare(mix);
    // E[x_1 - x_0 | x_t, k] = m_k + Cov(x_1 - x_0, x_t) / Var(x_t) * (x_t - t m_k)
    const slope = TARGET.stds.map((s, k) => (t * s * s - (1 - t)) / mix.stds[k] ** 2);
    return (x) => {
      const r = fillResponsibilities(x, P);
      let u = 0;
      for (let k = 0; k < P.K; k++) u += r[k] * (TARGET.means[k] + slope[k] * (x - P.mu[k]));
      return u;
    };
  }
  const fmVelocity = (x, t) => fmField(t)(x);

  // Integrate dx/dt = u(x, t) from t = 0 to t = 1 with `steps` Euler or RK4 steps.
  function fmStates(z, steps, method = "euler") {
    const n = z.length;
    const states = new Float32Array((steps + 1) * n);
    const x = Float64Array.from(z);
    states.set(x, 0);
    const dt = 1 / steps;
    for (let j = 0; j < steps; j++) {
      const t = j * dt;
      const u0 = fmField(t);
      if (method === "rk4") {
        const uh = fmField(t + 0.5 * dt);
        const u1 = fmField(t + dt);
        for (let i = 0; i < n; i++) {
          const k1 = u0(x[i]);
          const k2 = uh(x[i] + 0.5 * dt * k1);
          const k3 = uh(x[i] + 0.5 * dt * k2);
          const k4 = u1(x[i] + dt * k3);
          x[i] += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
        }
      } else {
        for (let i = 0; i < n; i++) x[i] += dt * u0(x[i]);
      }
      states.set(x, (j + 1) * n);
    }
    return states;
  }

  // Training pairs: each noise sample is paired with an independent data sample and moves on
  // the straight line between them. `cols` + 1 states at t = 0, 1/cols, ..., 1.
  function fmPairStates(z, x1, cols) {
    const n = z.length;
    const states = new Float32Array((cols + 1) * n);
    for (let c = 0; c <= cols; c++) {
      const t = c / cols;
      for (let i = 0; i < n; i++) states[c * n + i] = (1 - t) * z[i] + t * x1[i];
    }
    return states;
  }

  // ---- diffusion ----------------------------------------------------------------------------
  // One variance-preserving (VP) noise schedule, used by every sampler below. Continuous time
  // t in [0, 1] (t = 0: data, t = 1 = T: noise), linear beta(t) from 0.1 to 20, which is the
  // continuous version of the DDPM schedule with beta from 1e-4 to 0.02 over 1000 steps:
  //   alphaBar(t) = exp(-integral_0^t beta(s) ds),   x_t = sqrt(alphaBar) x_0 + sqrt(1 - alphaBar) eps.
  const SCHEDULE = Object.freeze({ betaMin: 0.1, betaMax: 20 });
  const beta = (t) => SCHEDULE.betaMin + t * (SCHEDULE.betaMax - SCHEDULE.betaMin);
  const alphaBar = (t) => Math.exp(-(SCHEDULE.betaMin * t + 0.5 * (SCHEDULE.betaMax - SCHEDULE.betaMin) * t * t));

  // Exact marginal p_t of the noised target.
  function diffMarginal(t) {
    const ab = alphaBar(t);
    const a = Math.sqrt(ab);
    return {
      weights: TARGET.weights.slice(),
      means: TARGET.means.map((m) => a * m),
      stds: TARGET.stds.map((s) => Math.sqrt(ab * s * s + 1 - ab)),
    };
  }

  // Ideal denoiser E[x_0 | x_t = x] (closed form for a mixture; avoids dividing by tiny
  // sqrt(alphaBar) near t = T).
  function diffDenoiser(t) {
    const a = Math.sqrt(alphaBar(t));
    const mix = diffMarginal(t);
    const P = prepare(mix);
    const slope = TARGET.stds.map((s, k) => (a * s * s) / mix.stds[k] ** 2);
    return (x) => {
      const r = fillResponsibilities(x, P);
      let m = 0;
      for (let k = 0; k < P.K; k++) m += r[k] * (TARGET.means[k] + slope[k] * (x - P.mu[k]));
      return m;
    };
  }
  function diffScoreField(t) {
    const P = prepare(diffMarginal(t));
    return (x) => {
      const r = fillResponsibilities(x, P);
      let s = 0;
      for (let k = 0; k < P.K; k++) s -= r[k] * (x - P.mu[k]) * P.iv[k];
      return s;
    };
  }
  const diffDenoise = (x, t) => diffDenoiser(t)(x);
  const diffScore = (x, t) => diffScoreField(t)(x);

  // Starting point x_T for the noise value z. The schedule leaves a small trace of the data at
  // t = T (p_T has mean sqrt(alphaBar(T)) * 5 = 0.03 instead of 0). Real models start from N(0, 1)
  // and accept this; the ideal model maps z exactly onto p_T (x_T = z + 0.03 within 1e-4), so
  // that its samplers are exact. Without this, the deterministic samplers would be off by a
  // noticeable amount in the gap between the modes.
  const diffPrior = (z) => transport(z, NOISE, diffMarginal(1));

  // Time of column c of an n-step sampler (c = 0: t = T = 1, c = steps: t = 0).
  const diffTime = (c, steps) => 1 - c / steps;

  // Reverse sampling from x_T = z.
  //   "ddpm": stochastic ancestral sampler (DDIM update with eta = 1), uses the sampler stream,
  //   "ddim": deterministic DDIM update (eta = 0),
  //   "ode":  probability-flow ODE dx/dt = -1/2 beta(t) (x + score), Heun's method.
  // All three read the same alphaBar(t).
  function diffReverseStates(z, steps, sampler, samplerSeed = 1) {
    const n = z.length;
    const states = new Float32Array((steps + 1) * n);
    const x = Float64Array.from(z, diffPrior);
    states.set(x, 0);
    const next = stream(samplerSeed, "sampler");
    for (let j = 0; j < steps; j++) {
      const t = diffTime(j, steps);
      const s = diffTime(j + 1, steps);
      if (sampler === "ode") {
        const dt = s - t; // negative
        const scoreT = diffScoreField(t);
        const scoreS = diffScoreField(s);
        for (let i = 0; i < n; i++) {
          const k1 = -0.5 * beta(t) * (x[i] + scoreT(x[i]));
          const y = x[i] + dt * k1;
          const k2 = -0.5 * beta(s) * (y + scoreS(y));
          x[i] += 0.5 * dt * (k1 + k2);
        }
      } else {
        const eta = sampler === "ddpm" ? 1 : 0;
        const abT = alphaBar(t);
        const abS = alphaBar(s);
        const sigma = eta * Math.sqrt(((1 - abS) / (1 - abT)) * (1 - abT / abS));
        const keep = Math.sqrt(Math.max(0, 1 - abS - sigma * sigma));
        const xi = eta > 0 ? normals(n, next) : null;
        const denoise = diffDenoiser(t);
        for (let i = 0; i < n; i++) {
          const x0 = denoise(x[i]);
          const eps = (x[i] - Math.sqrt(abT) * x0) / Math.sqrt(1 - abT);
          x[i] = Math.sqrt(abS) * x0 + keep * eps + (xi ? sigma * xi[i] : 0);
        }
      }
      states.set(x, (j + 1) * n);
    }
    return states;
  }

  // Forward noising of data samples x0 with exact transitions between the column times.
  // Returned in the same column order as the reverse process (column `steps` holds x0).
  function diffForwardStates(x0, steps, samplerSeed = 1) {
    const n = x0.length;
    const states = new Float32Array((steps + 1) * n);
    const x = Float64Array.from(x0);
    states.set(x, steps * n);
    const next = stream(samplerSeed, "sampler");
    for (let c = steps - 1; c >= 0; c--) {
      const ratio = alphaBar(diffTime(c, steps)) / alphaBar(diffTime(c + 1, steps));
      const xi = normals(n, next);
      for (let i = 0; i < n; i++) x[i] = Math.sqrt(ratio) * x[i] + Math.sqrt(1 - ratio) * xi[i];
      states.set(x, c * n);
    }
    return states;
  }

  // Density evaluation with the flow (the "x → z" direction): invert the layers one by one and
  // add up the change-of-variables terms. Returns h_K = x, ..., h_0 = z (in this order), the
  // terms log|f_k'(h_{k-1})| for k = K..1, and log p(x) = log N(z) - sum_k log|f_k'(h_{k-1})|.
  function nfEvaluate(x, K) {
    const h = [x];
    const logDets = [];
    let cur = x;
    for (let k = K; k >= 1; k--) {
      const prev = nfInverse(cur, k, K);
      logDets.push(nfLogDet(prev, k, K));
      h.push(prev);
      cur = prev;
    }
    const logNz = logPdf(cur, NOISE);
    return { h, logDets, z: cur, logNz, logp: logNz - logDets.reduce((a, b) => a + b, 0) };
  }

  // Many independent (noise, data) pairs, as used in flow-matching training.
  function fmPairs(n, seed) {
    return { x0: normals(n, stream(seed, "noise")), x1: sampleMixture(TARGET, n, stream(seed, "data")) };
  }
  // Training directions x1 - x0 of all pairs whose straight line passes within `band` of x at
  // time t. Their mean estimates u(x, t): this average is what the network learns.
  function fmPairAverage(pairs, x, t, band = 0.1) {
    const dirs = [];
    for (let i = 0; i < pairs.x0.length; i++) {
      if (Math.abs((1 - t) * pairs.x0[i] + t * pairs.x1[i] - x) < band) dirs.push(pairs.x1[i] - pairs.x0[i]);
    }
    const mean = dirs.length ? dirs.reduce((a, b) => a + b, 0) / dirs.length : NaN;
    return { directions: Float64Array.from(dirs), mean, count: dirs.length };
  }

  // ---- autoregressive -----------------------------------------------------------------------
  // 1-D: a value written as two "digits" (tokens), like a pixel value in a 256-way softmax:
  //   coarse token c in 0..11: the unit interval [-1 + c, c) the value lies in,
  //   fine token   f in 0..9:  the tenth within that interval.
  // The model is p(c, f) = p(c) p(f | c), with exact tables from the target cdf (the tails below
  // -1 and above 11 are added to the outermost bins). The generated value is the bin centre
  // x = -1 + c + (f + 0.5) / 10, so the model is discrete with resolution 0.1.
  const AR = Object.freeze({ lo: -1, coarse: 12, fine: 10 });
  const arEdge = (c, f = 0) => AR.lo + c + f / AR.fine;
  const arValue = (c, f) => AR.lo + c + (f + 0.5) / AR.fine;

  // Probability that the target lies in [a, b), with the outer tails added to the outer bins.
  function binMass(a, b, first, last) {
    const lo = first ? 0 : cdf(a, TARGET);
    const hi = last ? 1 : cdf(b, TARGET);
    return hi - lo;
  }

  // Sampling temperature: p_tau proportional to p^(1/tau), applied to every categorical.
  function withTemperature(p, tau) {
    if (tau === 1) return Float64Array.from(p);
    const q = Float64Array.from(p, (v) => (v > 0 ? Math.exp(Math.log(v) / tau) : 0));
    const s = q.reduce((a, b) => a + b, 0);
    return q.map((v) => v / s);
  }

  function arTables(tau = 1) {
    const joint = [];
    const pc = new Float64Array(AR.coarse);
    for (let c = 0; c < AR.coarse; c++) {
      const row = new Float64Array(AR.fine);
      for (let f = 0; f < AR.fine; f++) {
        row[f] = binMass(arEdge(c, f), arEdge(c, f + 1), c === 0 && f === 0, c === AR.coarse - 1 && f === AR.fine - 1);
      }
      joint.push(row);
      pc[c] = row.reduce((a, b) => a + b, 0);
    }
    return {
      pc: withTemperature(pc, tau),
      pf: joint.map((row, c) => withTemperature(row.map((v) => v / pc[c]), tau)),
    };
  }

  // Index chosen by a uniform number u: the first category whose cumulative probability exceeds u.
  function categorical(p, u) {
    let acc = 0;
    for (let k = 0; k < p.length; k++) {
      acc += p[k];
      if (u < acc) return k;
    }
    return p.length - 1;
  }

  // Ancestral sampling: first c ~ p(c) with u1, then f ~ p(f | c) with u2.
  function arSample(n, seed, tau = 1) {
    const tables = arTables(tau);
    const next = stream(seed, "noise");
    const u1 = new Float64Array(n);
    const u2 = new Float64Array(n);
    const c = new Int32Array(n);
    const f = new Int32Array(n);
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      u1[i] = next();
      u2[i] = next();
      c[i] = categorical(tables.pc, u1[i]);
      f[i] = categorical(tables.pf[c[i]], u2[i]);
      x[i] = arValue(c[i], f[i]);
    }
    return { tables, u1, u2, c, f, x, coarse: Float64Array.from(c, (v) => AR.lo + v + 0.5) };
  }

  // 2-D companion: p(x1, x2) = p(x1) p(x2 | x1). x1 follows the shared 1-D target; given x1,
  //   x2 ~ N(mu(x1), sigma(x1)^2),  mu(x1) = 1 + 0.25 (x1 - 4.5)^2,  sigma(x1) = 0.5 + 0.06 x1,
  // so the two modes of x1 bend in opposite directions and the spread of x2 grows with x1.
  const AR2D = Object.freeze({
    mean: (x1) => 1 + 0.25 * (x1 - 4.5) ** 2,
    std: (x1) => 0.5 + 0.06 * x1,
  });
  const ar2dConditional = (x1) => ({ weights: [1], means: [AR2D.mean(x1)], stds: [AR2D.std(x1)] });
  const ar2dPdf = (x1, x2) => pdf(x1, TARGET) * pdf(x2, ar2dConditional(x1));
  function ar2dSample(n, seed) {
    const x1 = sampleMixture(TARGET, n, stream(seed, "data"));
    const e = normals(n, stream(seed, "noise"));
    const x2 = Float64Array.from(x1, (v, i) => AR2D.mean(v) + AR2D.std(v) * e[i]);
    return { x1, x2 };
  }

  // ---- autoregressive sequence ----------------------------------------------------------------
  // A short sequence x_1, ..., x_n (like a row of pixels), generated value by value:
  //   p(x_1, ..., x_n) = p(x_1) p(x_2 | x_1) ... p(x_n | x_1, ..., x_{n-1}).
  // The data come from two patterns with the target's weights (0.4 / 0.6): all sequences start
  // near SEQ.start and move towards the left mode (2) or the right mode (7). Each pattern is a
  // Gaussian with mean m_j(k) = start + g(k) (target mean_j - start), g(k) = ((k-1)/(n-1))^power,
  // std s_j(k) = startStd + g(k) (target std_j - startStd), and correlation rho^|k-l| between
  // positions. The last value therefore follows the target exactly. Every conditional
  // p(x_k | x_1..x_{k-1}) is a mixture of two Gaussians in closed form; its weights say how well
  // the values so far fit each pattern.
  const SEQ = Object.freeze({ n: 8, start: 4.5, startStd: 0.6, rho: 0.7, power: 1.5 });

  const seqPatterns = (() => {
    const { n, start, startStd, rho, power } = SEQ;
    return TARGET.weights.map((w, j) => {
      const mean = [];
      const std = [];
      for (let k = 0; k < n; k++) {
        const g = Math.pow(k / (n - 1), power);
        mean.push(start + g * (TARGET.means[j] - start));
        std.push(startStd + g * (TARGET.stds[j] - startStd));
      }
      const cov = mean.map((_, a) => mean.map((__, b) => std[a] * std[b] * Math.pow(rho, Math.abs(a - b))));
      return { weight: w, mean, std, cov };
    });
  })();

  // Solve A v = b for a small symmetric positive-definite A (Cholesky).
  function cholSolve(A, b) {
    const m = A.length;
    const Lm = A.map(() => new Float64Array(m));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= Lm[i][k] * Lm[j][k];
        Lm[i][j] = i === j ? Math.sqrt(s) : s / Lm[j][j];
      }
    }
    const y = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= Lm[i][k] * y[k];
      y[i] = s / Lm[i][i];
    }
    const v = new Float64Array(m);
    for (let i = m - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < m; k++) s -= Lm[k][i] * v[k];
      v[i] = s / Lm[i][i];
    }
    let logDet = 0;
    for (let i = 0; i < m; i++) logDet += 2 * Math.log(Lm[i][i]);
    return { v, logDet };
  }

  // Log-density of the prefix under one pattern (multivariate normal of the first k positions).
  function seqPatternLogPdf(p, prefix) {
    const k = prefix.length;
    if (k === 0) return 0;
    const A = p.cov.slice(0, k).map((row) => row.slice(0, k));
    const d = prefix.map((x, i) => x - p.mean[i]);
    const { v, logDet } = cholSolve(A, d);
    let q = 0;
    for (let i = 0; i < k; i++) q += d[i] * v[i];
    return -0.5 * (q + logDet + k * Math.log(2 * Math.PI));
  }

  // p(x_{k+1} | prefix) with k = prefix.length: a mixture over the two patterns. The weights are the
  // posterior probabilities of the patterns given the prefix.
  // Posterior probability of each pattern given the values so far (any prefix length).
  function seqPosterior(prefix) {
    const logW = seqPatterns.map((p) => Math.log(p.weight) + seqPatternLogPdf(p, prefix));
    const lse = logSumExp(logW);
    return logW.map((l) => Math.exp(l - lse));
  }

  function seqConditional(prefix) {
    const k = prefix.length;
    const means = [];
    const stds = [];
    for (const p of seqPatterns) {
      if (k === 0) {
        means.push(p.mean[0]);
        stds.push(p.std[0]);
        continue;
      }
      const A = p.cov.slice(0, k).map((row) => row.slice(0, k));
      const c = p.cov[k].slice(0, k);
      const d = prefix.map((x, i) => x - p.mean[i]);
      const sd = cholSolve(A, d).v;
      const sc = cholSolve(A, c).v;
      let m = p.mean[k];
      let v = p.cov[k][k];
      for (let i = 0; i < k; i++) {
        m += c[i] * sd[i];
        v -= c[i] * sc[i];
      }
      means.push(m);
      stds.push(Math.sqrt(Math.max(v, 1e-12)));
    }
    return { weights: seqPosterior(prefix), means, stds };
  }

  // Marginal distribution of position k (0-based), without any context.
  const seqMarginal = (k) => ({ weights: seqPatterns.map((p) => p.weight), means: seqPatterns.map((p) => p.mean[k]), stds: seqPatterns.map((p) => p.std[k]) });
  // Log-density of a prefix (marginal over the patterns).
  const seqLogPdf = (prefix) => logSumExp(seqPatterns.map((p) => Math.log(p.weight) + seqPatternLogPdf(p, prefix)));

  // Ancestral sampling, value by value from the conditionals (u picks the mixture component, e the
  // value within it). Returns n sequences. Uses the "data" stream: the sequences play the role of
  // data samples.
  function seqSample(n, seed) {
    const next = stream(seed, "data");
    const out = [];
    for (let i = 0; i < n; i++) {
      const xs = [];
      for (let k = 0; k < SEQ.n; k++) {
        const c = seqConditional(xs);
        const j = categorical(c.weights, next());
        const r = Math.sqrt(-2 * Math.log(1 - next()));
        xs.push(c.means[j] + c.stds[j] * r * Math.cos(2 * Math.PI * next()));
      }
      out.push(Float64Array.from(xs));
    }
    return out;
  }

  // ---- trained models (GAN, VAE) --------------------------------------------------------------
  // Exported by tools/train_models.py as lookup tables on regular grids ({ lo, hi, n }); the page
  // evaluates them by linear interpolation (linear extrapolation outside the grid).
  function gridInterp(grid, values, v) {
    const step = (grid.hi - grid.lo) / (grid.n - 1);
    const pos = (v - grid.lo) / step;
    const i = Math.min(Math.max(Math.floor(pos), 0), grid.n - 2);
    const f = pos - i;
    return values[i] + f * (values[i + 1] - values[i]);
  }
  // GAN: x = G(z), one function evaluation per sample.
  const ganGenerate = (models, snapshot, z) => Float64Array.from(z, (v) => gridInterp(models.zGrid, snapshot.g, v));
  const ganDiscriminator = (models, snapshot, x) => gridInterp(models.xGrid, snapshot.d, x);
  // VAE: decoder p(x | z) = N(mean(z), std(z)^2), encoder q(z | x) = N(mean(x), std(x)^2).
  const vaeDecoder = (models, z) => ({ mean: gridInterp(models.zGrid, models.vae.decMean, z), std: gridInterp(models.zGrid, models.vae.decStd, z) });
  const vaeEncoder = (models, x) => ({ mean: gridInterp(models.xGrid, models.vae.encMean, x), std: gridInterp(models.xGrid, models.vae.encStd, x) });
  // Ancestral sampling: z -> decoder parameters -> x = mean + std * e (or the mean only).
  function vaeGenerate(models, z, e, withNoise = true) {
    return Float64Array.from(z, (v, i) => {
      const d = vaeDecoder(models, v);
      return withNoise ? d.mean + d.std * e[i] : d.mean;
    });
  }

  // ---- comparisons --------------------------------------------------------------------------

  // Wasserstein-1 distance between samples and a mixture: integral |F_n(x) - F(x)| dx.
  function wasserstein1(samples, mix, lo = -15, hi = 25, step = 0.002) {
    const sorted = Float64Array.from(samples).sort();
    const n = sorted.length;
    let j = 0;
    let sum = 0;
    for (let x = lo + step / 2; x < hi; x += step) {
      while (j < n && sorted[j] <= x) j++;
      sum += Math.abs(j / n - cdf(x, mix)) * step;
    }
    return sum;
  }

  // Mean |x - x_ref| per sample: error of a deterministic sampler against an accurate reference
  // started from the same noise.
  function referenceError(x, ref) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += Math.abs(x[i] - ref[i]);
    return sum / x.length;
  }

  // Kolmogorov-Smirnov statistic: largest difference between the empirical and the true cdf.
  function ks(samples, mix) {
    const sorted = Float64Array.from(samples).sort();
    const n = sorted.length;
    let d = 0;
    for (let i = 0; i < n; i++) {
      const f = cdf(sorted[i], mix);
      d = Math.max(d, Math.abs(f - i / n), Math.abs(f - (i + 1) / n));
    }
    return d;
  }

  function histogram(values, lo, hi, bins) {
    const width = (hi - lo) / bins;
    const counts = new Array(bins).fill(0);
    for (const v of values) {
      if (v < lo || v >= hi) continue;
      counts[Math.floor((v - lo) / width)]++;
    }
    const n = values.length || 1;
    return counts.map((c, i) => ({ y0: lo + i * width, y1: lo + (i + 1) * width, density: c / (n * width) }));
  }

  // ---- views --------------------------------------------------------------------------------
  // A view is everything the page draws for one family and setting:
  //   states:   Float32Array, column-major: states[c * n + i] = value of sample i in column c,
  //   cols:     number of columns (positions along the generative process, left to right),
  //   pos(c):   horizontal position of column c in [0, 1],
  //   marginal(c): exact distribution of the ideal process in column c,
  //   density(p): exact distribution at any position p in [0, 1] (continuous processes only),
  //   colorKey: per sample in [0, 1] (quantile of the starting value), for coloring paths,
  //   nfe:      number of model evaluations per sample.
  function simulate({ family, mode, steps, layers, sampler, seed, samplerSeed, n }) {
    const z = initialNoise(n, seed);
    const noiseQuantile = Float64Array.from(z, (v) => normalCdf(v));
    if (family === "nf") {
      return {
        states: nfStates(z, layers), cols: layers + 1, n,
        pos: (c) => c / layers, marginal: (c) => nfLayerDist(c, layers),
        density: null, colorKey: noiseQuantile, nfe: layers, deterministic: true,
      };
    }
    if (family === "fm") {
      if (mode === "pairs") {
        const cols = 100;
        const x1 = sampleMixture(TARGET, n, stream(samplerSeed, "data"));
        return {
          states: fmPairStates(z, x1, cols), cols: cols + 1, n,
          pos: (c) => c / cols, marginal: (c) => fmMarginal(c / cols), density: fmMarginal,
          colorKey: noiseQuantile, nfe: 0, deterministic: false, straight: true,
        };
      }
      return {
        states: fmStates(z, steps, "euler"), cols: steps + 1, n,
        pos: (c) => c / steps, marginal: (c) => fmMarginal(c / steps), density: fmMarginal,
        colorKey: noiseQuantile, nfe: steps, deterministic: true,
      };
    }
    // diffusion
    if (mode === "forward") {
      const x0 = sampleMixture(TARGET, n, stream(seed, "data"));
      return {
        states: diffForwardStates(x0, steps, samplerSeed), cols: steps + 1, n,
        pos: (c) => c / steps, marginal: (c) => diffMarginal(diffTime(c, steps)), density: (p) => diffMarginal(1 - p),
        colorKey: Float64Array.from(x0, (v) => cdf(v, TARGET)), nfe: 0, deterministic: false, forward: true,
      };
    }
    return {
      states: diffReverseStates(z, steps, sampler, samplerSeed), cols: steps + 1, n,
      pos: (c) => c / steps, marginal: (c) => diffMarginal(diffTime(c, steps)), density: (p) => diffMarginal(1 - p),
      colorKey: noiseQuantile, nfe: sampler === "ode" ? 2 * steps - 1 : steps, deterministic: sampler !== "ddpm",
    };
  }

  return {
    TARGET, NOISE, Y_MIN, Y_MAX, SCHEDULE,
    rng, normals, stream, initialNoise, sampleMixture,
    erfc, normalCdf, normalSf, logPdf, pdf, cdf, sf, quantile, score, responsibilities,
    NF_JOBS, nfLayerDist, nfLayerJobs, nfForward, nfInverse, nfLogDet, nfStates, nfEvaluate,
    fmPairs, fmPairAverage,
    AR, arEdge, arValue, arTables, categorical, arSample, AR2D, ar2dConditional, ar2dPdf, ar2dSample,
    SEQ, seqPosterior, seqConditional, seqMarginal, seqLogPdf, seqSample,
    gridInterp, ganGenerate, ganDiscriminator, vaeDecoder, vaeEncoder, vaeGenerate,
    fmMarginal, fmVelocity, fmStates, fmPairStates,
    beta, alphaBar, diffMarginal, diffDenoise, diffScore, diffPrior, diffTime, diffReverseStates, diffForwardStates,
    wasserstein1, referenceError, ks, histogram, simulate,
  };
})();
