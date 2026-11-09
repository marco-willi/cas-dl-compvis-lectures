// Illustrative sampling processes on an image: no DOM access, no trained models.
// Images are Float32Arrays of RGB values in [-1, 1], row-major (H x W x 3).
(function () {
  "use strict";

  const T_DIFFUSION = 50; // displayed diffusion steps
  const N_FLOW = 50; // Euler steps of flow matching
  const Z_DIM = 2; // latent dimensions of GAN and VAE, so that z can be plotted
  const RHO = 0.9; // correlation of the injected noise between two reverse diffusion steps

  // ---- random numbers --------------------------------------------------------------------

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normals(seed, n) {
    const rand = mulberry32(seed);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      const u = 1 - rand(), v = rand();
      const r = Math.sqrt(-2 * Math.log(u));
      out[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return out;
  }

  // ---- conversions -----------------------------------------------------------------------

  function fromRGBA(rgba) {
    const n = rgba.length / 4;
    const x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) x[3 * i + c] = rgba[4 * i + c] / 127.5 - 1;
    return x;
  }

  function toRGBA(x, out) {
    const n = x.length / 3;
    out = out || new Uint8ClampedArray(4 * n);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) out[4 * i + c] = (x[3 * i + c] + 1) * 127.5;
      out[4 * i + 3] = 255;
    }
    return out;
  }

  // ---- diffusion and flow matching -------------------------------------------------------

  // cosine schedule (Nichol & Dhariwal, 2021): alpha_bar(0) = 1, alpha_bar(T) ~ 0
  function alphaBar(t, T = T_DIFFUSION) {
    const s = 0.008;
    const f = (u) => Math.cos(((u / T + s) / (1 + s)) * Math.PI / 2) ** 2;
    return Math.min(1, Math.max(0, f(t) / f(0)));
  }

  function mix(a, wa, b, wb) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = wa * a[i] + wb * b[i];
    return out;
  }

  // forward process, sampled directly from x0: x_t = sqrt(ab) x0 + sqrt(1 - ab) eps
  function diffusionForward(x0, eps, t) {
    const ab = alphaBar(t);
    return mix(x0, Math.sqrt(ab), eps, Math.sqrt(1 - ab));
  }

  // Reverse process, illustrative: the states have the noise level of step t, and the noise is
  // partly renewed in every step (fresh noise as in DDPM). `seed` sets the starting noise x_T,
  // `freshSeed` the noise added along the way. Returns the states x_T, ..., x_0.
  function diffusionReverse(x0, seed, freshSeed = seed, T = T_DIFFUSION) {
    let eps = normals(seed, x0.length);
    const states = [];
    for (let t = T; t >= 0; t--) {
      if (t < T) eps = mix(eps, RHO, normals(freshSeed * 7919 + t, x0.length), Math.sqrt(1 - RHO * RHO));
      const ab = alphaBar(t, T);
      states.push(mix(x0, Math.sqrt(ab), eps, Math.sqrt(1 - ab)));
    }
    return states;
  }

  // flow matching path, illustrative: the straight line between noise (t = 0) and data (t = 1)
  function flow(x1, eps, t) {
    return mix(eps, 1 - t, x1, t);
  }

  // ---- autoregressive tokens -------------------------------------------------------------

  // mean squared distance of patch `index` (raster order) to every codebook entry
  function patchDistances(x, width, codebook, patch, index) {
    const gw = width / patch;
    const r0 = Math.floor(index / gw) * patch, c0 = (index % gw) * patch;
    const k = codebook.length / (patch * patch * 3);
    const d = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      let sum = 0, p = j * patch * patch * 3;
      for (let r = 0; r < patch; r++) {
        for (let c = 0; c < patch; c++) {
          const q = 3 * ((r0 + r) * width + c0 + c);
          for (let ch = 0; ch < 3; ch++, p++) {
            const diff = x[q + ch] - (codebook[p] / 127.5 - 1);
            sum += diff * diff;
          }
        }
      }
      d[j] = sum / (patch * patch * 3);
    }
    return d;
  }

  // Illustrative p(x_i | x_<i): a softmax over the negative distances, so the tokens that match
  // the image's patch are the most likely. Returns probabilities and the most likely token.
  function tokenDistribution(x, width, codebook, patch, index, temperature = 0.04) {
    const d = patchDistances(x, width, codebook, patch, index);
    const dMin = Math.min(...d);
    const w = Array.from(d, (v) => Math.exp(-(v - dMin) / temperature));
    const total = w.reduce((a, b) => a + b, 0);
    const probs = w.map((v) => v / total);
    return { probs, token: probs.indexOf(Math.max(...probs)) };
  }

  // ---- GAN and VAE -----------------------------------------------------------------------

  function latent(seed) {
    return Array.from(normals(seed, Z_DIM));
  }

  // illustrative encoder output for image `index` of `n`: the means lie on a circle in the latent
  // space, the standard deviations are small
  function encode(index, n = 5) {
    const angle = (2 * Math.PI * index) / n + 0.4;
    const mu = [1.4 * Math.cos(angle), 1.4 * Math.sin(angle)];
    const sigma = Array.from(normals(2000 + index, Z_DIM), (v) => 0.2 + 0.08 * Math.abs(v));
    return { mu, sigma };
  }

  // reparameterization: z = mu + sigma * eps
  function sampleLatent(mu, sigma, seed) {
    const eps = normals(seed, Z_DIM);
    return mu.map((m, i) => m + sigma[i] * eps[i]);
  }

  globalThis.Sampling = {
    T_DIFFUSION, N_FLOW, Z_DIM,
    mulberry32, normals, fromRGBA, toRGBA,
    alphaBar, diffusionForward, diffusionReverse, flow,
    patchDistances, tokenDistribution,
    latent, encode, sampleLatent,
  };
})();
