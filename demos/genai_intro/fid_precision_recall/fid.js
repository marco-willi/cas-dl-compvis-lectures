// Two-dimensional stand-ins for image features: sampling, Frechet distance (FID) and
// precision / recall for generative models. Points are arrays of [x, y].
// Pure functions without DOM access, shared by the page (app.js) and the tests.
globalThis.Fid = (() => {
  "use strict";

  // The real features: three clusters, the same as in the lecture figure fid_pipeline.svg.
  // Never changed by the interface; generator parameters are kept separately in app.js.
  const CENTERS = Object.freeze([
    Object.freeze([-1.6, -0.9]),
    Object.freeze([1.6, -0.9]),
    Object.freeze([0.0, 1.5]),
  ]);
  const STD = 0.55;
  const REAL = Object.freeze({ shift: Object.freeze([0, 0]), spread: 1, clusters: Object.freeze([true, true, true]) });

  // Plot range.
  const X_MIN = -5;
  const X_MAX = 5.5;
  const Y_MIN = -4.5;
  const Y_MAX = 5;

  // Number of neighbours that defines the radius of each ball (Kynkaanniemi et al., 2019).
  const K = 3;

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

  // Random input: for each sample one uniform number (selects the cluster) and two
  // standard-normal numbers (Box-Muller). The same noise is reused when the parameters
  // change, so samples move smoothly with the sliders.
  function noise(n, seed) {
    const next = rng(seed);
    const u = new Float64Array(n);
    const e1 = new Float64Array(n);
    const e2 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      u[i] = next();
      const r = Math.sqrt(-2 * Math.log(1 - next()));
      const phi = 2 * Math.PI * next();
      e1[i] = r * Math.cos(phi);
      e2[i] = r * Math.sin(phi);
    }
    return { u, e1, e2 };
  }

  // Generator: x = center_k + shift + spread * STD * e, where the cluster k is chosen
  // uniformly among the active clusters with u. Without active clusters, no samples.
  function sample(params, { u, e1, e2 }) {
    const active = CENTERS.filter((_, k) => params.clusters[k]);
    if (!active.length) return [];
    const s = params.spread * STD;
    const out = new Array(u.length);
    for (let i = 0; i < u.length; i++) {
      const c = active[Math.min(Math.floor(u[i] * active.length), active.length - 1)];
      out[i] = [c[0] + params.shift[0] + s * e1[i], c[1] + params.shift[1] + s * e2[i]];
    }
    return out;
  }

  // Real features: the first n of a fixed set of samples (seed 2026).
  const realFeatures = (n, seed = 2026) => sample(REAL, noise(n, seed));

  // Mean and (unbiased) covariance [[a, b], [b, d]] of a set of points.
  function meanCov(points) {
    const n = points.length;
    let mx = 0;
    let my = 0;
    for (const [x, y] of points) { mx += x; my += y; }
    mx /= n;
    my /= n;
    let a = 0;
    let b = 0;
    let d = 0;
    for (const [x, y] of points) {
      a += (x - mx) ** 2;
      b += (x - mx) * (y - my);
      d += (y - my) ** 2;
    }
    const m = Math.max(n - 1, 1);
    return { mean: [mx, my], cov: [[a / m, b / m], [b / m, d / m]] };
  }

  // Frechet distance between N(mu1, S1) and N(mu2, S2), split into its two terms:
  //   mean term       |mu1 - mu2|^2
  //   covariance term tr(S1 + S2 - 2 (S1 S2)^(1/2))
  // For 2x2 matrices, the eigenvalues of S1 S2 are real and non-negative, so
  //   tr((S1 S2)^(1/2)) = sqrt(l1) + sqrt(l2) = sqrt(tr(S1 S2) + 2 sqrt(det(S1 S2))).
  function frechet(g1, g2) {
    const meanTerm = (g1.mean[0] - g2.mean[0]) ** 2 + (g1.mean[1] - g2.mean[1]) ** 2;
    const [[a1, b1], [, d1]] = g1.cov;
    const [[a2, b2], [, d2]] = g2.cov;
    const trProd = a1 * a2 + 2 * b1 * b2 + d1 * d2;
    const detProd = Math.max((a1 * d1 - b1 * b1) * (a2 * d2 - b2 * b2), 0);
    const trSqrt = Math.sqrt(Math.max(trProd + 2 * Math.sqrt(detProd), 0));
    const covTerm = Math.max(a1 + d1 + a2 + d2 - 2 * trSqrt, 0);
    return { fid: meanTerm + covTerm, meanTerm, covTerm };
  }

  const fid = (real, generated) => frechet(meanCov(real), meanCov(generated));

  // Radius of each point's ball: distance to its k-th nearest neighbour within the same set.
  function knnRadii(points, k = K) {
    const n = points.length;
    const m = Math.min(k, n - 1) + 1; // k neighbours plus the point itself (distance 0)
    const radii = new Float64Array(n);
    const best = new Float64Array(m); // the m smallest squared distances, ascending
    for (let i = 0; i < n; i++) {
      best.fill(Infinity);
      for (let j = 0; j < n; j++) {
        const d = (points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2;
        if (d >= best[m - 1]) continue;
        let p = m - 1;
        while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; p--; }
        best[p] = d;
      }
      radii[i] = Math.sqrt(best[m - 1]);
    }
    return radii;
  }

  // For each query point: does it lie in at least one ball of the reference set?
  function insideManifold(queries, reference, radii) {
    return queries.map(([x, y]) => {
      for (let j = 0; j < reference.length; j++) {
        if ((x - reference[j][0]) ** 2 + (y - reference[j][1]) ** 2 <= radii[j] ** 2) return true;
      }
      return false;
    });
  }

  // Precision: share of generated points inside the manifold of the real points (fidelity).
  // Recall: share of real points inside the manifold of the generated points (coverage).
  function precisionRecall(real, generated, k = K) {
    if (!real.length || generated.length <= k) {
      return { precision: 0, recall: 0, genInside: generated.map(() => false), realInside: real.map(() => false), realRadii: knnRadii(real, k), genRadii: new Float64Array(generated.length) };
    }
    const realRadii = knnRadii(real, k);
    const genRadii = knnRadii(generated, k);
    const genInside = insideManifold(generated, real, realRadii);
    const realInside = insideManifold(real, generated, genRadii);
    const share = (flags) => flags.filter(Boolean).length / flags.length;
    return { precision: share(genInside), recall: share(realInside), genInside, realInside, realRadii, genRadii };
  }

  // Ellipse of a fitted Gaussian at `scale` standard deviations: center, semi-axes, angle (radians).
  function ellipse(g, scale = 2) {
    const [[a, b], [, d]] = g.cov;
    const tr = a + d;
    const disc = Math.sqrt(Math.max(((a - d) / 2) ** 2 + b * b, 0));
    const l1 = tr / 2 + disc;
    const l2 = Math.max(tr / 2 - disc, 0);
    const angle = Math.abs(b) < 1e-12 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
    return { cx: g.mean[0], cy: g.mean[1], rx: scale * Math.sqrt(l1), ry: scale * Math.sqrt(l2), angle };
  }

  return {
    CENTERS, STD, REAL, X_MIN, X_MAX, Y_MIN, Y_MAX, K,
    rng, noise, sample, realFeatures, meanCov, frechet, fid, knnRadii, insideManifold, precisionRecall, ellipse,
  };
})();
