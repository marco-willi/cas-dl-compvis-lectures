// Checks for toy.js. Run from the repository root:
//   quarto run demos/genai_models/generative_paths/tests/toy_test.ts
import "../toy.js";
import "../../../genai_intro/distribution_fitting/dist.js";
import "../data/models.js";

const { Toy, Dist, TRAINED_MODELS } = globalThis as any;
let failed = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
}
const fmt = (x: number) => (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4));
const last = (states: Float32Array, cols: number, n: number) => states.subarray((cols - 1) * n, cols * n);
const column = (states: Float32Array, c: number, n: number) => states.subarray(c * n, (c + 1) * n);

// ---- target and special functions ----------------------------------------------------------

check(
  "target equals the intro lecture's target",
  JSON.stringify([Toy.TARGET.weights, Toy.TARGET.means, Toy.TARGET.stds]) ===
    JSON.stringify([Dist.TARGET.weights, Dist.TARGET.means, Dist.TARGET.stds]),
);
const erfcRef: [number, number][] = [
  [0, 1], [0.5, 0.4795001221869535], [1, 0.15729920705028513], [2.9, 4.1097878099458e-05],
  [3, 2.209049699858544e-05], [5, 1.5374597944280349e-12], [-1, 1.8427007929497148],
];
const erfcErr = Math.max(...erfcRef.map(([x, y]) => Math.abs(Toy.erfc(x) - y) / y));
check("erfc relative error < 1e-13", erfcErr < 1e-13, fmt(erfcErr));
let qErr = 0;
for (const p of [1e-9, 1e-4, 0.1, 0.4, 0.5, 0.9]) {
  qErr = Math.max(qErr, Math.abs(Toy.cdf(Toy.quantile(p, Toy.TARGET), Toy.TARGET) - p) / p);
  qErr = Math.max(qErr, Math.abs(Toy.sf(Toy.quantile(p, Toy.TARGET, false), Toy.TARGET) - p) / p);
}
check("quantile inverts cdf and sf (relative error < 1e-9)", qErr < 1e-9, fmt(qErr));

// ---- shared acceptance test: endpoint distribution vs. target --------------------------------

const N = 20000;
const z = Toy.initialNoise(N, 7);
function endpoint(name: string, x: Float32Array | Float64Array) {
  const w1 = Toy.wasserstein1(x, Toy.TARGET);
  const ks = Toy.ks(x, Toy.TARGET);
  check(`${name}: W1 < 0.05 and KS < 0.02`, w1 < 0.05 && ks < 0.02, `W1 ${fmt(w1)}, KS ${fmt(ks)}`);
}
// The test itself: samples drawn directly from the target pass, a shifted copy does not.
const direct = Toy.sampleMixture(Toy.TARGET, N, Toy.rng(11));
endpoint("direct samples from the target", direct);
const shifted = direct.map((v: number) => v + 0.2);
check("shifted samples fail the endpoint test", Toy.wasserstein1(shifted, Toy.TARGET) > 0.15);
const collapsed = Toy.sampleMixture({ weights: [1], means: [7], stds: [1] }, N, Toy.rng(12));
check("mode-collapsed samples fail the endpoint test", Toy.wasserstein1(collapsed, Toy.TARGET) > 1);

// ---- normalizing flow ----------------------------------------------------------------------

const K = 4;
const nf = Toy.nfStates(z, K);
endpoint("normalizing flow (exact layers)", last(nf, K + 1, N));
let orderOk = true;
const order = Array.from(z.keys()).sort((a, b) => z[a] - z[b]);
for (let k = 0; k <= K; k++) {
  const h = column(nf, k, N);
  for (let j = 1; j < N; j++) if (h[order[j]] < h[order[j - 1]]) orderOk = false;
}
check("NF: sample order preserved at every layer boundary", orderOk);
let roundTrip = 0;
let fdErr = 0;
for (let h = -4; h <= 10; h += 0.25) {
  for (let k = 1; k <= K; k++) {
    roundTrip = Math.max(roundTrip, Math.abs(Toy.nfInverse(Toy.nfForward(h, k, K), k, K) - h));
    const e = 1e-5;
    const fd = (Toy.nfForward(h + e, k, K) - Toy.nfForward(h - e, k, K)) / (2 * e);
    fdErr = Math.max(fdErr, Math.abs(Math.log(fd) - Toy.nfLogDet(h, k, K)));
  }
}
check("NF: inverse(forward(h)) = h to 1e-8", roundTrip < 1e-8, fmt(roundTrip));
check("NF: log|f'| matches finite differences", fdErr < 1e-5, fmt(fdErr));
let cov = 0;
for (const zz of [-2.5, -1, 0, 0.7, 2]) {
  let h = zz;
  let logDet = 0;
  for (let k = 1; k <= K; k++) {
    logDet += Toy.nfLogDet(h, k, K);
    h = Toy.nfForward(h, k, K);
  }
  cov = Math.max(cov, Math.abs(Toy.logPdf(zz, Toy.NOISE) - logDet - Toy.logPdf(h, Toy.TARGET)));
}
check("NF: log p(x) = log N(z) - sum log|f'| (change of variables)", cov < 1e-8, fmt(cov));

// ---- flow matching -------------------------------------------------------------------------

endpoint("flow matching (RK4, 200 steps)", last(Toy.fmStates(z, 200, "rk4"), 201, N));
// Analytic velocity = average of x1 - x0 over random pairs near (x, t) (Monte Carlo), using the
// same function as the page's probe.
{
  const pairs = Toy.fmPairs(400000, 21);
  let worst = 0;
  let minCount = Infinity;
  for (const [x, t] of [[1, 0.3], [4.5, 0.6], [6, 0.8], [3.5, 0.5]]) {
    const avg = Toy.fmPairAverage(pairs, x, t, 0.05);
    // Monte Carlo standard error of the mean, from the spread of the directions.
    const sd = Math.sqrt(avg.directions.reduce((a: number, d: number) => a + (d - avg.mean) ** 2, 0) / avg.count);
    worst = Math.max(worst, Math.abs(avg.mean - Toy.fmVelocity(x, t)) / (sd / Math.sqrt(avg.count) + 0.02));
    minCount = Math.min(minCount, avg.count);
  }
  check("FM: mean direction of nearby training pairs = u(x, t) (within 4 standard errors)", worst < 4, `${fmt(worst)} SE, >= ${minCount} pairs`);
}
{
  const zs = z.subarray(0, 2000);
  const ref = last(Toy.fmStates(zs, 400, "rk4"), 401, 2000);
  const errs = [1, 4, 16, 64].map((s) => {
    const x = last(Toy.fmStates(zs, s, "euler"), s + 1, 2000);
    let e = 0;
    for (let i = 0; i < 2000; i++) e += Math.abs(x[i] - ref[i]);
    return e / 2000;
  });
  check("FM: Euler error to RK4 reference decreases with steps", errs.every((e, i) => i === 0 || e < errs[i - 1] * 0.7), errs.map(fmt).join(" > "));
  check("FM: 64 Euler steps are close to the reference", errs[3] < 0.05, fmt(errs[3]));
}
{
  // In this setup, one Euler step moves every sample to u(z, 0) + z = mean of the target (5),
  // i.e. into the low-density gap between the modes.
  const gap = (x: ArrayLike<number>) => Array.from(x).filter((v) => v > 3.5 && v < 5.5).length / x.length;
  const one = last(Toy.fmStates(z, 1, "euler"), 2, N);
  const targetGap = Toy.cdf(5.5, Toy.TARGET) - Toy.cdf(3.5, Toy.TARGET);
  check("FM: one Euler step puts the samples in the gap between the modes", gap(one) > 0.9 && targetGap < 0.1, `${fmt(gap(one))} vs. target ${fmt(targetGap)}`);
}
check("FM: marginal at t = 0 is N(0, 1), at t = 1 the target",
  JSON.stringify(Toy.fmMarginal(0).stds) === "[1,1]" && JSON.stringify(Toy.fmMarginal(1).means) === "[2,7]");

// ---- diffusion -----------------------------------------------------------------------------

const ab1 = Toy.alphaBar(1);
check("diffusion: alphaBar(0) = 1 and alphaBar(T) ~ 0", Toy.alphaBar(0) === 1 && ab1 < 1e-4, fmt(ab1));
const ddpm = Toy.diffReverseStates(z, 1000, "ddpm", 3);
const ddim = Toy.diffReverseStates(z, 1000, "ddim", 3);
const ode = Toy.diffReverseStates(z, 1000, "ode", 3);
endpoint("diffusion: stochastic sampler (DDPM, 1000 steps)", last(ddpm, 1001, N));
endpoint("diffusion: DDIM (1000 steps)", last(ddim, 1001, N));
endpoint("diffusion: probability-flow ODE (Heun, 1000 steps)", last(ode, 1001, N));
{
  // DDIM and the probability-flow ODE discretize the same ODE of the same schedule: with many
  // steps their deterministic endpoints agree sample by sample.
  const a = last(ddim, 1001, N);
  const b = last(ode, 1001, N);
  let d = 0;
  for (let i = 0; i < N; i++) d += Math.abs(a[i] - b[i]);
  check("diffusion: DDIM and probability-flow ODE agree per sample", d / N < 0.02, fmt(d / N));
}
{
  // Separate random streams: the sampler seed changes the stochastic endpoints, not x_T.
  const zs = z.subarray(0, 500);
  const r1 = Toy.diffReverseStates(zs, 100, "ddpm", 1);
  const r2 = Toy.diffReverseStates(zs, 100, "ddpm", 2);
  const d1 = Toy.diffReverseStates(zs, 100, "ddim", 1);
  const d2 = Toy.diffReverseStates(zs, 100, "ddim", 2);
  const same = (a: Float32Array, b: Float32Array) => a.every((v, i) => v === b[i]);
  check("diffusion: same x_T for both sampler seeds", same(column(r1, 0, 500), column(r2, 0, 500)));
  check("diffusion: stochastic sampler differs between sampler seeds", !same(last(r1, 101, 500), last(r2, 101, 500)));
  check("diffusion: deterministic samplers ignore the sampler seed", same(d1, d2));
  const n1 = Toy.initialNoise(500, 1);
  const n2 = Toy.initialNoise(500, 2);
  check("initial noise depends on its seed only", !n1.every((v: number, i: number) => v === n2[i]));
}
{
  const shift = Math.max(...[-2, 0, 2].map((v) => Math.abs(Toy.diffPrior(v) - v - Math.sqrt(ab1) * 5)));
  check("diffusion: x_T = z + 0.03 (exact p_T)", shift < 1e-3, fmt(shift));
}
{
  // In 1-D the order-preserving map from N(0, 1) to the target is unique (quantile transport).
  // NF, the flow-matching ODE and the diffusion probability-flow ODE all realize it: same z,
  // same endpoint x. Only the path differs.
  const a = last(nf, K + 1, N);
  const b = last(Toy.fmStates(z, 200, "rk4"), 201, N);
  const c = last(ode, 1001, N);
  let dab = 0;
  let dac = 0;
  for (let i = 0; i < N; i++) {
    dab += Math.abs(a[i] - b[i]);
    dac += Math.abs(a[i] - c[i]);
  }
  check("deterministic endpoints agree per sample: NF = FM ODE = diffusion ODE", dab / N < 0.01 && dac / N < 0.02, `FM ${fmt(dab / N)}, diffusion ${fmt(dac / N)}`);
}
{
  // Lecture diagnostics: the DDIM error against the accurate reference (ODE, 1000 steps, same start)
  // shrinks with the number of steps.
  const zs = z.subarray(0, 2000);
  const ref = last(Toy.diffReverseStates(zs, 1000, "ode", 1), 1001, 2000);
  const errs = [2, 10, 100].map((st) => Toy.referenceError(last(Toy.diffReverseStates(zs, st, "ddim", 1), st + 1, 2000), ref));
  check("diffusion: DDIM error vs. reference decreases with steps", errs[0] > errs[1] && errs[1] > errs[2] && errs[2] < 0.05, errs.map(fmt).join(" > "));
}
{
  // Forward noising reaches the exact marginals.
  const x0 = Toy.sampleMixture(Toy.TARGET, N, Toy.rng(31));
  const fwd = Toy.diffForwardStates(x0, 50, 4);
  let worst = 0;
  for (const c of [0, 25, 40]) worst = Math.max(worst, Toy.wasserstein1(column(fwd, c, N), Toy.diffMarginal(Toy.diffTime(c, 50))));
  check("diffusion: forward noising matches the exact marginals p_t", worst < 0.05, fmt(worst));
}
{
  const one = last(Toy.diffReverseStates(z, 1, "ddim", 1), 2, N);
  const mean = one.reduce((a: number, b: number) => a + b, 0) / N;
  check("diffusion: one DDIM step returns the denoiser's mean guess (~5)", Math.abs(mean - 5) < 0.1, fmt(mean));
}

// ---- normalizing flow: jobs and density evaluation -------------------------------------------

{
  const jobs3 = [1, 2, 3].map((k) => Toy.nfLayerJobs(k, 3));
  check("NF: with K = 3, every layer has exactly one job, in order",
    jobs3.every((j: string[], k: number) => j.length === 1 && j[0] === Toy.NF_JOBS[k]), JSON.stringify(jobs3));
  check("NF: with K = 1, the single layer does all three jobs", Toy.nfLayerJobs(1, 1).length === 3);
  check("NF: p_0 = N(0, 1) and p_K = target",
    JSON.stringify(Toy.nfLayerDist(0, 3).means) === "[0,0]" && JSON.stringify(Toy.nfLayerDist(0, 3).stds) === "[1,1]" &&
    Toy.nfLayerDist(3, 3).means.every((m: number, j: number) => Math.abs(m - Toy.TARGET.means[j]) < 1e-12));
  let worst = 0;
  for (const x of [-0.5, 2, 4.5, 7, 9.5]) {
    for (const K of [1, 3, 6]) {
      const e = Toy.nfEvaluate(x, K);
      worst = Math.max(worst, Math.abs(e.logp - Toy.logPdf(x, Toy.TARGET)));
      let h = e.z;
      for (let k = 1; k <= K; k++) h = Toy.nfForward(h, k, K);
      worst = Math.max(worst, Math.abs(h - x));
    }
  }
  check("NF: density evaluation x → z gives the exact log p_data(x)", worst < 1e-8, fmt(worst));
}

// ---- autoregressive ------------------------------------------------------------------------

{
  let sumErr = 0;
  for (const tau of [0.5, 1, 2]) {
    const t = Toy.arTables(tau);
    sumErr = Math.max(sumErr, Math.abs(t.pc.reduce((a: number, b: number) => a + b, 0) - 1));
    for (const row of t.pf) sumErr = Math.max(sumErr, Math.abs(row.reduce((a: number, b: number) => a + b, 0) - 1));
  }
  check("AR: p(c) and every p(f | c) sum to 1 (all temperatures)", sumErr < 1e-12, fmt(sumErr));
  const tables = Toy.arTables(1);
  let jointErr = 0;
  for (let c = 0; c < Toy.AR.coarse; c++) {
    for (let f = 0; f < Toy.AR.fine; f++) {
      const exact = Toy.cdf(Toy.arEdge(c, f + 1), Toy.TARGET) - Toy.cdf(Toy.arEdge(c, f), Toy.TARGET);
      if (c > 0 && c < Toy.AR.coarse - 1) jointErr = Math.max(jointErr, Math.abs(tables.pc[c] * tables.pf[c][f] - exact));
    }
  }
  check("AR: p(c) p(f | c) = probability of the bin", jointErr < 1e-12, fmt(jointErr));
  const s = Toy.arSample(N, 5, 1);
  const w1 = Toy.wasserstein1(s.x, Toy.TARGET);
  const ks = Toy.ks(s.x, Toy.TARGET);
  check("AR: samples match the target up to quantisation (W1 < 0.06, KS < 0.03)", w1 < 0.06 && ks < 0.03, `W1 ${fmt(w1)}, KS ${fmt(ks)}`);
  check("AR: values are bin centres", s.x.every((v: number, i: number) => Math.abs(v - Toy.arValue(s.c[i], s.f[i])) < 1e-12));
  const entropy = (p: Float64Array) => -p.reduce((a: number, v: number) => a + (v > 0 ? v * Math.log(v) : 0), 0);
  const h = [0.5, 1, 2].map((tau) => entropy(Toy.arTables(tau).pc));
  check("AR: lower temperature sharpens, higher flattens p(c)", h[0] < h[1] && h[1] < h[2], h.map(fmt).join(" < "));
  // Lower temperature narrows each mode (here by about 1/sqrt(2) at tau = 0.5); it does not
  // remove the smaller mode, because that mode is also the narrower one.
  const spread = (x: Float64Array) => {
    const r = Array.from(x).filter((v) => v > 4.5);
    const m = r.reduce((a, b) => a + b, 0) / r.length;
    return Math.sqrt(r.reduce((a, b) => a + (b - m) ** 2, 0) / r.length);
  };
  const cold = spread(Toy.arSample(N, 5, 0.5).x);
  const normal = spread(s.x);
  check("AR: temperature 0.5 narrows the modes by about 1/sqrt(2)", Math.abs(cold / normal - Math.SQRT1_2) < 0.05, `${fmt(cold)} vs. ${fmt(normal)}`);
}
{
  const s = Toy.ar2dSample(N, 6);
  const w1 = Toy.wasserstein1(s.x1, Toy.TARGET);
  const ks = Toy.ks(s.x1, Toy.TARGET);
  check("AR 2-D: x1 marginal of ancestral samples matches the target", w1 < 0.05 && ks < 0.02, `W1 ${fmt(w1)}, KS ${fmt(ks)}`);
  let worst = 0;
  for (let x1 = -1; x1 <= 10; x1 += 0.5) {
    let integral = 0;
    for (let x2 = -15; x2 <= 30; x2 += 0.002) integral += Toy.pdf(x2, Toy.ar2dConditional(x1)) * 0.002;
    worst = Math.max(worst, Math.abs(integral - 1));
  }
  check("AR 2-D: p(x2 | x1) integrates to 1 for every x1", worst < 1e-6, fmt(worst));
  // The conditional depends on x1: samples near the two modes of x1 have different x2 spreads.
  const sel = (lo: number, hi: number) => Array.from(s.x2).filter((_, i) => s.x1[i] > lo && s.x1[i] < hi);
  const sd = (a: number[]) => Math.sqrt(a.reduce((u, v) => u + v * v, 0) / a.length - (a.reduce((u, v) => u + v, 0) / a.length) ** 2);
  const sdLeft = sd(sel(1.8, 2.2));
  const sdRight = sd(sel(6.8, 7.2));
  check("AR 2-D: x2 | x1 differs between the modes (spread)", Math.abs(sdLeft - Toy.AR2D.std(2)) < 0.08 && Math.abs(sdRight - Toy.AR2D.std(7)) < 0.08 && sdRight > sdLeft + 0.2,
    `${fmt(sdLeft)} vs. ${fmt(sdRight)}`);
}

// ---- autoregressive sequence ---------------------------------------------------------------

{
  const n = Toy.SEQ.n;
  // The last value follows the target exactly (by construction) and in the samples.
  const lastMix = Toy.seqMarginal(n - 1);
  check("AR sequence: last position's distribution is the target",
    lastMix.means.every((m: number, j: number) => Math.abs(m - Toy.TARGET.means[j]) < 1e-12) &&
    lastMix.stds.every((s: number, j: number) => Math.abs(s - Toy.TARGET.stds[j]) < 1e-12));
  const seqs = Toy.seqSample(6000, 3);
  const last = Float64Array.from(seqs, (x: Float64Array) => x[n - 1]);
  const w1 = Toy.wasserstein1(last, Toy.TARGET);
  check("AR sequence: last values of ancestral samples match the target", w1 < 0.06, `W1 ${fmt(w1)}`);
  // Each position's samples follow its marginal (ancestral sampling through the conditionals
  // reproduces the joint distribution).
  let worst = 0;
  for (let k = 0; k < n; k++) worst = Math.max(worst, Toy.wasserstein1(Float64Array.from(seqs, (x: Float64Array) => x[k]), Toy.seqMarginal(k)));
  check("AR sequence: every position matches its marginal", worst < 0.06, fmt(worst));
  // The conditional is the ratio of joint densities: p(x_k | prefix) = p(prefix, x_k) / p(prefix).
  let ratioErr = 0;
  for (const s of seqs.slice(0, 5)) {
    for (let k = 1; k < n; k++) {
      const prefix = Array.from(s).slice(0, k);
      const c = Toy.seqConditional(prefix);
      for (const x of [2, 4.5, 7]) {
        const viaRatio = Math.exp(Toy.seqLogPdf([...prefix, x]) - Toy.seqLogPdf(prefix));
        ratioErr = Math.max(ratioErr, Math.abs(Toy.pdf(x, c) - viaRatio) / Math.max(viaRatio, 1e-12));
      }
    }
  }
  check("AR sequence: conditional = p(prefix, x) / p(prefix)", ratioErr < 1e-8, fmt(ratioErr));
  // Teaching claims: the model gets surer of the direction with every value, and later
  // positions are bimodal on their own but not given the values so far.
  const certainty = [1, 2, 3, 4, 5].map((k) => seqs.slice(0, 1500).reduce((a: number, s: Float64Array) => a + Math.max(...Toy.seqPosterior(Array.from(s).slice(0, k))), 0) / 1500);
  check("AR sequence: certainty about the direction grows over several values",
    certainty.every((c: number, i: number) => i === 0 || c > certainty[i - 1]) && certainty[1] < 0.8 && certainty[4] > 0.9, certainty.map(fmt).join(" < "));
  const modes = (mix: any) => {
    let c = 0, prev = 0, up = true;
    for (let x = -3; x <= 12; x += 0.01) { const p = Toy.pdf(x, mix); if (up && p < prev) { c++; up = false; } else if (!up && p > prev) up = true; prev = p; }
    return c;
  };
  const late = seqs.slice(0, 300).filter((s: Float64Array) => modes(Toy.seqConditional(Array.from(s).slice(0, 5))) === 1).length / 300;
  check("AR sequence: position 6 is bimodal alone, but mostly unimodal given x1..x5", modes(Toy.seqMarginal(5)) === 2 && late > 0.85, `unimodal share ${fmt(late)}`);
  check("AR sequence: x1 has a single peak", modes(Toy.seqConditional([])) === 1);
}

// ---- cross-check with the independent Python implementation (tests/fixtures.json) -------------

{
  const fx = JSON.parse(Deno.readTextFileSync(new URL("./fixtures.json", import.meta.url)));
  let worst = 0;
  const cmp = (a: number, b: number) => { worst = Math.max(worst, Math.abs(a - b) / Math.max(1, Math.abs(b))); };
  for (const [t, v] of fx.alphaBar) cmp(Toy.alphaBar(t), v);
  for (const [x, t, v] of fx.fmVelocity) cmp(Toy.fmVelocity(x, t), v);
  for (const [x, t, v] of fx.diffScore) cmp(Toy.diffScore(x, t), v);
  for (const [x, t, v] of fx.diffDenoise) cmp(Toy.diffDenoise(x, t), v);
  for (const [h, k, K, v] of fx.nfForward) cmp(Toy.nfForward(h, k, K), v);
  for (const [x, v] of fx.targetCdf) cmp(Toy.cdf(x, Toy.TARGET), v);
  for (const tau of [0.5, 1, 2]) {
    const ref = fx[`arTables_${tau.toFixed(1)}`];
    const t = Toy.arTables(tau);
    ref.pc.forEach((v: number, i: number) => cmp(t.pc[i], v));
    ref.pf3.forEach((v: number, i: number) => cmp(t.pf[3][i], v));
    ref.pf8.forEach((v: number, i: number) => cmp(t.pf[8][i], v));
  }
  check("toy.js matches the Python reference values (tests/fixtures.json) to 1e-6", worst < 1e-6, fmt(worst));
}

// ---- trained models (data/models.js) -------------------------------------------------------

{
  const m = TRAINED_MODELS;
  const runs = Object.keys(m.gan);
  const okShape = runs.length === 2 && runs.every((r) => m.gan[r].snapshots.every((s: any) => s.g.length === m.zGrid.n && s.d.length === m.xGrid.n)) &&
    m.vae.decMean.length === m.zGrid.n && m.vae.encStd.length === m.xGrid.n && m.vae.decStd.every((v: number) => v > 0);
  check("trained models: lookup tables have the grid sizes", okShape, `${runs.join(", ")}; ${m.gan.standard.snapshots.length} snapshots`);
  // Sampling from the interpolated tables in JS reproduces the W1 measured in Python.
  const zz = Toy.initialNoise(20000, 9);
  const e = Toy.normals(20000, Toy.rng(10));
  let worst = 0;
  for (const run of runs) {
    for (const s of m.gan[run].snapshots) worst = Math.max(worst, Math.abs(Toy.wasserstein1(Toy.ganGenerate(m, s, zz), Toy.TARGET) - s.w1));
  }
  const vaeW1 = Toy.wasserstein1(Toy.vaeGenerate(m, zz, e, true), Toy.TARGET);
  const vaeMean = Toy.wasserstein1(Toy.vaeGenerate(m, zz, e, false), Toy.TARGET);
  worst = Math.max(worst, Math.abs(vaeW1 - m.vae.w1), Math.abs(vaeMean - m.vae.w1MeanOnly));
  check("trained models: JS sampling reproduces the Python W1 (within 0.05)", worst < 0.05, fmt(worst));
  const fin = m.gan.standard.snapshots.at(-1);
  const leftShare = Array.from(Toy.ganGenerate(m, fin, zz)).filter((v) => v < 4.5).length / zz.length;
  check("trained GAN (standard, final): both modes with about the right weights", Math.abs(leftShare - 0.4) < 0.06, `left mode ${fmt(leftShare)} vs. 0.4`);
}

// ---- views ---------------------------------------------------------------------------------

for (const cfg of [
  { family: "nf", layers: 4 },
  { family: "fm", mode: "flow", steps: 16 },
  { family: "fm", mode: "pairs" },
  { family: "diffusion", mode: "reverse", sampler: "ddpm", steps: 50 },
  { family: "diffusion", mode: "forward", steps: 50 },
]) {
  const v = Toy.simulate({ seed: 1, samplerSeed: 1, n: 300, steps: 10, layers: 4, sampler: "ode", mode: "flow", ...cfg });
  const ok = v.states.length === v.cols * v.n && v.states.every(Number.isFinite) && v.pos(0) === 0 && v.pos(v.cols - 1) === 1;
  check(`view ${JSON.stringify(cfg)} is well-formed`, ok);
}

console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
if (failed) Deno.exit(1);
