// Checks for sampling.js. Run from the repository root:
//   quarto run demos/genai_models/image_sampling/tests/sampling_test.ts
import "../sampling.js";

const { Sampling: S } = globalThis as any;
let failed = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
}
const mean = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v, 0) / a.length;
const std = (a: ArrayLike<number>) => {
  const m = mean(a);
  return Math.sqrt(mean(Array.from(a, (v) => (v - m) ** 2)));
};

// ---- random numbers --------------------------------------------------------------------

const z = S.normals(1, 100001);
check("normals: mean ~ 0", Math.abs(mean(z)) < 0.01, mean(z).toFixed(4));
check("normals: std ~ 1", Math.abs(std(z) - 1) < 0.01, std(z).toFixed(4));
check("normals: same seed, same numbers", S.normals(5, 10).join() === S.normals(5, 10).join());

// ---- conversions -----------------------------------------------------------------------

const rgba = new Uint8ClampedArray([0, 128, 255, 255, 10, 20, 30, 255]);
check("RGBA round trip", S.toRGBA(S.fromRGBA(rgba)).join() === rgba.join());

// ---- diffusion -------------------------------------------------------------------------

const T = S.T_DIFFUSION;
check("alpha_bar(0) = 1", S.alphaBar(0) === 1);
check("alpha_bar(T) ~ 0", S.alphaBar(T) < 1e-3, S.alphaBar(T).toExponential(2));
check("alpha_bar decreasing", Array.from({ length: T }, (_, t) => S.alphaBar(t + 1) < S.alphaBar(t)).every(Boolean));

const n = 3 * 64 * 36;
const x0 = new Float32Array(n).fill(0.5);
const eps = S.normals(3, n);
check("forward: x_0 is the data", S.diffusionForward(x0, eps, 0).every((v: number, i: number) => v === x0[i]));
const states = S.diffusionReverse(x0, 3);
check("reverse: T + 1 states", states.length === T + 1);
check("reverse: x_T is the noise", states[0].every((v: number, i: number) => Math.abs(v - (Math.sqrt(S.alphaBar(T)) * 0.5 + Math.sqrt(1 - S.alphaBar(T)) * eps[i])) < 1e-6));
check("reverse: x_0 is the data", states[T].every((v: number, i: number) => Math.abs(v - x0[i]) < 1e-6));
const resid = Array.from(states[T / 2], (v: number) => v - Math.sqrt(S.alphaBar(T / 2)) * 0.5);
const expected = Math.sqrt(1 - S.alphaBar(T / 2));
check("reverse: noise level of step t", Math.abs(std(resid) - expected) < 0.02 * expected, `${std(resid).toFixed(3)} vs ${expected.toFixed(3)}`);
const rerun = S.diffusionReverse(x0, 3, 99);
check("reverse: same x_T for the same seed", rerun[0].every((v: number, i: number) => v === states[0][i]));
check("reverse: fresh noise changes the path", rerun[T / 2].some((v: number, i: number) => v !== states[T / 2][i]));

// ---- flow matching ---------------------------------------------------------------------

check("flow: t = 0 is the noise", S.flow(x0, eps, 0).every((v: number, i: number) => Math.abs(v - eps[i]) < 1e-6));
check("flow: t = 1 is the data", S.flow(x0, eps, 1).every((v: number, i: number) => Math.abs(v - x0[i]) < 1e-6));

// ---- autoregressive tokens -------------------------------------------------------------

// 2 x 1 tokens of 2 x 2 pixels; codebook: black, white, grey
const width = 4, patch = 2;
const img = S.fromRGBA(new Uint8ClampedArray(Array.from({ length: 8 }, (_, i) => (i % 4 < 2 ? [0, 0, 0, 255] : [255, 255, 255, 255])).flat()));
const codebook = new Uint8Array([...Array(12).fill(0), ...Array(12).fill(255), ...Array(12).fill(128)]);
const left = S.tokenDistribution(img, width, codebook, patch, 0);
const right = S.tokenDistribution(img, width, codebook, patch, 1);
check("tokens: probabilities sum to 1", Math.abs(left.probs.reduce((a: number, b: number) => a + b, 0) - 1) < 1e-12);
check("tokens: most likely token matches the patch", left.token === 0 && right.token === 1, `${left.token}, ${right.token}`);

// ---- GAN and VAE -----------------------------------------------------------------------

const enc = S.encode(2);
check("encoder: Z_DIM means and positive sigmas", enc.mu.length === S.Z_DIM && enc.sigma.every((s: number) => s > 0));
const zs = Array.from({ length: 2000 }, (_, k) => S.sampleLatent(enc.mu, enc.sigma, k)[0]);
check("reparameterization: mean ~ mu", Math.abs(mean(zs) - enc.mu[0]) < 0.03, `${mean(zs).toFixed(3)} vs ${enc.mu[0].toFixed(3)}`);

console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
if (failed) Deno.exit(1);
