// Checks for fid.js. Run from the repository root:
//   quarto run demos/genai_intro/fid_precision_recall/tests/fid_test.ts
import "../fid.js";

const { Fid } = globalThis as any;
let failed = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const fmt = (x: number) => x.toFixed(3);

// The real features cannot be changed.
check("real parameters are frozen", Object.isFrozen(Fid.REAL) && Object.isFrozen(Fid.CENTERS) && Object.isFrozen(Fid.CENTERS[0]));

// Same seed, same samples; different seed, different samples.
const a = Fid.sample(Fid.REAL, Fid.noise(100, 1));
const b = Fid.sample(Fid.REAL, Fid.noise(100, 1));
const c = Fid.sample(Fid.REAL, Fid.noise(100, 2));
check("seed is deterministic", a.every((p: number[], i: number) => p[0] === b[i][0] && p[1] === b[i][1]));
check("different seeds differ", a.some((p: number[], i: number) => p[0] !== c[i][0]));
check("no active cluster, no samples", Fid.sample({ shift: [0, 0], spread: 1, clusters: [false, false, false] }, Fid.noise(10, 1)).length === 0);

// Mean and covariance of a single cluster: center and STD^2 on the diagonal.
const one = Fid.sample({ shift: [0, 0], spread: 1, clusters: [false, true, false] }, Fid.noise(200000, 3));
const g = Fid.meanCov(one);
check("cluster mean", close(g.mean[0], 1.6, 0.01) && close(g.mean[1], -0.9, 0.01), `${fmt(g.mean[0])}, ${fmt(g.mean[1])}`);
check("cluster covariance", close(g.cov[0][0], 0.3025, 0.005) && close(g.cov[1][1], 0.3025, 0.005) && close(g.cov[0][1], 0, 0.005),
  `${fmt(g.cov[0][0])}, ${fmt(g.cov[0][1])}, ${fmt(g.cov[1][1])}`);

// Frechet distance in closed form.
const ga = { mean: [0, 0], cov: [[1, 0], [0, 4]] };
const gb = { mean: [3, 4], cov: [[4, 0], [0, 1]] };
// diagonal covariances: |mu|^2 + sum (sqrt(s_a) - sqrt(s_b))^2 = 25 + (1 - 2)^2 + (2 - 1)^2 = 27
const fd = Fid.frechet(ga, gb);
check("Frechet distance of diagonal Gaussians", close(fd.fid, 27, 1e-9) && close(fd.meanTerm, 25, 1e-9) && close(fd.covTerm, 2, 1e-9), fmt(fd.fid));
check("Frechet distance is symmetric", close(Fid.frechet(gb, ga).fid, fd.fid, 1e-9));
const gr = { mean: [1, 2], cov: [[2, 0.8], [0.8, 1]] };
check("Frechet distance to itself is 0", close(Fid.frechet(gr, gr).fid, 0, 1e-9), fmt(Fid.frechet(gr, gr).fid));
// rotated covariance: equal spectra, different orientation; reference value computed with scipy.linalg.sqrtm
const rot = { mean: [1, 2], cov: [[1, 0.8], [0.8, 2]] };
check("Frechet distance of rotated covariances", close(Fid.frechet(gr, rot).covTerm, 0.3431, 0.0005), fmt(Fid.frechet(gr, rot).covTerm));

// k-NN radii: on a regular line with spacing 1, the 3rd neighbour is at distance 2 (inner points).
const line = Array.from({ length: 11 }, (_, i) => [i, 0]);
const radii = Fid.knnRadii(line, 3);
check("k-NN radius on a line", radii[5] === 2 && radii[0] === 3, `${radii[5]}, ${radii[0]}`);

// Presets (300 points per set): which metric reacts to which failure.
const real = Fid.realFeatures(300);
function evaluate(params: unknown) {
  const gen = Fid.sample(params, Fid.noise(300, 1));
  return { ...Fid.fid(real, gen), ...Fid.precisionRecall(real, gen) };
}
const good = evaluate({ shift: [0, 0], spread: 1, clusters: [true, true, true] });
const shift = evaluate({ shift: [1.5, 1.1], spread: 1, clusters: [true, true, true] });
const collapse = evaluate({ shift: [0, 0], spread: 1, clusters: [false, true, false] });
const broad = evaluate({ shift: [0, 0], spread: 2, clusters: [true, true, true] });
const narrow = evaluate({ shift: [0, 0], spread: 0.4, clusters: [true, true, true] });
const show = (r: any) => `FID ${fmt(r.fid)}, P ${fmt(r.precision)}, R ${fmt(r.recall)}`;
check("good: low FID, high precision and recall", good.fid < 0.1 && good.precision > 0.9 && good.recall > 0.9, show(good));
check("shift: high FID from the mean term, low precision", shift.fid > 2 && shift.meanTerm > 0.95 * shift.fid && shift.precision < 0.6, show(shift));
check("collapse: high FID, high precision, low recall", collapse.fid > 2 && collapse.precision > 0.9 && collapse.recall < 0.5, show(collapse));
check("broad: lower precision, high recall", broad.precision < 0.85 && broad.recall > 0.95, show(broad));
check("narrow: FID barely changes, recall drops", narrow.fid < 0.1 && narrow.precision > 0.9 && narrow.recall < 0.7, show(narrow));
check("shift and collapse have a similar FID", close(shift.fid, collapse.fid, 1.5), `${fmt(shift.fid)} vs. ${fmt(collapse.fid)}`);

// Small samples: FID of a perfect generator is biased upwards.
const fidSmall = Fid.fid(Fid.realFeatures(50), Fid.sample(Fid.REAL, Fid.noise(50, 1))).fid;
const fidLarge = Fid.fid(Fid.realFeatures(1000), Fid.sample(Fid.REAL, Fid.noise(1000, 1))).fid;
check("FID of a perfect generator decreases with sample size", fidSmall > fidLarge, `${fmt(fidSmall)} > ${fmt(fidLarge)}`);

// Ellipse of an axis-aligned Gaussian.
const e = Fid.ellipse({ mean: [0, 0], cov: [[4, 0], [0, 1]] }, 2);
check("ellipse semi-axes and angle", close(e.rx, 4, 1e-9) && close(e.ry, 2, 1e-9) && close(e.angle, 0, 1e-9));
const e2 = Fid.ellipse({ mean: [0, 0], cov: [[1, 0], [0, 4]] }, 2);
check("ellipse of a vertical Gaussian", close(e2.rx, 4, 1e-9) && close(e2.angle, Math.PI / 2, 1e-9));

console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
if (failed) Deno.exit(1);
