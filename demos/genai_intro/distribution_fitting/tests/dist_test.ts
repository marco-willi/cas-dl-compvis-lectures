// Checks for dist.js. Run from the repository root:
//   quarto run demos/genai_intro/distribution_fitting/tests/dist_test.ts
import "../dist.js";

const { Dist } = globalThis as any;
let failed = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const fmt = (x: number) => x.toFixed(3);

// Densities integrate to 1.
function integral(model: unknown) {
  let s = 0;
  for (let x = -20; x <= 30; x += 0.001) s += Dist.pdf(x, model) * 0.001;
  return s;
}
check("target density integrates to 1", close(integral(Dist.TARGET), 1, 1e-4), fmt(integral(Dist.TARGET)));
const narrow = { means: [7], stds: [0.2], weights: [1] };
check("narrow Gaussian integrates to 1", close(integral(narrow), 1, 1e-4));
check("a density can exceed 1", Dist.pdf(7, narrow) > 1, fmt(Dist.pdf(7, narrow)));

// The target cannot be changed.
check("target is frozen", Object.isFrozen(Dist.TARGET) && Object.isFrozen(Dist.TARGET.means));

// Same seed, same samples; different seed, different samples.
const a = Dist.sample(Dist.TARGET, Dist.noise(100, 1));
const b = Dist.sample(Dist.TARGET, Dist.noise(100, 1));
const c = Dist.sample(Dist.TARGET, Dist.noise(100, 2));
check("seed is deterministic", a.every((x: number, i: number) => x === b[i]));
check("different seeds differ", a.some((x: number, i: number) => x !== c[i]));

// Sample moments match the target: mean 0.4 * 2 + 0.6 * 7 = 5,
// variance 0.4 * (0.64 + 4) + 0.6 * (1 + 49) - 25 = 6.856.
const big = Dist.sample(Dist.TARGET, Dist.noise(200000, 3));
const moments = Dist.fitGaussian(big);
check("sample mean of target", close(moments.means[0], 5, 0.03), fmt(moments.means[0]));
check("sample std of target", close(moments.stds[0], Math.sqrt(6.856), 0.03), fmt(moments.stds[0]));

// Histogram is a density: total area 1 when all samples lie inside the range.
const hist = Dist.histogram(Dist.sample({ means: [5], stds: [1], weights: [1] }, Dist.noise(5000, 4)));
const area = hist.reduce((s: number, h: any) => s + h.density * (h.x1 - h.x0), 0);
check("histogram area is 1", close(area, 1, 1e-9), fmt(area));

// Maximum likelihood on training data from the target.
const train = Dist.trainingData();
check("training data has 1000 samples", train.length === 1000);
const single = Dist.fitGaussian(train);
check("ML single Gaussian sits between the modes", close(single.means[0], 5, 0.3), `mean ${fmt(single.means[0])}, std ${fmt(single.stds[0])}`);
const mixture = Dist.fitMixture(train);
check("EM recovers the left mode", close(mixture.means[0], 2, 0.2) && close(mixture.stds[0], 0.8, 0.15) && close(mixture.weights[0], 0.4, 0.05),
  `${fmt(mixture.means[0])} / ${fmt(mixture.stds[0])} / ${fmt(mixture.weights[0])}`);
check("EM recovers the right mode", close(mixture.means[1], 7, 0.2) && close(mixture.stds[1], 1.0, 0.15) && close(mixture.weights[1], 0.6, 0.05),
  `${fmt(mixture.means[1])} / ${fmt(mixture.stds[1])} / ${fmt(mixture.weights[1])}`);

// The fitted single Gaussian maximizes the likelihood among single Gaussians.
const llSingle = Dist.logLikelihood(train, single);
for (const [dm, ds] of [[0.3, 0], [-0.3, 0], [0, 0.3], [0, -0.3]]) {
  const other = { means: [single.means[0] + dm], stds: [single.stds[0] + ds], weights: [1] };
  check(`perturbed single Gaussian (${dm}, ${ds}) has lower likelihood`, Dist.logLikelihood(train, other) < llSingle);
}
const llMixture = Dist.logLikelihood(train, mixture);
const llTarget = Dist.logLikelihood(train, Dist.TARGET);
check("mixture fits better than a single Gaussian", llMixture > llSingle, `${fmt(llMixture)} > ${fmt(llSingle)}`);
check("mixture is close to the target likelihood", close(llMixture, llTarget, 0.02), `${fmt(llMixture)} vs. ${fmt(llTarget)}`);

// HDR threshold: the region above the threshold holds 95% of the mass.
const t = Dist.hdrThreshold(Dist.TARGET, 0.95);
let mass = 0;
for (let x = -15; x <= 25; x += 0.001) {
  const p = Dist.pdf(x, Dist.TARGET);
  if (p >= t) mass += p * 0.001;
}
check("95% region of the target holds 95% of the mass", close(mass, 0.95, 0.005), fmt(mass));

// Fidelity and coverage of the presets (10,000 generated samples).
const presets: Record<string, unknown> = {
  "mode collapse": { means: [7], stds: [0.6], weights: [1] },
  "ML single Gaussian": single,
  "too broad": { means: [4.5], stds: [4], weights: [1] },
  "good fit": mixture,
};
const scores: Record<string, { fidelity: number; coverage: number; ll: number }> = {};
for (const [name, model] of Object.entries(presets)) {
  const generated = Dist.sample(model, Dist.noise(10000, 5));
  scores[name] = { ...Dist.compare(generated, train, model), ll: Dist.logLikelihood(train, model) };
  console.log(`     ${name.padEnd(20)} fidelity ${fmt(scores[name].fidelity)}  coverage ${fmt(scores[name].coverage)}  log-lik ${fmt(scores[name].ll)}`);
}
check("mode collapse: high fidelity", scores["mode collapse"].fidelity > 0.9);
check("mode collapse: low coverage", scores["mode collapse"].coverage < 0.65);
check("ML single Gaussian: high coverage", scores["ML single Gaussian"].coverage > 0.9);
check("ML single Gaussian: lower fidelity than good fit", scores["ML single Gaussian"].fidelity < scores["good fit"].fidelity - 0.05);
check("too broad: lowest fidelity", scores["too broad"].fidelity < scores["ML single Gaussian"].fidelity);
check("good fit: both about 95%", close(scores["good fit"].fidelity, 0.95, 0.02) && close(scores["good fit"].coverage, 0.95, 0.03));
check("mode collapse has a much lower likelihood than the good fit", scores["mode collapse"].ll < scores["good fit"].ll - 1);

console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
if (failed) Deno.exit(1);
