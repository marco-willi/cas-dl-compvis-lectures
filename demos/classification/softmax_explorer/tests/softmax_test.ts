// Checks for softmax.js and config.js. Run from the repository root:
//   quarto run demos/classification/softmax_explorer/tests/softmax_test.ts
import "../softmax.js";
import "../config.js";

const { Softmax: S, SoftmaxDemoConfig: CFG } = globalThis as any;
let failed = 0;
function check(name: string, actual: unknown, expected: unknown) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${ok ? "" : `: got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}`}`);
}
const near = (a: number[], b: number[], tol = 1e-12) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < tol);

// --- Softmax -------------------------------------------------------------------------
// [0, ln 2, ln 3] -> e^z = [1, 2, 3] -> [1/6, 2/6, 3/6]
check("softmax of [0, ln 2, ln 3]", near(S.softmax([0, Math.log(2), Math.log(3)]), [1 / 6, 2 / 6, 3 / 6]), true);
check("uniform logits give uniform probabilities", near(S.softmax([1, 1, 1, 1]), [0.25, 0.25, 0.25, 0.25]), true);
check("shift invariance", near(S.softmax([5, 1, 0.5]), S.softmax([6, 2, 1.5])), true);
check("no overflow for large logits", near(S.softmax([1000, 1000]), [0.5, 0.5]), true);
check("sums to 1", Math.abs(S.softmax([3, -2, 0.1, 7]).reduce((a: number, b: number) => a + b, 0) - 1) < 1e-12, true);

// --- Entropy, argmax, snap -------------------------------------------------------------
check("entropy of one-hot is 0", S.entropy([1, 0, 0]), 0);
check("entropy of uniform is ln K", Math.abs(S.entropy([0.2, 0.2, 0.2, 0.2, 0.2]) - Math.log(5)) < 1e-12, true);
check("argmax picks first on ties", S.argmax([1, 3, 3, 2]), 1);
check("snap 0.1 + 0.2", S.snap(0.1 + 0.2, 0.1, -10, 10), 0.3);
check("snap clamps", [S.snap(12.34, 0.1, -10, 10), S.snap(-11, 0.1, -10, 10)], [10, -10]);
check("snap -0.04 is 0, not -0", Object.is(S.snap(-0.04, 0.1, -10, 10), 0), true);

// --- Configuration ---------------------------------------------------------------------
const K = CFG.CLASSES.length;
check("every preset has one logit per class", CFG.PRESETS.every((p: any) => p.logits.length === K), true);
check(
  "preset logits lie in the axis range",
  CFG.PRESETS.every((p: any) => p.logits.every((z: number) => z >= CFG.LOGIT_MIN && z <= CFG.LOGIT_MAX)),
  true,
);
check("preset ids are unique", new Set(CFG.PRESETS.map((p: any) => p.id)).size, CFG.PRESETS.length);
check("default preset exists", CFG.PRESETS.some((p: any) => p.id === CFG.DEFAULT_PRESET), true);

console.log(failed ? `\n${failed} check(s) FAILED` : "\nAll checks passed");
if (failed) Deno.exit(1);
