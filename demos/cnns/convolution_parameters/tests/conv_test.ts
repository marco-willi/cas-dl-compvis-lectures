// Checks for conv.js and config.js. Run from the repository root:
//   quarto run demos/cnns/convolution_parameters/tests/conv_test.ts
import "../conv.js";
import "../config.js";
import "../exercises.js";

const { Conv, ConvDemoConfig: CFG, ConvExercises: EX } = globalThis as any;
let failed = 0;
function check(name: string, actual: unknown, expected: unknown) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${ok ? "" : `: got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}`}`);
}
const close = (a: number[][], b: number[][]) =>
  a.length === b.length && a.every((row, i) => row.length === b[i].length && row.every((v, j) => Math.abs(v - b[i][j]) < 1e-12));

// --- Hand-calculated example -------------------------------------------------
// Input x[r][c] = 5r + c (5 x 5), kernel with negative and decimal weights (sum 4.5).
const x = Array.from({ length: 5 }, (_, r) => Array.from({ length: 5 }, (_, c) => 5 * r + c));
const w = [
  [1, 2, 3],
  [0, -1, 0.5],
  [-2, 0, 1],
];
const run = (padding: number, stride: number, dilation: number, relu = false) =>
  Conv.correlate(x, w, { padding, stride, dilation, relu });

// y[0][0] = (0·1 + 1·2 + 2·3) + (5·0 − 6 + 7·0.5) + (−2·10 + 0 + 12) = 8 − 2.5 − 8 = −2.5.
// One step right adds 4.5 (sum of weights), one step down adds 5 · 4.5 = 22.5.
check("hand: no padding, stride 1", run(0, 1, 1).values, [
  [-2.5, 2, 6.5],
  [20, 24.5, 29],
  [42.5, 47, 51.5],
]);
// Stride 2 keeps every second position of the stride-1 result.
check("hand: stride 2", run(0, 2, 1).values, [
  [-2.5, 6.5],
  [42.5, 51.5],
]);
// Dilation 2 samples rows/columns 0, 2, 4: (0 + 4 + 12) + (0 − 12 + 7) + (−40 + 0 + 24) = −5.
check("hand: dilation 2", run(0, 1, 2).values, [[-5]]);
// Padding 1, y[0][0]: kernel row 0 lies in the padding; row 1: 0.5·x[0][1] = 0.5; row 2: 1·x[1][1] = 6.
check("hand: padding 1, corner", run(1, 1, 1).values[0][0], 6.5);
check("hand: padding 1, size", run(1, 1, 1).size, 5);
check("hand: padding 1, interior equals unpadded", run(1, 1, 1).values[1][1], -2.5);
check("hand: ReLU", run(0, 1, 1, true).values[0], [0, 2, 6.5]);
check("hand: sums stay unclipped with ReLU", run(0, 1, 1, true).sums[0][0], -2.5);

// --- Sampled positions ---------------------------------------------------------
check(
  "terms: i=1, j=0, padding 1, stride 2, dilation 2",
  Conv.terms(x, w, { padding: 1, stride: 2, dilation: 2 }, 1, 0).map((t: any) => [t.row, t.col, t.inside, t.x]),
  [
    [1, -1, false, 0], [1, 1, true, 6], [1, 3, true, 8],
    [3, -1, false, 0], [3, 1, true, 16], [3, 3, true, 18],
    [5, -1, false, 0], [5, 1, false, 0], [5, 3, false, 0],
  ],
);

// --- Output size, effective kernel size, invalid configurations ------------------
const sizes: [number, number, number, number, number, number][] = [
  // n, k, p, s, d, expected
  [9, 3, 0, 1, 1, 7], [9, 3, 1, 1, 1, 9], [9, 5, 2, 1, 1, 9], [9, 7, 3, 1, 1, 9], [9, 1, 3, 1, 1, 15],
  [9, 3, 0, 2, 1, 4], [9, 3, 1, 2, 1, 5], [9, 3, 0, 3, 1, 3], [9, 5, 0, 3, 1, 2],
  [9, 3, 0, 1, 2, 5], [9, 3, 0, 1, 3, 3], [9, 5, 0, 1, 2, 1], [9, 7, 2, 1, 2, 1],
];
for (const [n, k, p, s, d, expected] of sizes) check(`size n=${n} k=${k} p=${p} s=${s} d=${d}`, Conv.outputSize(n, k, p, s, d), expected);
check("effective kernel size k=3 d=2", Conv.effectiveKernelSize(3, 2), 5);
check("effective kernel size k=7 d=3", Conv.effectiveKernelSize(7, 3), 19);
for (const [k, p, s, d] of [[5, 0, 1, 3], [7, 0, 1, 2], [7, 1, 1, 2], [7, 0, 3, 2], [7, 3, 2, 3]]) {
  const out = Conv.correlate(CFG.INPUT_PRESETS[0].values, CFG.FILTER_PRESETS[0].weights(k), { padding: p, stride: s, dilation: d });
  check(`invalid k=${k} p=${p} s=${s} d=${d}`, [out.valid, out.size <= 0, out.values], [false, true, []]);
}

// --- Every configuration of the demo against an explicit pad-then-slide reference --
function reference(input: number[][], kernel: number[][], p: number, s: number, d: number) {
  const n = input.length, k = kernel.length, m = n + 2 * p, eff = d * (k - 1) + 1;
  const padded = Array.from({ length: m }, (_, r) => Array.from({ length: m }, (_, c) => (input[r - p]?.[c - p] ?? 0)));
  const out: number[][] = [];
  for (let top = 0; top + eff <= m; top += s) {
    const row: number[] = [];
    for (let left = 0; left + eff <= m; left += s) {
      let sum = 0;
      for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) sum += padded[top + u * d][left + v * d] * kernel[u][v];
      row.push(sum);
    }
    out.push(row);
  }
  return out;
}
let combos = 0, mismatches = 0;
for (const preset of CFG.INPUT_PRESETS) {
  for (const k of CFG.KERNEL_SIZES) {
    const kernel = Array.from({ length: k }, (_, u) => Array.from({ length: k }, (_, v) => Math.sin(1 + 3 * u + 7 * v)));
    for (const p of CFG.PADDINGS) for (const s of CFG.STRIDES) for (const d of CFG.DILATIONS) {
      const out = Conv.correlate(preset.values, kernel, { padding: p, stride: s, dilation: d });
      const ref = reference(preset.values, kernel, p, s, d);
      combos++;
      if (!close(out.values, ref) || out.valid !== ref.length > 0 || (out.valid && out.size !== ref.length)) mismatches++;
    }
  }
}
check(`reference comparison over ${combos} configurations: mismatches`, mismatches, 0);

// --- Configuration ---------------------------------------------------------------
const n = CFG.INPUT_PRESETS[0].values.length;
check("inputs are square and of equal size", CFG.INPUT_PRESETS.every((p: any) => p.values.length === n && p.values.every((r: number[]) => r.length === n)), true);
for (const k of CFG.KERNEL_SIZES) {
  const box = CFG.FILTER_PRESETS.find((f: any) => f.id === "box").weights(k);
  const sum = box.flat().reduce((a: number, b: number) => a + b, 0);
  check(`box filter ${k}x${k} is uniform and sums to 1`, [box.length, box.flat().every((v: number) => v === 1 / (k * k)), Math.abs(sum - 1) < 1e-12], [k, true, true]);
  check(`fallback filter defined for ${k}x${k}`, CFG.FILTER_PRESETS.find((f: any) => f.id === CFG.FALLBACK_FILTER).weights(k) !== null, true);
  for (const f of CFG.FILTER_PRESETS) {
    const m = f.weights(k);
    if (m) check(`${f.id} ${k}x${k} has shape and |output| <= 1`, [m.length, m.every((r: number[]) => r.length === k), m.flat().filter((v: number) => v > 0).reduce((a: number, b: number) => a + b, 0) <= 1 + 1e-12], [k, true, true]);
  }
}

// --- Exercises: each intended solution passes, the default state and near misses do not --
const T = CFG.INPUT_PRESETS.find((p: any) => p.id === "t").values;
const preset = (id: string, k: number) => CFG.FILTER_PRESETS.find((f: any) => f.id === id).weights(k);
const above = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]; // x(r, c) - x(r - 1, c)
function solves(id: string, kernel: number[][], padding: number, stride: number, dilation: number, relu = false) {
  const task = EX.TASKS.find((t: any) => t.id === id);
  const result = Conv.correlate(T, kernel, { padding, stride, dilation, relu });
  return task.check({ input: T, kernelSize: kernel.length, padding, stride, dilation, relu, result });
}
check("exercise same-size: k=5, p=2", solves("same-size", preset("box", 5), 2, 1, 1), true);
check("exercise same-size: k=5, p=1 fails", solves("same-size", preset("box", 5), 1, 1, 1), false);
check("exercise downsample: k=3, p=1, s=2", solves("downsample", preset("box", 3), 1, 2, 1), true);
check("exercise downsample: k=3, p=2, s=2, d=2", solves("downsample", preset("box", 3), 2, 2, 2), true);
check("exercise downsample: k=3, p=0, s=2 fails", solves("downsample", preset("box", 3), 0, 2, 1), false);
check("exercise whole-t: k=3, d=3", solves("whole-t", preset("box", 3), 0, 1, 3), true);
check("exercise whole-t: k=3, d=3, p=3", solves("whole-t", preset("box", 3), 3, 1, 3), true);
check("exercise whole-t: k=3, d=2 fails", solves("whole-t", preset("box", 3), 2, 1, 2), false);
check("exercise whole-t: k=3, d=3, s=3, p=1 fails", solves("whole-t", preset("box", 3), 1, 3, 3), false);
check("exercise copy: identity k=3, p=1", solves("copy", preset("identity", 3), 1, 1, 1), true);
check("exercise copy: identity k=1, p=0", solves("copy", preset("identity", 1), 0, 1, 1), true);
check("exercise copy: identity k=3, d=2, p=2", solves("copy", preset("identity", 3), 2, 1, 2), true);
check("exercise copy: identity k=3, p=0 fails", solves("copy", preset("identity", 3), 0, 1, 1), false);
check("exercise top-edge: x(r,c) - x(r-1,c) with ReLU", solves("top-edge", above, 1, 1, 1, true), true);
check("exercise top-edge: same without ReLU fails", solves("top-edge", above, 1, 1, 1, false), false);
check("exercise top-edge: preset horizontal edges fails", solves("top-edge", preset("horizontal", 3), 1, 1, 1, true), false);
check("top edges of T: row 1, columns 1-7", EX.topEdges(T).flatMap((row: boolean[], r: number) => row.flatMap((v, c) => (v ? [[r, c]] : []))),
  [1, 2, 3, 4, 5, 6, 7].map((c) => [1, c]));
const D = CFG.DEFAULTS;
check("default state solves no exercise", EX.TASKS.filter((t: any) => {
  const kernel = preset(D.filter, D.kernelSize);
  const result = Conv.correlate(T, kernel, D);
  return t.check({ input: T, ...D, result });
}).map((t: any) => t.id), []);

console.log(failed ? `\n${failed} check(s) FAILED` : "\nAll checks passed");
if (failed) Deno.exit(1);
