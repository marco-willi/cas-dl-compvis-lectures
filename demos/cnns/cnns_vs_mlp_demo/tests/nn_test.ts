// Checks nn.js against the PyTorch references exported with the models. Run from the
// repository root:
//   quarto run demos/cnns/cnns_vs_mlp_demo/tests/nn_test.ts
import "./window_shim.ts";
import "../nn.js";
import "../model/manifest.js";
import "../model/samples.js";
import "../model/position_maps.js";
import "../model/reference.js";
import "../model/weights/mlp_tl.js";
import "../model/weights/cnn_fc_tl.js";
import "../model/weights/cnn_gap_tl.js";
import "../model/weights/mlp_all.js";
import "../model/weights/cnn_fc_all.js";
import "../model/weights/cnn_gap_all.js";

const G = globalThis as any;
const NN = G.MnistShiftNN;
const MANIFEST = G.MNIST_SHIFT_MANIFEST;
const REF = G.MNIST_SHIFT_REFERENCE;
const SAMPLES = G.MNIST_SHIFT_SAMPLES;
let failed = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ": " + detail : ""}`);
}

// --- Primitives -----------------------------------------------------------------------
check("half 0x3c00 = 1", NN.halfToFloat(0x3c00) === 1);
check("half 0xc000 = -2", NN.halfToFloat(0xc000) === -2);
check("half 0x0001 = 2^-24", NN.halfToFloat(0x0001) === 2 ** -24);
const digit = NN.decodeDigit(SAMPLES.items[0]);
const placed = NN.place(digit, 28, 3);
check("place: pixel (r, c) of the digit lands at (3 + r, 28 + c)",
  placed.data[(3 + 10) * 56 + 28 + 12] === digit[10 * 28 + 12]);
check("place: canvas sum equals digit sum",
  Math.abs(placed.data.reduce((a: number, b: number) => a + b, 0) - digit.reduce((a: number, b: number) => a + b, 0)) < 1e-4);

// --- Reference outputs ----------------------------------------------------------------
const tol = REF.tolerance;
const weights: Record<string, unknown> = {};
for (const [id, cases] of Object.entries(REF.models) as [string, any[]][]) {
  weights[id] = NN.decodeWeights(G.MNIST_SHIFT_WEIGHTS[id]);
  let err = 0;
  const start = performance.now();
  for (const c of cases) {
    const input = NN.place(NN.decodeDigit(SAMPLES.items[c.sample]), c.dx, c.dy);
    const { logits, acts } = NN.forward(MANIFEST.models[id].layers, weights[id], input);
    c.logits.forEach((v: number, i: number) => (err = Math.max(err, Math.abs(v - logits[i]))));
    for (const key of ["conv1", "conv3"]) {
      const ref = c[key + "_first20"];
      if (ref) ref.forEach((v: number, i: number) => (err = Math.max(err, Math.abs(v - acts[key].data[i]))));
    }
  }
  const ms = (performance.now() - start) / cases.length;
  check(`${id}: logits and activations within ${tol}`, err < tol, `max error ${err.toExponential(2)}, ${ms.toFixed(1)} ms per pass`);
}

// --- Samples are correct at the training position ---------------------------------------
for (const id of ["mlp_tl", "cnn_fc_tl", "cnn_gap_tl"]) {
  const correct = SAMPLES.items.filter((item: any) => {
    const { logits } = NN.forward(MANIFEST.models[id].layers, weights[id], NN.place(NN.decodeDigit(item), 0, 0));
    return logits.indexOf(Math.max(...logits)) === item.label;
  }).length;
  check(`${id}: all 50 samples correct at (0, 0)`, correct === 50, `${correct}/50`);
}

// --- Position maps ------------------------------------------------------------------------
const maps = G.MNIST_SHIFT_MAPS;
check("position maps: 6 models x 29 x 29",
  Object.keys(MANIFEST.models).every((id) => maps.maps[id].acc.length === 29 && maps.maps[id].acc.every((r: number[]) => r.length === 29)));

if (failed) {
  console.log(`\n${failed} check(s) failed`);
  Deno.exit(1);
}
console.log("\nall checks passed");
