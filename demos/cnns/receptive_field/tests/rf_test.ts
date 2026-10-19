// Checks for rf.js and config.js. Run from the repository root:
//   quarto run demos/cnns/receptive_field/tests/rf_test.ts
import "../rf.js";
import "../config.js";

const { RF, RFDemoConfig: CFG } = globalThis as any;
let failed = 0;
function check(name: string, actual: unknown, expected: unknown) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${ok ? "" : `: got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}`}`);
}

// --- Hand-calculated examples ----------------------------------------------------
// Quiz in the lecture: (k=3, s=1), (k=3, s=2), (k=5, s=1) -> 2 + 2 + 4*2 + 1 = 13.
const quiz = [{ k: 3, s: 1 }, { k: 3, s: 2 }, { k: 5, s: 1 }];
check("quiz: receptive field 13", RF.receptiveField(quiz), 13);
check("quiz: per layer 3, 5, 13", [1, 2, 3].map((l) => RF.receptiveField(quiz, l)), [3, 5, 13]);
check("three 3x3, stride 1: 7", RF.receptiveField([{ k: 3, s: 1 }, { k: 3, s: 1 }, { k: 3, s: 1 }]), 7);
check("input has receptive field 1", RF.receptiveField(quiz, 0), 1);
check("sizes 32 -> 32 -> 16 -> 16", RF.sizes(32, quiz), [32, 32, 16, 16]);
check("output size k=7, s=2, i=5", RF.outputSize(5, 7, 2), 3);
check("padding of k = 1, 3, 5, 7", [1, 3, 5, 7].map(RF.padding), [0, 1, 2, 3]);
// Index 0 of the last map of `quiz`: [0, 0] -> [-2, 2] -> [-2·2 - 1, 2·2 - 1 + 2] = [-5, 5] -> [-6, 6].
check("ranges of index 0", RF.ranges(quiz, 0), [[-6, 6], [-5, 5], [-2, 2], [0, 0]]);
// 1-D paths for two 3x3 layers, stride 1: [1, 2, 3, 2, 1] around the selected index.
check("paths: two 3x3 layers", RF.paths(9, [{ k: 3, s: 1 }, { k: 3, s: 1 }], 4)[0], [0, 0, 1, 2, 3, 2, 1, 0, 0]);

// --- Every configuration of the demo against a brute-force dependency search -------
// Follows every connection from the selected index backwards. `all` also continues from
// padding positions and gives the unclipped range; `paths` continues only from activations
// inside the map (padding is a constant and depends on nothing) and counts the paths.
function bruteForce(i: number, layers: { k: number; s: number }[], index: number) {
  const size = RF.sizes(i, layers);
  let all = new Set<number>([index]);
  let paths = new Map<number, number>([[index, 1]]);
  const out = [{ lo: index, hi: index, paths }];
  for (let l = layers.length - 1; l >= 0; l--) {
    const { k, s } = layers[l];
    const p = Math.floor(k / 2);
    const nextAll = new Set<number>();
    const nextPaths = new Map<number, number>();
    for (const j of all) for (let u = 0; u < k; u++) nextAll.add(j * s - p + u);
    for (const [j, count] of paths) {
      if (j < 0 || j >= size[l + 1]) continue;
      for (let u = 0; u < k; u++) nextPaths.set(j * s - p + u, (nextPaths.get(j * s - p + u) ?? 0) + count);
    }
    out.unshift({ lo: Math.min(...nextAll), hi: Math.max(...nextAll), paths: nextPaths });
    all = nextAll;
    paths = nextPaths;
  }
  return out;
}
const kernelChoices = CFG.KERNEL_SIZES.flatMap((k: number) => CFG.STRIDES.map((s: number) => ({ k, s })));
let combos = 0, mismatches = 0;
function* networks(depth: number): Generator<{ k: number; s: number }[]> {
  if (depth === 0) return yield [];
  for (const rest of networks(depth - 1)) for (const layer of kernelChoices) yield [...rest, layer];
}
for (const depth of CFG.LAYER_COUNTS) {
  for (const net of networks(depth)) {
    const size = RF.sizes(CFG.INPUT_SIZE, net);
    const last = size[size.length - 1];
    for (const index of [0, Math.floor((last - 1) / 2), last - 1]) {
      combos++;
      const ranges = RF.ranges(net, index);
      const paths = RF.paths(CFG.INPUT_SIZE, net, index);
      const ref = bruteForce(CFG.INPUT_SIZE, net, index);
      const ok =
        size.every((n: number) => n >= 1) &&
        ranges[0][1] - ranges[0][0] + 1 === RF.receptiveField(net) &&
        ref.every((m, l) => {
          const inside = paths[l].every((count: number, j: number) => count === (m.paths.get(j) ?? 0));
          return inside && m.lo === ranges[l][0] && m.hi === ranges[l][1];
        });
      if (!ok) mismatches++;
    }
  }
}
check(`brute-force comparison over ${combos} networks and positions: mismatches`, mismatches, 0);

// --- Configuration ---------------------------------------------------------------
check("defaults hold the maximum number of layers", CFG.DEFAULTS.layers.length, Math.max(...CFG.LAYER_COUNTS));
check("default layer count is offered", CFG.LAYER_COUNTS.includes(CFG.DEFAULTS.layerCount), true);
check(
  "default kernel sizes and strides are offered",
  CFG.DEFAULTS.layers.every((l: any) => CFG.KERNEL_SIZES.includes(l.k) && CFG.STRIDES.includes(l.s)),
  true,
);

console.log(failed ? `\n${failed} check(s) FAILED` : "\nAll checks passed");
if (failed) Deno.exit(1);
