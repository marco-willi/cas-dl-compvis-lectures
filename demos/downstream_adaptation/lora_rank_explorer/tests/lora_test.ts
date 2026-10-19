// Checks for lora.js. Run from the repository root:
//   quarto run demos/downstream_adaptation/lora_rank_explorer/tests/lora_test.ts
import "../lora.js";

const { Lora } = globalThis as any;
let failed = 0;
function check(name: string, actual: unknown, expected: unknown) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${ok ? "" : `: got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}`}`);
}

// 768 x 768, r = 8: 589,824 full, 8 * (768 + 768) = 12,288 LoRA, ratio 1 / 48 = 2.0833...%.
const vit = Lora.counts(768, 768, 8);
check("768x768 r=8: full", vit.full, 589824);
check("768x768 r=8: lora", vit.lora, 12288);
check("768x768 r=8: ratio", vit.ratio, 1 / 48);
check("768x768 r=8: ratio in percent", (vit.ratio * 100).toFixed(4), "2.0833");
check("768x768 r=8: difference", vit.difference, 577536);

// 64 x 64, r = 32: break-even, 4,096 parameters each.
const even = Lora.counts(64, 64, 32);
check("64x64 r=32: full", even.full, 4096);
check("64x64 r=32: lora", even.lora, 4096);
check("64x64 r=32: ratio", even.ratio, 1);
check("64x64 r=32: difference", even.difference, 0);
check("64x64: break-even rank", Lora.breakEvenRank(64, 64), 32);

// Above the break-even rank LoRA has more parameters than the full matrix.
const over = Lora.counts(64, 64, 64);
check("64x64 r=64: lora", over.lora, 8192);
check("64x64 r=64: ratio", over.ratio, 2);
check("64x64 r=64: difference", over.difference, -4096);

// Rectangular matrix: 1024 inputs, 256 outputs.
const rect = Lora.counts(1024, 256, 4);
check("256x1024 r=4: full", rect.full, 262144);
check("256x1024 r=4: lora", rect.lora, 5120);
check("256x1024: max rank", Lora.maxRank(1024, 256), 256);
check("256x1024: break-even rank", Lora.breakEvenRank(1024, 256), 204.8);

// Several target matrices multiply both counts and leave the ratio unchanged
// (ViT-B/16, query and value in 12 blocks: 24 matrices).
const many = Lora.counts(768, 768, 8, 24);
check("24 matrices: full", many.full, 14155776);
check("24 matrices: lora", many.lora, 294912);
check("24 matrices: ratio", many.ratio, vit.ratio);

if (failed) {
  console.log(`${failed} check(s) failed`);
  Deno.exit(1);
}
console.log("all checks passed");
