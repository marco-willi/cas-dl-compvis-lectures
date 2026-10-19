// Parameter counts of a LoRA update W' = W + (alpha / r) * B A for one weight matrix
// W (d_out x d_in), B (d_out x r), A (r x d_in).
// Pure functions without DOM access, shared by the page (app.js) and the tests.
globalThis.Lora = (() => {
  "use strict";

  // The rank of B A cannot exceed the smaller dimension of W.
  const maxRank = (dIn, dOut) => Math.min(dIn, dOut);

  // Rank at which LoRA has as many parameters as the full matrix (not necessarily an integer).
  const breakEvenRank = (dIn, dOut) => (dIn * dOut) / (dIn + dOut);

  // `modules` is the number of adapted weight matrices of this shape. The classifier head is
  // not part of either count.
  function counts(dIn, dOut, rank, modules = 1) {
    const full = dOut * dIn * modules;
    const lora = rank * (dIn + dOut) * modules;
    return {
      full,
      lora,
      ratio: lora / full,
      difference: full - lora, // > 0: LoRA trains fewer parameters
    };
  }

  return { maxRank, breakEvenRank, counts };
})();
