// Softmax and summary statistics. Pure functions without DOM access, shared by the page
// (app.js) and the tests.
globalThis.Softmax = (() => {
  "use strict";

  // Softmax with the maximum subtracted first (same result, no overflow for large logits).
  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map((z) => Math.exp(z - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  // Entropy in nats; 0 for a one-hot distribution, log(K) for the uniform distribution.
  const entropy = (probs) => -probs.reduce((h, p) => (p > 0 ? h + p * Math.log(p) : h), 0);

  // Index of the largest value (first one on ties).
  const argmax = (values) => values.reduce((best, v, i) => (v > values[best] ? i : best), 0);

  // Rounds to a multiple of `step` and clamps to [lo, hi]; avoids results like 0.30000000000000004.
  function snap(value, step, lo, hi) {
    const decimals = Math.max(0, -Math.floor(Math.log10(step)));
    const snapped = Number((Math.round(value / step) * step).toFixed(decimals));
    return Math.min(hi, Math.max(lo, snapped)) || 0; // `|| 0` turns -0 into 0
  }

  return { softmax, entropy, argmax, snap };
})();
