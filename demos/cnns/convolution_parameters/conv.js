// CNN-style cross-correlation with one input channel, one filter and zero bias.
// Pure functions without DOM access, shared by the page (app.js) and the tests.
globalThis.Conv = (() => {
  "use strict";

  // Extent of the input covered by a k x k kernel with dilation d.
  const effectiveKernelSize = (k, d) => d * (k - 1) + 1;

  // Output size along one axis; a value <= 0 means the kernel does not fit.
  const outputSize = (n, k, p, s, d) => Math.floor((n + 2 * p - d * (k - 1) - 1) / s) + 1;

  // Input index sampled by kernel entry u for output index i (may lie outside 0..n-1).
  const sampleIndex = (i, u, p, s, d) => i * s - p + u * d;

  // The k*k terms of output cell (i, j): sampled position, input value and weight.
  // Positions outside the input are zero-padding (`inside: false`, x = 0).
  function terms(input, kernel, { padding, stride, dilation }, i, j) {
    const n = input.length;
    const k = kernel.length;
    const out = [];
    for (let u = 0; u < k; u++) {
      for (let v = 0; v < k; v++) {
        const row = sampleIndex(i, u, padding, stride, dilation);
        const col = sampleIndex(j, v, padding, stride, dilation);
        const inside = row >= 0 && row < n && col >= 0 && col < n;
        out.push({ u, v, row, col, inside, x: inside ? input[row][col] : 0, w: kernel[u][v] });
      }
    }
    return out;
  }

  // Full output map. `sums` holds the values before, `values` after the optional ReLU.
  function correlate(input, kernel, { padding, stride, dilation, relu = false }) {
    const size = outputSize(input.length, kernel.length, padding, stride, dilation);
    const result = { size, valid: size > 0, sums: [], values: [] };
    for (let i = 0; i < size; i++) {
      const sums = [];
      for (let j = 0; j < size; j++) {
        let sum = 0;
        for (const t of terms(input, kernel, { padding, stride, dilation }, i, j)) sum += t.x * t.w;
        sums.push(sum);
      }
      result.sums.push(sums);
      result.values.push(relu ? sums.map((v) => Math.max(0, v)) : sums.slice());
    }
    return result;
  }

  return { effectiveKernelSize, outputSize, sampleIndex, terms, correlate };
})();
