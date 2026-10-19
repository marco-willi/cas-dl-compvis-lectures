// Receptive fields of a stack of convolutional layers (one channel, no dilation).
// Pure functions without DOM access, shared by the page (app.js) and the tests.
//
// A network is a list of layers [{ k, s }, ...] (kernel size, stride). Every layer uses the
// padding p = floor(k / 2), which keeps the size for stride 1 ("same" padding for odd k).
// Maps are numbered 0 (input) to L (output of the last layer); all maps are square, so one
// axis is enough: the 2-D receptive field is the product of the row and the column range.
globalThis.RF = (() => {
  "use strict";

  const padding = (k) => Math.floor(k / 2);

  // Output size along one axis.
  const outputSize = (i, k, s, p = padding(k)) => Math.floor((i + 2 * p - k) / s) + 1;

  // Sizes of all maps: sizes[0] = input size, sizes[l] = output size of layer l.
  function sizes(i, layers) {
    const out = [i];
    for (const { k, s } of layers) out.push(outputSize(out[out.length - 1], k, s));
    return out;
  }

  // Receptive field of one activation of layer l with respect to the input (closed form):
  // r = sum_{j=1..l} (k_j - 1) * prod_{m=1..j-1} s_m + 1.
  function receptiveField(layers, l = layers.length) {
    let r = 1;
    let jump = 1; // product of the strides of the preceding layers
    for (let j = 0; j < l; j++) {
      r += (layers[j].k - 1) * jump;
      jump *= layers[j].s;
    }
    return r;
  }

  // Index range [lo, hi] in every map that influences index `index` of the last map.
  // The ranges are not clipped: indices outside 0..size-1 fall into the zero-padding.
  function ranges(layers, index) {
    const L = layers.length;
    const out = new Array(L + 1);
    out[L] = [index, index];
    for (let l = L; l >= 1; l--) {
      const { k, s } = layers[l - 1];
      const [lo, hi] = out[l];
      out[l - 1] = [lo * s - padding(k), hi * s - padding(k) + k - 1];
    }
    return out;
  }

  // Number of paths from every index of every map to `index` of the last map (along one
  // axis; in 2-D the count of cell (r, c) is paths[r] * paths[c]). Padding is not counted.
  function paths(i, layers, index) {
    const size = sizes(i, layers);
    const L = layers.length;
    const out = new Array(L + 1);
    out[L] = Array.from({ length: size[L] }, (_, j) => (j === index ? 1 : 0));
    for (let l = L; l >= 1; l--) {
      const { k, s } = layers[l - 1];
      const previous = new Array(size[l - 1]).fill(0);
      out[l].forEach((count, j) => {
        if (!count) return;
        for (let u = 0; u < k; u++) {
          const x = j * s - padding(k) + u;
          if (x >= 0 && x < size[l - 1]) previous[x] += count;
        }
      });
      out[l - 1] = previous;
    }
    return out;
  }

  return { padding, outputSize, sizes, receptiveField, ranges, paths };
})();
