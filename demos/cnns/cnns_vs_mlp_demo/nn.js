// Forward pass of the exported networks (no DOM access; also runs in Deno for the tests).
// Executes the layer list of model/manifest.js literally. Tensors are {data: Float32Array,
// shape: [C, H, W]} or {data, shape: [N]}; weights use the PyTorch layouts
// (conv [out, in, k, k], linear [out, in]), row-major.
"use strict";

(function (global) {
  // ---------------------------------------------------------------------------
  // Decoding
  // ---------------------------------------------------------------------------
  function base64Bytes(text) {
    const binary = atob(text);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  // IEEE 754 half precision -> float32.
  function halfToFloat(h) {
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x03ff;
    if (exponent === 0) return sign * fraction * 2 ** -24;
    if (exponent === 31) return fraction ? NaN : sign * Infinity;
    return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
  }

  function decodeTensor({ shape, data }, dtype) {
    const bytes = base64Bytes(data);
    const view = new DataView(bytes.buffer);
    const n = shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(n);
    if (dtype === "float16") for (let i = 0; i < n; i++) out[i] = halfToFloat(view.getUint16(2 * i, true));
    else for (let i = 0; i < n; i++) out[i] = view.getFloat32(4 * i, true);
    return { data: out, shape };
  }

  function decodeWeights(entry) {
    const tensors = {};
    for (const [name, t] of Object.entries(entry.tensors)) tensors[name] = decodeTensor(t, entry.dtype);
    return tensors;
  }

  // 28 x 28 uint8 digit (base64) -> Float32Array with values in [0, 1].
  function decodeDigit(item, size = 28) {
    const bytes = base64Bytes(item.pixels);
    const out = new Float32Array(size * size);
    for (let i = 0; i < out.length; i++) out[i] = bytes[i] / 255;
    return out;
  }

  // Paste the digit with its top-left corner at (dx, dy) onto a zero canvas (1 x canvas x canvas).
  function place(digit, dx, dy, canvas = 56, size = 28) {
    const data = new Float32Array(canvas * canvas);
    for (let r = 0; r < size; r++)
      for (let c = 0; c < size; c++) data[(dy + r) * canvas + dx + c] = digit[r * size + c];
    return { data, shape: [1, canvas, canvas] };
  }

  // ---------------------------------------------------------------------------
  // Layers
  // ---------------------------------------------------------------------------
  function conv2d(x, w, b, pad) {
    const [C, H, W] = x.shape;
    const [O, , k] = w.shape;
    const Ho = H + 2 * pad - k + 1;
    const Wo = W + 2 * pad - k + 1;
    const out = new Float32Array(O * Ho * Wo);
    for (let o = 0; o < O; o++) {
      const plane = o * Ho * Wo;
      out.fill(b.data[o], plane, plane + Ho * Wo);
      for (let c = 0; c < C; c++) {
        const xc = c * H * W;
        for (let i = 0; i < k; i++)
          for (let j = 0; j < k; j++) {
            const wv = w.data[((o * C + c) * k + i) * k + j];
            if (wv === 0) continue;
            // Output rows/columns for which the input pixel lies inside the image.
            const r0 = Math.max(0, pad - i), r1 = Math.min(Ho, H + pad - i);
            const c0 = Math.max(0, pad - j), c1 = Math.min(Wo, W + pad - j);
            for (let r = r0; r < r1; r++) {
              const src = xc + (r + i - pad) * W + (j - pad);
              const dst = plane + r * Wo;
              for (let col = c0; col < c1; col++) out[dst + col] += wv * x.data[src + col];
            }
          }
      }
    }
    return { data: out, shape: [O, Ho, Wo] };
  }

  function relu(x) {
    const out = new Float32Array(x.data.length);
    for (let i = 0; i < out.length; i++) out[i] = x.data[i] > 0 ? x.data[i] : 0;
    return { data: out, shape: x.shape };
  }

  function maxpool2d(x, k) {
    const [C, H, W] = x.shape;
    const Ho = Math.floor(H / k), Wo = Math.floor(W / k);
    const out = new Float32Array(C * Ho * Wo);
    for (let c = 0; c < C; c++)
      for (let r = 0; r < Ho; r++)
        for (let col = 0; col < Wo; col++) {
          let m = -Infinity;
          for (let i = 0; i < k; i++)
            for (let j = 0; j < k; j++) m = Math.max(m, x.data[(c * H + r * k + i) * W + col * k + j]);
          out[(c * Ho + r) * Wo + col] = m;
        }
    return { data: out, shape: [C, Ho, Wo] };
  }

  function gap(x) {
    const [C, H, W] = x.shape;
    const out = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let i = 0; i < H * W; i++) s += x.data[c * H * W + i];
      out[c] = s / (H * W);
    }
    return { data: out, shape: [C, 1, 1] };
  }

  function linear(x, w, b) {
    const [O, I] = w.shape;
    const out = new Float32Array(O);
    for (let o = 0; o < O; o++) {
      let s = b.data[o];
      for (let i = 0; i < I; i++) s += w.data[o * I + i] * x.data[i];
      out[o] = s;
    }
    return { data: out, shape: [O] };
  }

  // ---------------------------------------------------------------------------
  // Network
  // ---------------------------------------------------------------------------
  // Returns {logits, acts}: acts holds the output of every named conv layer after its ReLU
  // (e.g. acts.conv1) and acts.head, the vector that enters the first linear layer.
  function forward(layers, weights, input) {
    let x = input;
    let last = null;
    const acts = {};
    for (const layer of layers) {
      switch (layer.type) {
        case "conv2d":
          x = conv2d(x, weights[layer.name + ".weight"], weights[layer.name + ".bias"], layer.pad);
          last = layer.name;
          break;
        case "relu":
          x = relu(x);
          if (last) acts[last] = x;
          last = null;
          break;
        case "maxpool2d": x = maxpool2d(x, layer.k); break;
        case "gap": x = gap(x); break;
        case "flatten": x = { data: x.data, shape: [x.data.length] }; break;
        case "linear":
          if (!acts.head) acts.head = x;
          x = linear(x, weights[layer.name + ".weight"], weights[layer.name + ".bias"]);
          break;
        default: throw new Error("Unknown layer type: " + layer.type);
      }
    }
    return { logits: x.data, acts };
  }

  function softmax(logits) {
    const m = Math.max(...logits);
    const e = Array.from(logits, (v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  global.MnistShiftNN = { decodeWeights, decodeDigit, place, forward, softmax, halfToFloat };
})(globalThis);
