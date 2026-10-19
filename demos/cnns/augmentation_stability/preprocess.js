// Model preprocessing for the augmentation stability demo.
//
// Reproduces torchvision's transform for EfficientNet_B0_Weights.IMAGENET1K_V1 on a PIL image:
//   resize (shorter side -> 256, bicubic with antialiasing) -> center crop 224 -> [0, 1] -> normalize
//
// The resize follows Pillow's implementation (separable convolution with 8-bit fixed-point
// coefficients, horizontal pass first), so the result matches Python bit for bit when both
// start from the same pixels.
"use strict";

const Preprocess = (() => {
  const PRECISION_BITS = 32 - 8 - 2; // as in Pillow's Resample.c for 8-bit images

  // Keys cubic convolution kernel with a = -0.5 (Pillow's BICUBIC).
  function bicubic(x) {
    const a = -0.5;
    if (x < 0) x = -x;
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
  }

  // For every output position: first contributing input index, and integer weights.
  function coefficients(inSize, outSize) {
    const scale = inSize / outSize;
    const filterScale = Math.max(scale, 1); // widen the kernel when downscaling (antialiasing)
    const support = 2 * filterScale;
    const ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2);
    const weights = new Int32Array(outSize * ksize);
    const k = new Float64Array(ksize);
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      const xmin = Math.max(0, Math.trunc(center - support + 0.5));
      const count = Math.min(inSize, Math.trunc(center + support + 0.5)) - xmin;
      let sum = 0;
      for (let x = 0; x < count; x++) {
        k[x] = bicubic((x + xmin - center + 0.5) / filterScale);
        sum += k[x];
      }
      for (let x = 0; x < count; x++) {
        const w = (sum !== 0 ? k[x] / sum : k[x]) * (1 << PRECISION_BITS);
        weights[xx * ksize + x] = Math.trunc(w < 0 ? w - 0.5 : w + 0.5);
      }
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = count;
    }
    return { bounds, weights, ksize };
  }

  const clip8 = v => (v < 0 ? 0 : v > 255 ? 255 : v);

  // Resample along one axis. `data` is RGBA; alpha is ignored and set to 255.
  function resampleAxis(data, w, h, outSize, horizontal) {
    const inSize = horizontal ? w : h;
    const { bounds, weights, ksize } = coefficients(inSize, outSize);
    const outW = horizontal ? outSize : w, outH = horizontal ? h : outSize;
    const out = new Uint8ClampedArray(outW * outH * 4);
    const step = (horizontal ? 1 : w) * 4;       // distance between neighbours along the axis
    const lines = horizontal ? h : w;            // number of independent rows / columns
    const lineStep = (horizontal ? w : 1) * 4;
    const outStep = (horizontal ? 1 : outW) * 4;
    const outLineStep = (horizontal ? outW : 1) * 4;
    for (let line = 0; line < lines; line++) {
      for (let xx = 0; xx < outSize; xx++) {
        const xmin = bounds[xx * 2], count = bounds[xx * 2 + 1], k0 = xx * ksize;
        let r = 1 << (PRECISION_BITS - 1), g = r, b = r; // rounding offset
        let p = line * lineStep + xmin * step;
        for (let x = 0; x < count; x++, p += step) {
          const wgt = weights[k0 + x];
          r += data[p] * wgt;
          g += data[p + 1] * wgt;
          b += data[p + 2] * wgt;
        }
        const o = line * outLineStep + xx * outStep;
        out[o] = clip8(r >> PRECISION_BITS);
        out[o + 1] = clip8(g >> PRECISION_BITS);
        out[o + 2] = clip8(b >> PRECISION_BITS);
        out[o + 3] = 255;
      }
    }
    return out;
  }

  // Pillow-compatible bicubic resize of an RGBA buffer.
  function resizeBicubic(data, w, h, newW, newH) {
    if (newW !== w) data = resampleAxis(data, w, h, newW, true);
    if (newH !== h) data = resampleAxis(data, newW, h, newH, false);
    return data;
  }

  // Python's round(): halves go to the nearest even integer (used by torchvision's center crop).
  function roundHalfEven(x) {
    const r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  // Geometry of the preprocessing for a w x h image: resized size and crop offset.
  function geometry(w, h, cfg) {
    const [newW, newH] = w <= h
      ? [cfg.resize, Math.trunc(cfg.resize * h / w)]
      : [Math.trunc(cfg.resize * w / h), cfg.resize];
    return {
      newW, newH,
      left: roundHalfEven((newW - cfg.crop) / 2),
      top: roundHalfEven((newH - cfg.crop) / 2),
    };
  }

  // RGBA pixels (w x h) -> { tensor: Float32Array in CHW order, crop: RGBA pixels the model sees }.
  function run(data, w, h, cfg) {
    const { newW, newH, left, top } = geometry(w, h, cfg);
    const resized = resizeBicubic(data, w, h, newW, newH);
    const n = cfg.crop, plane = n * n;
    const crop = new Uint8ClampedArray(plane * 4);
    const tensor = new Float32Array(3 * plane);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const s = ((y + top) * newW + (x + left)) * 4, d = y * n + x;
        for (let c = 0; c < 3; c++) {
          crop[d * 4 + c] = resized[s + c];
          tensor[c * plane + d] = (resized[s + c] / 255 - cfg.mean[c]) / cfg.std[c];
        }
        crop[d * 4 + 3] = 255;
      }
    }
    return { tensor, crop };
  }

  return { run, geometry, resizeBicubic };
})();
