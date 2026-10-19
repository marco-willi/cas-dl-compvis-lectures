// Configuration of the convolution-parameter demo: edit the arrays below to change what the
// demo offers. Loaded as a classic script (works from file:// and on GitHub Pages).
globalThis.ConvDemoConfig = (() => {
  "use strict";

  // Builds a k x k matrix from a function of (row, column).
  const matrix = (k, fn) => Array.from({ length: k }, (_, u) => Array.from({ length: k }, (_, v) => fn(u, v)));

  // -------------------------------------------------------------------------
  // Input presets: square matrices, all of the same size, values in [0, 1]
  // (0 = black, 1 = white). The first size found defines the input size.
  // -------------------------------------------------------------------------
  const INPUT_PRESETS = [
    {
      id: "t",
      name: "Letter T",
      values: [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
      ],
    },
    {
      id: "square",
      name: "Square",
      values: [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
      ],
    },
    {
      id: "cross",
      name: "Cross",
      values: [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
      ],
    },
    { id: "diagonal", name: "Diagonal line", values: matrix(9, (r, c) => (r === c ? 1 : 0)) },
    { id: "checkerboard", name: "Checkerboard", values: matrix(9, (r, c) => (r + c) % 2) },
    { id: "gradient", name: "Gradient (left to right)", values: matrix(9, (r, c) => c / 8) },
    { id: "ones", name: "Uniform (all ones)", values: matrix(9, () => 1) },
  ];

  // -------------------------------------------------------------------------
  // Filter presets. `weights(k)` returns the k x k weights for kernel size k, or null if
  // the preset is not defined for that size (the demo then falls back to FALLBACK_FILTER).
  // All presets are scaled so that an input in [0, 1] gives an output in [-1, 1].
  // -------------------------------------------------------------------------
  const FILTER_PRESETS = [
    {
      id: "box",
      name: "Box average (normalized)",
      weights: (k) => matrix(k, () => 1 / (k * k)),
    },
    {
      id: "identity",
      name: "Identity",
      weights: (k) => matrix(k, (u, v) => (u === (k - 1) / 2 && v === (k - 1) / 2 ? 1 : 0)),
    },
    {
      id: "vertical",
      name: "Vertical edges",
      // Left half negative, centre column zero, right half positive.
      weights: (k) => (k < 3 ? null : matrix(k, (u, v) => Math.sign(v - (k - 1) / 2) / (k * (k - 1) / 2))),
    },
    {
      id: "horizontal",
      name: "Horizontal edges",
      weights: (k) => (k < 3 ? null : matrix(k, (u, v) => Math.sign(u - (k - 1) / 2) / (k * (k - 1) / 2))),
    },
    {
      id: "surround",
      name: "Centre minus surround",
      weights: (k) =>
        k < 3 ? null : matrix(k, (u, v) => (u === (k - 1) / 2 && v === (k - 1) / 2 ? 1 : -1 / (k * k - 1))),
    },
  ];

  return {
    INPUT_PRESETS,
    FILTER_PRESETS,
    FALLBACK_FILTER: "box", // must be defined for every kernel size

    // Values offered by the controls.
    KERNEL_SIZES: [1, 3, 5, 7],
    PADDINGS: [0, 1, 2, 3],
    STRIDES: [1, 2, 3],
    DILATIONS: [1, 2, 3],

    // State after loading the page and after "Reset".
    DEFAULTS: {
      input: "t",
      filter: "box",
      kernelSize: 3,
      padding: 0,
      stride: 1,
      dilation: 1,
      relu: false,
      colorRange: 1, // the output colour scale covers [-colorRange, +colorRange]
    },
  };
})();
