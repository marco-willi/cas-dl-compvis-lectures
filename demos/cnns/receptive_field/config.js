// Configuration of the receptive-field demo: edit the values below to change what the demo
// offers. Loaded as a classic script (works from file:// and on GitHub Pages).
globalThis.RFDemoConfig = {
  INPUT_SIZE: 32, // side length i of the (square) input

  // Values offered by the controls.
  LAYER_COUNTS: [3, 4],
  KERNEL_SIZES: [1, 3, 5, 7],
  STRIDES: [1, 2],

  // State after loading the page and after "Reset". `layers` always holds the maximum number
  // of layers; only the first `layerCount` are used.
  DEFAULTS: {
    layerCount: 3,
    layers: [
      { k: 3, s: 1 },
      { k: 3, s: 1 },
      { k: 3, s: 1 },
      { k: 3, s: 1 },
    ],
    paths: false,
  },
};
