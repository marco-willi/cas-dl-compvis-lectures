// Configuration of the softmax demo: edit the values below to change what the demo offers.
// Loaded as a classic script (works from file:// and on GitHub Pages).
globalThis.SoftmaxDemoConfig = {
  // Class names (camera-trap example used throughout the course). The number of names
  // defines the number of classes; every preset needs one logit per class.
  CLASSES: ["empty", "deer", "fox", "badger", "bird"],

  // Range of the logit axis and step of the inputs. Dragged values are rounded to STEP.
  LOGIT_MIN: -10,
  LOGIT_MAX: 10,
  STEP: 0.1,

  PRESETS: [
    { id: "clear", name: "Clear winner", logits: [5, 1, 0.5, 0, -0.5] },
    { id: "uniform", name: "Uniform logits", logits: [1, 1, 1, 1, 1] },
    { id: "close", name: "Close competition", logits: [2, 1.8, 1.5, 1.2, 1] },
    { id: "confident", name: "Very confident", logits: [10, 0, 0, 0, 0] },
    { id: "negative", name: "Only negative logits", logits: [-0.5, -1, -1.5, -2, -2.5] },
    { id: "two", name: "Two leaders", logits: [4, 3.8, 0, 0, 0] },
    { id: "zeros", name: "All zero", logits: [0, 0, 0, 0, 0] },
  ],

  DEFAULT_PRESET: "clear", // shown after loading and after "Reset"
};
