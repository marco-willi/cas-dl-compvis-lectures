// Texts and settings per model family. Notation follows the lecture "GenAI Model Families".
globalThis.CONFIG = {
  families: ["ar", "gan", "vae", "diffusion", "fm"],
  playSeconds: { ar: 6, diffusion: 4, fm: 3 }, // duration of "Play"
  stripSize: 6, // thumbnails of intermediate states
  ar: {
    tab: "Autoregressive",
    subtitle: "One token at a time: each step predicts a distribution over the next token, given all previous tokens.",
  },
  gan: {
    tab: "GAN",
    subtitle: "One forward pass: a new random input <b>z</b> gives a new image.",
  },
  vae: {
    tab: "VAE",
    modes: { generate: "Generate", reconstruct: "Reconstruct" },
    subtitle: {
      generate: "Generate: draw <b>z</b> from the prior and decode it in one pass. The encoder is only needed for training.",
      reconstruct: "Reconstruct: the encoder maps an image to a distribution over <b>z</b>, the decoder maps <b>z</b> back.",
    },
  },
  diffusion: {
    tab: "Diffusion",
    modes: { reverse: "Reverse (generate)", forward: "Forward (add noise)" },
    subtitle: {
      reverse: "Start from pure noise and remove noise step by step. Fresh noise is added in every step.",
      forward: "Add Gaussian noise step by step. Fixed process, no network: it creates the training examples.",
    },
  },
  fm: {
    tab: "Flow matching",
    subtitle: "Follow the learned velocity from noise (t = 0) to data (t = 1). Deterministic: the same noise always gives the same image.",
  },
};
