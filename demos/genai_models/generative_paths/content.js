// Texts of the demo: per method a name, the question it answers and, per view, a fixed
// Observe / Explain / Try box (see .claude/plans/SPEC_GENAI_DEMOS_LECTURE_VIEW.md). `progress` is the
// label of the slider, for the views that have one.
globalThis.CONTENT = {
  diffusion: {
    name: "Diffusion",
    question: "How does iterative denoising turn noise into data?",
    progress: "Generation progress",
    views: {
      reverse: {
        observe: "The highlighted sample starts as pure noise and wanders. Only in the last part of the process does it settle into one of the two modes.",
        explain: "In every step, the model estimates where the clean data point lies and moves a small step towards it, then adds a little fresh noise. Early steps decide the coarse structure (which mode), late steps the details.",
        try: "Turn on “Show distribution”: at every step, the histogram follows the exact distribution of the noised data (dashed).",
      },
      forward: {
        progress: "Noising progress",
        observe: "Data samples (right) are mixed with more and more Gaussian noise until, at t = T, nothing but noise is left (left). Read this view from right to left.",
        explain: "This is the direction used for training: for every noise level, the noised data distribution is known exactly, and the model learns to undo one small step of it.",
        try: "Press Play: the slider runs from the data to the noise.",
      },
    },
  },
  ar: {
    name: "Autoregressive",
    question: "How does each choice depend on the previous one?",
    views: {
      tokens: {
        observe: "First the coarse token picks the unit interval, then the fine token picks the tenth within it. The probabilities of the second token depend on the first.",
        explain: "The model factorizes p(c, f) = p(c) · p(f | c) and samples one token after the other, each from the probabilities it predicts given the previous ones. Pixel models work the same way, with 256 values per pixel.",
        try: "Click a different coarse bar: how does p(f | c) change?",
      },
      seq: {
        observe: "Early on, the next value is drawn from a distribution shared by both directions (blue). After a few values it sits on one side only, although the plain distribution of that position (dashed) still has two peaks.",
        explain: "The model factorizes p(x₁, …, x₈) = p(x₁) · p(x₂ | x₁) · … · p(x₈ | x₁, …, x₇). Each value is drawn given all previous ones, so early random choices decide where the sequence goes. The last value follows the target distribution. Pixel and language models generate rows of pixels or tokens the same way.",
        try: "Draw a few sequences to the end: after how many values is the direction decided? Turn on “Show distribution” to see many sequences at once.",
      },
      "2d": {
        observe: "The spread and the position of x₂ depend on the x₁ drawn first.",
        explain: "p(x₁, x₂) = p(x₁) · p(x₂ | x₁): the second value is drawn given the first. Image models generate pixels or image tokens the same way, each given all previous ones.",
        try: "Drag x₁ from the left mode to the right one and watch p(x₂ | x₁) on the right.",
      },
    },
  },
  gan: {
    name: "GAN",
    question: "How can a single network evaluation produce a sample?",
    views: {
      generate: {
        observe: "The generator curve x = G(z) is flat over long stretches of z and very steep in between.",
        explain: "Flat parts collect many z into a small range of x: a mode. The steep part spreads few z over a wide range: the gap between the modes. Generating a sample is one evaluation of G, without intermediate steps.",
        try: "Drag z (the orange handle) slowly through the steep part: x jumps from the left mode to the right one.",
      },
      training: {
        progress: "Training step (not a generation step)",
        observe: "During training, G changes a lot. In the unstable run, it first collapses onto one mode and then onto a single value.",
        explain: "G is trained against a discriminator D that tries to tell real from generated samples (dashed curve, right). G improves by fooling D. This game does not have to converge smoothly.",
        try: "Switch to the unstable run and step through the training steps 25 to 100.",
      },
    },
  },
  vae: {
    name: "VAE",
    question: "Why does the decoder output a distribution instead of a value?",
    views: {
      generate: {
        observe: "The same z gives a different x every time you sample. The spread is the decoder's σθ(z).",
        explain: "The decoder maps z to a distribution over x. Generating is two draws: z from the prior N(0, 1), then x from the decoder's distribution. Training (ELBO) learns the decoder and an encoder together.",
        try: "Drag z (the orange handle) through the prior: the decoder's mean switches between the two modes, and its spread changes along the way.",
      },
      infer: {
        observe: "An observed x (right) is mapped by the encoder to a narrow distribution q(z | x) over latent values (left).",
        explain: "Inference runs the other way than generation: the encoder proposes which z could have produced x. Decoding samples of q(z | x) gives reconstructions close to x. Generation itself does not need the encoder.",
        try: "Click a point between the two modes: is it encoded and reconstructed as precisely as a point in a mode?",
      },
    },
  },
  compare: {
    name: "Compare final distributions",
    question: "Where do the errors of a generative model come from?",
    views: {
      final: {
        observe: "Flow matching with an accurate solver matches the target up to sampling noise. With only 4 Euler steps, the same model puts too many samples between the modes. The trained GAN finds both modes but makes each of them too narrow.",
        explain: "The three panels show three sources of error: none (an exact model, integrated accurately), a numerical one (the same model, integrated too coarsely), and a learning one (a network that only saw 1,000 training samples). This compares sources of error, not model families: the exact models know the true target.",
        try: "Go back to Flow matching, open Advanced and increase the Euler steps: the samples between the modes disappear.",
      },
    },
  },
  fm: {
    name: "Flow matching",
    question: "How does a velocity field move noise into data?",
    progress: "Generation progress",
    views: {
      pairs: {
        observe: "Straight training lines between random noise and data samples cross. Near the probe (circle), lines head towards both modes.",
        explain: "For the same input (x, t), the network is asked for very different directions. With a squared-error loss, it learns their average: the velocity field u(x, t).",
        try: "Click a point between the two modes and a point close to one mode. Where do the directions split into two groups?",
      },
      flow: {
        observe: "The sampled paths bend and never cross, although every training line was straight.",
        explain: "Each sample follows the averaged velocity field. Where lines towards both modes meet, the average points in between, so the paths curve.",
        try: "Turn on “Show distribution”: at every t, the samples have the same distribution as the straight training lines.",
      },
    },
  },
};
