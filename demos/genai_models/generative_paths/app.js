// Interface of the demo. Every method uses the same template: method head (name, question,
// badge), one plot, a main action row, a fixed Observe / Explain / Try box (texts in content.js),
// and collapsible "Advanced" and "Diagnostics" panels. Computation is in toy.js.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";

  const N_SAMPLES = 4000; // per setting: histogram and W1
  const N_DRAW = 120; // drawn as paths
  const N_SIMPLE = 20; // drawn as paths when following a sample
  const LECTURE_DIFF_STEPS = [2, 10, 100, 1000];
  const LECTURE_FM_STEPS = [1, 4, 16, 64];
  const CONTENT = globalThis.CONTENT;
  // ?full=1: the playground, with all options and the Advanced and Diagnostics panels open.
  const FULL = new URLSearchParams(window.location.search).get("full") === "1";
  // ?embed=1: one method per lecture section, without tabs; a link opens the whole demo.
  const EMBED = new URLSearchParams(window.location.search).get("embed") === "1";
  const DIFF_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 1000];
  const FM_STEPS = [1, 2, 4, 8, 16, 32, 64];
  const TEMPERATURES = [0.5, 0.75, 1, 1.5, 2];
  const FAMILIES = ["ar", "gan", "vae", "diffusion", "fm", "compare"];
  const MODELS = globalThis.TRAINED_MODELS; // GAN and VAE, exported by tools/train_models.py
  // 2-D autoregressive view: the slider runs over x1, the plot shows x2 in this range.
  const X1 = { lo: -1.5, hi: 11.5, step: 0.1 };
  const X2 = { lo: -1, hi: 11 };

  const DEFAULTS = Object.freeze({
    family: "diffusion",
    seed: 1,
    samplerSeed: 1,
    showDensity: false,
    showDist: false, // "Show distribution" (else: follow a sample)
    ar: { mode: "2d", tau: 1, gauges: false },
    gan: { run: "standard", mode: "generate" },
    vae: { mode: "generate", noise: true },
    diffusion: { mode: "reverse", sampler: "ddpm", steps: 100 },
    fm: { mode: "pairs", steps: 64, field: false },
    compare: { mode: "final" },
  });
  const state = structuredClone(DEFAULTS);
  let cursor = 0; // column index (for AR: token stage; for 2-D AR: x1 grid index)
  let trace = null; // index of the traced sample (among the drawn ones)
  let traceAuto = true; // pick a sample to highlight until the user chooses one
  let view = null;
  let playing = 0;
  // Flow matching: point of the velocity probe. Lines towards both modes pass here, so the
  // directions split into two groups and their average lies in between.
  let fmProbe = { x: 1.5, t: 0.3 };
  let arIndex = 0; // AR 1-D: sample shown step by step
  let arOverrideC = null; // AR 1-D: coarse bin clicked by the user
  let ar2dShown = 0; // AR 2-D: number of samples drawn so far
  let arDone = []; // AR 1-D: values finished so far (shown in grey)
  // AR sequence: the sequence being drawn (index into the samples), how many of its values are
  // drawn, and the finished sequences (shown in grey).
  let seqIndex = 0;
  let seqStep = 0;
  let seqDone = [];
  let seqView = null;
  let ar2dLast = null; // AR 2-D: index of the last single sample
  let runs = { key: "", ends: [] }; // diffusion: endpoints of the traced sample over sampler runs
  let vaeInferX = 6.5; // VAE inference: observed x
  // GAN and VAE: one selected z (draggable) and the samples drawn so far.
  let ganZ = 0.8;
  let ganHistory = [];
  let vaeZ = 0.8;
  let vaeDraws = [];
  let dragging = false;
  const drawRng = Toy.rng(4242);
  const nextNormal = () => Toy.normals(1, drawRng)[0];
  const clampZ = (z) => Math.min(Math.max(z, -3), 3);

  // URL parameters choose the family (and optionally the view) when the demo is embedded.
  {
    const params = new URLSearchParams(window.location.search);
    let family = params.get("family");
    let mode = params.get("mode");
    if (family === "ar2d") {
      family = "ar";
      mode = "2d";
    }
    if (params.get("view") === "compare") family = "compare";
    if (FAMILIES.includes(family)) state.family = family;
    if (state.family === "ar" && ["tokens", "seq", "2d"].includes(mode)) state.ar.mode = mode;
    if (state.family === "vae" && ["generate", "infer"].includes(mode)) state.vae.mode = mode;
    if (["standard", "unstable"].includes(params.get("run"))) state.gan.run = params.get("run");
    if (state.family === "diffusion" && ["reverse", "forward"].includes(mode)) state.diffusion.mode = mode;
    if (state.family === "fm" && ["flow", "pairs"].includes(mode)) state.fm.mode = mode;
  }

  const fixed = (x, d = 2) => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const minus = (s) => String(s).replace("-", "−");
  const num = (x, d = 2) => minus(fixed(x, d));
  const sub = (n) => String(n).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]);

  // ---- simulation (cached) ---------------------------------------------------------------

  // Only the settings that change the simulation are part of the cache key.
  function simulationConfig() {
    const f = state.family;
    const s = state[f];
    const base = { family: f, seed: state.seed, samplerSeed: state.samplerSeed, n: N_SAMPLES };
    if (f === "ar") return { ...base, mode: s.mode, tau: s.tau };
    if (f === "gan") return { ...base, run: s.run };
    if (f === "vae") return { ...base };
    if (f === "compare") return { ...base };
    if (f === "fm") return { ...base, mode: s.mode, steps: s.steps };
    return { ...base, mode: s.mode, sampler: s.sampler, steps: s.steps };
  }

  // Autoregressive views do not transform noise; they keep their own sample data.
  function arView(cfg) {
    if (cfg.mode === "seq") {
      const samples = Toy.seqSample(cfg.n, cfg.seed);
      const last = Float64Array.from(samples, (x) => x[Toy.SEQ.n - 1]);
      return { arseq: true, cols: 1, n: cfg.n, samples, w1: Toy.wasserstein1(last, Toy.TARGET), pos: () => 0 };
    }
    if (cfg.mode === "2d") {
      const s = Toy.ar2dSample(cfg.n, cfg.seed);
      const cols = Math.round((X1.hi - X1.lo) / X1.step) + 1;
      return { ar2d: true, cols, n: cfg.n, x1: s.x1, x2: s.x2, w1: Toy.wasserstein1(s.x1, Toy.TARGET), nfe: 2, deterministic: false, pos: (c) => c / (cols - 1) };
    }
    const s = Toy.arSample(cfg.n, cfg.seed, cfg.tau);
    return { ar: true, cols: 3, n: cfg.n, sample: s, w1: Toy.wasserstein1(s.x, Toy.TARGET), nfe: 2, deterministic: false, pos: (c) => c / 2 };
  }

  // GAN: the generator applied to the same noise at every training snapshot.
  function ganView(cfg) {
    const snaps = MODELS.gan[cfg.run].snapshots;
    const z = Toy.initialNoise(cfg.n, cfg.seed);
    const samples = snaps.map((s) => Toy.ganGenerate(MODELS, s, z));
    return {
      gan: true, cols: snaps.length, n: cfg.n, z, snaps, samples, w1s: samples.map((x) => Toy.wasserstein1(x, Toy.TARGET)),
      colorKey: Float64Array.from(z, (v) => Toy.normalCdf(v)), nfe: 1, deterministic: true, pos: (c) => c / (snaps.length - 1),
    };
  }

  // VAE generation as three stages per sample: z, decoder mean, x (with or without decoder noise).
  function vaeView(cfg) {
    const n = cfg.n;
    const z = Toy.initialNoise(n, cfg.seed);
    const e = Toy.normals(n, Toy.stream(cfg.seed, "sampler"));
    const mean = new Float64Array(n);
    const std = new Float64Array(n);
    z.forEach((v, i) => {
      const d = Toy.vaeDecoder(MODELS, v);
      mean[i] = d.mean;
      std[i] = d.std;
    });
    const stages = (last) => {
      const st = new Float32Array(3 * n);
      st.set(z, 0);
      st.set(mean, n);
      st.set(last, 2 * n);
      return st;
    };
    const withNoise = Float64Array.from(mean, (m, i) => m + std[i] * e[i]);
    return {
      vae: true, cols: 3, n, z, mean, std, pos: (c) => c / 2, nfe: 1,
      colorKey: Float64Array.from(z, (v) => Toy.normalCdf(v)),
      statesNoise: stages(withNoise), statesMean: stages(mean),
      w1Noise: Toy.wasserstein1(withNoise, Toy.TARGET), w1Mean: Toy.wasserstein1(mean, Toy.TARGET),
    };
  }

  const finalColumn = (v) => v.states.subarray((v.cols - 1) * v.n, v.cols * v.n);
  const ideal = (cfg) => (seed) => finalColumn(Toy.simulate({ seed, samplerSeed: 1, n: N_SAMPLES, ...cfg }));
  const ganAt = (run, step) => (seed) => Toy.ganGenerate(MODELS, MODELS.gan[run].snapshots.find((s) => s.step === step), Toy.initialNoise(N_SAMPLES, seed));
  // The Compare view: three settings, three sources of error.
  const PRESETS = [
    { label: "Flow matching, accurate solver", kind: "ideal model, RK4 with 200 steps", sample: (seed) => Toy.fmStates(Toy.initialNoise(N_SAMPLES, seed), 200, "rk4").subarray(200 * N_SAMPLES) },
    { label: "Flow matching, 4 Euler steps", kind: "ideal model, coarse integration", sample: ideal({ family: "fm", mode: "flow", steps: 4 }) },
    { label: "GAN, trained", kind: "trained network", sample: ganAt("standard", 6000) },
  ];
  const presetCache = new Map();
  function presetSamples(k, seed) {
    const key = `${k}:${seed}`;
    if (!presetCache.has(key)) {
      const x = PRESETS[k].sample(seed);
      presetCache.set(key, { ...PRESETS[k], x, w1: Toy.wasserstein1(x, Toy.TARGET) });
    }
    return presetCache.get(key);
  }
  function compareView(cfg) {
    return { compare: true, cols: 1, n: N_SAMPLES, pos: () => 0, panels: PRESETS.map((_, k) => presetSamples(k, cfg.seed)) };
  }

  const cache = new Map();
  function currentView() {
    const cfg = simulationConfig();
    const key = JSON.stringify(cfg);
    if (!cache.has(key)) {
      let v;
      if (cfg.family === "ar") {
        v = arView(cfg);
      } else if (cfg.family === "gan") {
        v = ganView(cfg);
      } else if (cfg.family === "vae") {
        v = vaeView(cfg);
      } else if (cfg.family === "compare") {
        v = compareView(cfg);
      } else {
        v = Toy.simulate(cfg);
        v.w1 = Toy.wasserstein1(v.states.subarray((v.cols - 1) * v.n, v.cols * v.n), Toy.TARGET);
      }
      v.cfg = cfg;
      cache.set(key, v);
      if (cache.size > 8) cache.delete(cache.keys().next().value);
    }
    return cache.get(key);
  }
  const value = (c, i) => view.states[c * view.n + i];
  // Fewer lines for the training pairs, so that their crossings stay visible.
  const drawnCount = () => Math.min(view.straight ? 60 : simple() ? N_SIMPLE : N_DRAW, view.n);

  // Highlight the sample (among the first N_SIMPLE) that ends closest to the centre
  // of the right mode, so that it is visible in both viewing modes.
  function autoTrace() {
    let best = 0;
    for (let i = 1; i < Math.min(N_SIMPLE, view.n); i++) {
      if (Math.abs(value(view.cols - 1, i) - 7) < Math.abs(value(view.cols - 1, best) - 7)) best = i;
    }
    return best;
  }

  // Accurate reference for flow-matching sampling: RK4 with 200 steps from the same noise (per seed).
  const fmReferenceCache = new Map();
  function fmReference() {
    if (!fmReferenceCache.has(state.seed)) {
      const z = Toy.initialNoise(N_SAMPLES, state.seed);
      fmReferenceCache.set(state.seed, Toy.fmStates(z, 200, "rk4").subarray(200 * N_SAMPLES));
    }
    return fmReferenceCache.get(state.seed);
  }

  // Accurate reference for the deterministic diffusion samplers: the probability-flow ODE with
  // 1000 Heun steps from the same starting noise (per seed).
  const referenceCache = new Map();
  function diffusionReference() {
    if (!referenceCache.has(state.seed)) {
      const v = Toy.simulate({ family: "diffusion", mode: "reverse", sampler: "ode", steps: 1000, seed: state.seed, samplerSeed: 1, n: N_SAMPLES });
      referenceCache.set(state.seed, finalColumn(v));
    }
    return referenceCache.get(state.seed);
  }
  const startColumn = () => (view.forward ? view.cols - 1 : 0);
  const endColumn = () => (view.forward ? 0 : view.cols - 1);
  const colToX1 = (c) => X1.lo + c * X1.step;
  const x1ToCol = (x1) => Math.min(Math.max(Math.round((x1 - X1.lo) / X1.step), 0), view.cols - 1);
  // "Follow a sample" (default) or "Show distribution".
  const simple = () => !state.showDist;
  // Texts of the current method and view (content.js).
  const content = () => {
    const c = CONTENT[state.family];
    return { ...c, ...(c.views[state[state.family].mode] || {}) };
  };
  const ganGen = () => state.family === "gan" && state.gan.mode === "generate";
  const vaeSingle = () => state.family === "vae" && state.vae.mode === "generate" && !state.showDist;
  const isTraining = () => Boolean(view.forward || (state.family === "fm" && state.fm.mode === "pairs"));

  // Many training pairs for the flow-matching probe (per seed).
  let probePairs = null;
  function getProbePairs() {
    if (!probePairs || probePairs.seed !== state.seed) probePairs = { seed: state.seed, ...Toy.fmPairs(40000, state.seed) };
    return probePairs;
  }

  // ---- labels ------------------------------------------------------------------------------

  // Text for column c: what the position along the process means for this family.
  function columnLabel(c) {
    if (view.ar) return ["no token yet", "token 1 (c)", "token 2 (f)"][c];
    if (view.ar2d) return `x₁ = ${num(colToX1(c), 1)}`;
    if (view.gan) return `training step ${view.snaps[c].step}`;
    if (view.vae) return ["z ~ N(0, 1)", "decoder μθ(z) ± σθ(z)", state.vae.noise ? "x ~ N(μθ, σθ²)" : "x = μθ(z)"][c];
    if (view.compare) return "";
    const p = view.pos(c);
    if (state.family === "fm") return `t = ${fixed(p)}`;
    const t = 1 - p;
    return t === 1 ? "t = T (noise)" : t === 0 ? "t = 0 (data)" : `t = ${fixed(t)} T`;
  }

  // Colors along the starting value's quantile: purple (low) → teal → mustard (high).
  const STOPS = [[106, 61, 154], [27, 158, 119], [212, 160, 23]];
  function color(q) {
    const s = Math.min(Math.max(q, 0), 1) * (STOPS.length - 1);
    const k = Math.min(Math.floor(s), STOPS.length - 2);
    const f = s - k;
    const c = STOPS[k].map((v, j) => Math.round(v + f * (STOPS[k + 1][j] - v)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  // ---- controls -----------------------------------------------------------------------------

  function fillSelect(id, values, format = String) {
    $(id).replaceChildren(...values.map((v) => new Option(format(v), String(v))));
  }

  let lastFamily = null;
  function syncControls() {
    const c = content();
    const compare = state.family === "compare";
    $("method-name").textContent = c.name;
    $("question").textContent = c.question;
    $("advanced").hidden = compare;
    $("primary-extra").hidden = compare;
    // Method-specific controls marked data-primary stay visible in the main row.
    for (const g of document.querySelectorAll("[data-primary]")) {
      if (!g.placeholder) {
        g.placeholder = document.createComment("primary control");
        g.before(g.placeholder);
      }
      if (g.dataset.for === state.family) $("dist-toggle").before(g);
      else if (g.placeholder.nextSibling !== g) g.placeholder.after(g);
    }
    $("show-dist").checked = state.showDist;
    if (state.family !== lastFamily) {
      $("diagnostics").open = FULL;
      $("advanced").open = FULL;
      lastFamily = state.family;
    }
    // Fewer options by default; all of them in the playground (?full=1).
    const short = !FULL;
    const stepOptions = short && state.family === "diffusion" ? LECTURE_DIFF_STEPS : DIFF_STEPS;
    if ($("diff-steps").options.length !== stepOptions.length) fillSelect("diff-steps", stepOptions);
    const fmOptions = short && state.family === "fm" ? LECTURE_FM_STEPS : FM_STEPS;
    if ($("fm-steps").options.length !== fmOptions.length) fillSelect("fm-steps", fmOptions);
    $("fm-steps-caption").textContent = `${state.fm.steps} Euler step${state.fm.steps > 1 ? "s" : ""} = ${state.fm.steps} model evaluation${state.fm.steps > 1 ? "s" : ""} per sample. With 1 step, every sample lands at the mean of the data.`;
    $("sampler").querySelector('[value="ode"]').hidden = short;
    for (const b of document.querySelectorAll("[data-family]")) {
      const on = b.dataset.family === state.family;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on || (compare && b.dataset.family === "ar") ? 0 : -1;
    }
    $("compare-link").setAttribute("aria-current", String(compare));
    if (EMBED) {
      const link = $("open-full");
      link.hidden = false;
      link.href = compare ? "index.html?view=compare" : `index.html?family=${state.family}`;
    }
    $("family-controls").setAttribute("aria-labelledby", `tab-${state.family}`);
    for (const g of document.querySelectorAll(".group[data-for]")) {
      const f = g.dataset.for;
      g.hidden = f !== state.family ||
        Boolean(g.dataset.mode && g.dataset.mode !== state[f].mode) ||
        Boolean(g.dataset.sampler && g.dataset.sampler !== state[f].sampler);
    }
    for (const seg of document.querySelectorAll(".seg")) {
      const f = seg.closest("[data-for]").dataset.for;
      const key = seg.dataset.key;
      for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.value === state[f][key]));
    }
    $("tau").value = String(state.ar.tau);
    $("vae-noise").checked = state.vae.noise;
    $("sampler").value = state.diffusion.sampler;
    $("diff-steps").value = String(state.diffusion.steps);
    $("fm-steps").value = String(state.fm.steps);
    $("fm-field").checked = state.fm.field;
    $("seed").value = state.seed;
    $("show-density").checked = state.showDensity;
    $("density-group").hidden = ["ar", "gan", "vae", "compare"].includes(state.family);
    // The slider only ever means generation progress (or, in the GAN's training view, training).
    $("progress-row").hidden = ["compare", "vae", "ar"].includes(state.family) || ganGen();
    $("ar-gauges").checked = state.ar.gauges;
    $("dist-toggle").hidden = isTraining() || (state.family === "gan" && !ganGen()) || (state.family === "vae" && state.vae.mode === "infer");
    if (state.showDist) $("vae-primary").hidden = true;
    $("cursor-label").textContent = c.progress || "";
    if (state.family === "diffusion") {
      const s = state.diffusion;
      const calls = s.sampler === "ode" ? 2 * s.steps - 1 : s.steps;
      $("steps-caption").textContent = `${s.steps} steps = ${calls} model evaluations per sample. Fewer steps: faster, but less accurate.`;
    }
    const trained = ["gan", "vae"].includes(state.family);
    const badge = $("badge");
    badge.className = `badge${trained ? " trained" : state.family === "compare" ? " mixed" : ""}`;
    badge.textContent = trained ? "trained network: 1,000 training samples" : state.family === "compare" ? "ideal models and trained networks" : "ideal model: exact formula";
    badge.title = trained
      ? "A small neural network, trained on 1,000 samples of the target (see tools/train_models.py). Its errors are those of real training."
      : state.family === "compare"
        ? "Ideal models know the true target; trained networks only saw 1,000 samples."
        : "The probabilities, the velocity field and the score are computed exactly from the known target distribution. There is no trained network: all differences come from the sampling procedure itself (e.g. too few steps).";
    $("run-count").textContent = `sampler run ${state.samplerSeed}`;
    $("ar2d-one").disabled = ar2dShown >= N_SAMPLES;
    $("ar2d-many").disabled = ar2dShown >= N_SAMPLES;
  }

  // A change of the setting recomputes the view and moves the slider to the end of the process
  // (2-D AR: to x1 = 2, the left mode).
  function changed() {
    stop();
    view = currentView();
    cursor = view.ar2d ? x1ToCol(2) : view.compare ? 0 : endColumn();
    // AR tokens start at the beginning, so that "Next token" advances.
    if (view.ar) cursor = 0;
    // Pick the highlighted sample in a view that shows it (not in the training pairs, whose
    // end points are random data samples rather than results).
    if (view.arseq && view !== seqView) {
      seqView = view;
      seqIndex = 0;
      seqStep = 0;
      seqDone = [];
    }
    if (traceAuto && trace === null && view.states && !view.straight) {
      trace = autoTrace();
      traceAuto = false;
    }
    render();
  }

  function bindEvents() {
    const tabs = [...document.querySelectorAll("[data-family]")];
    for (const b of tabs) {
      b.addEventListener("click", () => {
        state.family = b.dataset.family;
        changed();
      });
      b.addEventListener("keydown", (e) => {
        const i = tabs.indexOf(b);
        const j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
        if (j === null) return;
        const next = tabs[(j + tabs.length) % tabs.length];
        next.focus();
        next.click();
      });
    }
    for (const seg of document.querySelectorAll(".seg")) {
      const f = seg.closest("[data-for]").dataset.for;
      const key = seg.dataset.key;
      for (const b of seg.querySelectorAll("button")) {
        b.addEventListener("click", () => {
          state[f][key] = b.dataset.value;
          changed();
        });
      }
    }
    $("tau").addEventListener("change", (e) => { state.ar.tau = Number(e.target.value); changed(); });
    $("vae-noise").addEventListener("change", (e) => { state.vae.noise = e.target.checked; vaeDraws = []; render(); });
    $("gan-generate").addEventListener("click", () => {
      ganHistory = [...ganHistory, ganZ].slice(-40);
      ganZ = clampZ(nextNormal());
      updateCursor();
    });
    $("vae-again").addEventListener("click", () => {
      const d = Toy.vaeDecoder(MODELS, vaeZ);
      vaeDraws = [...vaeDraws, state.vae.noise ? d.mean + d.std * nextNormal() : d.mean].slice(-40);
      updateCursor();
    });
    $("vae-newz").addEventListener("click", () => {
      vaeZ = clampZ(nextNormal());
      vaeDraws = [];
      updateCursor();
    });
    // Dragging the z handle (GAN: along the z axis; VAE: along the prior on the left).
    const plot = $("plot");
    const dragTo = (e) => {
      const r = plot.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * L.W;
      const y = ((e.clientY - r.top) / r.height) * H;
      if (view.ar2d) cursor = x1ToCol(X1.lo + ((x - L.xM0) / (L.xM1 - L.xM0)) * (X1.hi - X1.lo));
      else if (ganGen()) ganZ = clampZ(Z_LO + ((x - L.xM0) / (L.xM1 - L.xM0)) * (Z_HI - Z_LO));
      else {
        vaeZ = clampZ(unSy(y));
        vaeDraws = [];
      }
      updateCursor();
    };
    plot.addEventListener("pointerdown", (e) => {
      const r = plot.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * L.W;
      const ok = ((ganGen() || view.ar2d) && x >= L.xM0 && x <= L.xM1) || (vaeSingle() && x <= dyn.c0 + 40);
      if (!ok) return;
      dragging = true;
      plot.setPointerCapture(e.pointerId);
      dragTo(e);
    });
    plot.addEventListener("pointermove", (e) => { if (dragging) dragTo(e); });
    plot.addEventListener("pointerup", () => { dragging = false; });
    $("compare-link").addEventListener("click", () => {
      state.family = "compare";
      changed();
    });
    $("ar-next").addEventListener("click", () => {
      stop();
      if (cursor >= 2) {
        arDone = [...arDone, view.sample.x[arIndex]].slice(-40);
        arIndex = (arIndex + 1) % view.n;
        cursor = 1;
      } else {
        cursor += 1;
      }
      arOverrideC = null;
      updateCursor();
    });
    $("seq-next").addEventListener("click", () => {
      if (seqStep >= Toy.SEQ.n) {
        seqDone = [...seqDone, seqIndex].slice(-15);
        seqIndex = (seqIndex + 1) % view.n;
        seqStep = 1;
      } else {
        seqStep += 1;
      }
      updateCursor();
    });
    $("seq-clear").addEventListener("click", () => {
      seqDone = [];
      updateCursor();
    });
    $("ar-gauges").addEventListener("change", (e) => { state.ar.gauges = e.target.checked; render(); });
    $("ar2d-one").addEventListener("click", sampleOne2d);
    $("ar2d-many").addEventListener("click", () => {
      ar2dShown = Math.min(ar2dShown + 100, N_SAMPLES);
      ar2dLast = null;
      render();
    });
    $("ar2d-reset").addEventListener("click", () => {
      ar2dShown = 0;
      ar2dLast = null;
      render();
    });
    $("sampler").addEventListener("change", (e) => { state.diffusion.sampler = e.target.value; changed(); });
    $("diff-steps").addEventListener("change", (e) => { state.diffusion.steps = Number(e.target.value); changed(); });
    $("rerun").addEventListener("click", () => {
      state.samplerSeed += 1;
      changed();
    });
    $("fm-steps").addEventListener("change", (e) => { state.fm.steps = Number(e.target.value); changed(); });
    $("fm-resample").addEventListener("click", () => {
      state.seed = (state.seed % 99999) + 1;
      changed();
    });
    $("fm-field").addEventListener("change", (e) => { state.fm.field = e.target.checked; render(); });
    $("seed").addEventListener("change", (e) => {
      const v = Math.round(Number(e.target.value));
      state.seed = Number.isFinite(v) && v >= 1 ? Math.min(v, 99999) : DEFAULTS.seed;
      changed();
    });
    $("new-seed").addEventListener("click", () => {
      state.seed = (state.seed % 99999) + 1;
      changed();
    });
    $("show-density").addEventListener("change", (e) => { state.showDensity = e.target.checked; render(); });
    $("show-dist").addEventListener("change", (e) => { state.showDist = e.target.checked; render(); });
    $("cursor").addEventListener("input", (e) => {
      stop();
      cursor = Number(e.target.value);
      updateCursor();
    });
    $("play").addEventListener("click", () => (playing ? stop() : play()));
    $("plot").addEventListener("click", pick);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && trace !== null) {
        trace = null;
        render();
      }
    });
    $("trace").addEventListener("click", (e) => {
      if (e.target.closest("button")) {
        trace = null;
        render();
      }
    });
  }

  // Play: move the slider from the noise end to the data end (GAN: through the training) in a
  // few seconds.
  function play() {
    const from = startColumn();
    const to = endColumn();
    if (cursor === to) cursor = from;
    const startPos = cursor;
    const t0 = performance.now();
    const total = view.gan ? 6000 : 3000;
    const duration = total * Math.abs(to - startPos) / Math.max(Math.abs(to - from), 1);
    $("play").textContent = "❚❚ Pause";
    const tick = (now) => {
      const f = Math.min((now - t0) / Math.max(duration, 1), 1);
      // GAN snapshots: one after the other, not a smooth movement.
      cursor = view.gan ? Math.min(startPos + Math.floor(f * (to - startPos + 0.999)), to) : Math.round(startPos + f * (to - startPos));
      updateCursor();
      if (f < 1) playing = requestAnimationFrame(tick);
      else stop();
    };
    playing = requestAnimationFrame(tick);
  }
  function stop() {
    if (playing) cancelAnimationFrame(playing);
    playing = 0;
    $("play").textContent = "▶ Play";
  }

  // 2-D AR: draw one more sample: first x1 (the slider moves there), then x2 given x1.
  function sampleOne2d() {
    if (ar2dShown >= N_SAMPLES) return;
    stop();
    const i = ar2dShown;
    const target = x1ToCol(view.x1[i]);
    const startPos = cursor;
    const t0 = performance.now();
    const tick = (now) => {
      const f = Math.min((now - t0) / 500, 1);
      cursor = Math.round(startPos + f * (target - startPos));
      updateCursor();
      if (f < 1) {
        playing = requestAnimationFrame(tick);
      } else {
        playing = 0;
        ar2dShown = i + 1;
        ar2dLast = i;
        render();
      }
    };
    playing = requestAnimationFrame(tick);
  }

  // ---- plot geometry ---------------------------------------------------------------------------

  const H = 434;
  const M = { top: 54, bottom: 34, left: 6, right: 6 };
  const PH = H - M.top - M.bottom;
  const DMAX = 0.5; // density scale of the side strips
  let L = null; // layout in SVG pixels

  function layout(W) {
    const ls = Math.round(Math.min(Math.max(W * 0.08, 34), 70));
    const rs = Math.round(Math.min(Math.max(W * 0.17, 70), 150));
    const xL1 = M.left + ls;
    const xM0 = xL1 + 32;
    const xR1 = W - M.right;
    const xR0 = xR1 - rs;
    const xM1 = xR0 - 14;
    return { W, xL0: M.left, xL1, xM0, xM1, xR0, xR1, narrow: W < 700 };
  }
  const sy = (y) => M.top + ((Toy.Y_MAX - y) / (Toy.Y_MAX - Toy.Y_MIN)) * PH;
  const unSy = (py) => Toy.Y_MAX - ((py - M.top) / PH) * (Toy.Y_MAX - Toy.Y_MIN);
  const sxCol = (c) => L.xM0 + view.pos(c) * (L.xM1 - L.xM0);
  const sxT = (p) => L.xM0 + p * (L.xM1 - L.xM0); // position in [0, 1] → pixels
  const rx = (d, max = DMAX) => L.xR0 + (Math.min(d, max) / max) * (L.xR1 - L.xR0);

  function el(name, attrs, parent) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.append(node);
    return node;
  }
  function text(parent, x, y, s, attrs = {}) {
    const t = el("text", { x, y, ...attrs }, parent);
    t.textContent = s;
    return t;
  }
  function arrowMarker(defs, id, fill) {
    const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill }, m);
  }
  function titles(svg, left, middle, right) {
    text(svg, (L.xL0 + L.xL1) / 2, 16, left, { "text-anchor": "middle", class: "title" });
    text(svg, (L.xM0 + L.xM1) / 2, L.narrow ? 34 : 16, middle, { "text-anchor": "middle", class: "title" });
    text(svg, (L.xR0 + L.xR1) / 2, 16, right, { "text-anchor": "middle", class: "title" });
  }
  // Density of a mixture along the value axis, as polyline points in a side strip.
  function stripPoints(mix, toX, lo = Toy.Y_MIN, hi = Toy.Y_MAX, toY = sy) {
    const pts = [];
    for (let y = lo; y <= hi + 1e-9; y += 0.05) pts.push(`${toX(Toy.pdf(y, mix)).toFixed(1)},${toY(y).toFixed(1)}`);
    return pts;
  }
  function targetStrip(svg, max = DMAX) {
    el("line", { x1: L.xR0, x2: L.xR0, y1: M.top, y2: M.top + PH, stroke: "#9aa1ab" }, svg);
    const pts = stripPoints(Toy.TARGET, (d) => rx(d, max));
    el("polygon", { points: `${L.xR0},${sy(Toy.Y_MIN)} ${pts.join(" ")} ${L.xR0},${sy(Toy.Y_MAX)}`, fill: "var(--target-fill)", "fill-opacity": 0.8 }, svg);
    const hist = el("g", {}, svg);
    el("polyline", { points: pts.join(" "), fill: "none", stroke: "var(--target)", "stroke-width": 1.8 }, svg);
    return hist;
  }
  function drawHistogram(group, values, lo, hi, bins, toY = sy, max = DMAX) {
    group.replaceChildren();
    for (const b of Toy.histogram(values, lo, hi, bins)) {
      if (b.density <= 0) continue;
      el("rect", {
        x: L.xR0, y: toY(b.y1) + 0.5, width: rx(b.density, max) - L.xR0, height: Math.max(toY(b.y0) - toY(b.y1) - 1, 0.5),
        fill: "#c9ccd2", "fill-opacity": 0.85, stroke: "var(--hist)", "stroke-width": 0.6,
      }, group);
    }
  }
  function valueAxis(svg, x0 = L.xM0, x1 = L.xM1) {
    el("rect", { x: x0, y: M.top, width: x1 - x0, height: PH, fill: "#fff", stroke: "#e2e5ea" }, svg);
    for (let y = -4; y <= 10; y += 2) {
      el("line", { x1: x0, x2: x1, y1: sy(y), y2: sy(y), stroke: "#eef0f3" }, svg);
      text(svg, L.xM0 - 6, sy(y) + 4, minus(y), { "text-anchor": "end" });
    }
  }

  // Columns that are drawn: all of them when there are few, otherwise about one per 2 pixels.
  function drawnColumns() {
    if (view.straight) return [0, view.cols - 1];
    const stride = Math.max(1, Math.ceil(view.cols / ((L.xM1 - L.xM0) / 2)));
    const cols = [];
    for (let c = 0; c < view.cols; c += stride) cols.push(c);
    if (cols[cols.length - 1] !== view.cols - 1) cols.push(view.cols - 1);
    return cols;
  }
  function pathPoints(i, cols) {
    return cols.map((c) => `${sxCol(c).toFixed(1)},${sy(value(c, i)).toFixed(1)}`).join(" ");
  }

  // Density as a heat map (white → blue). `columnFn(p)` gives the mixture at position p in [0, 1]
  // or null; values are normalized per column.
  function heatImage(columnFn) {
    const w = 240;
    const h = 200;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    const column = (px, mix) => {
      const vals = new Float64Array(h);
      let max = 0;
      for (let r = 0; r < h; r++) {
        vals[r] = Toy.pdf(Toy.Y_MAX - ((r + 0.5) / h) * (Toy.Y_MAX - Toy.Y_MIN), mix);
        max = Math.max(max, vals[r]);
      }
      for (let r = 0; r < h; r++) {
        const o = (r * w + px) * 4;
        img.data[o] = 31;
        img.data[o + 1] = 78;
        img.data[o + 2] = 156;
        img.data[o + 3] = Math.round(150 * Math.pow(vals[r] / max, 0.6));
      }
    };
    for (let px = 0; px < w; px++) column(px, columnFn((px + 0.5) / w));
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL();
  }

  // ---- drawing ----------------------------------------------------------------------------------

  let dyn = null; // elements updated when only the slider moves

  function render() {
    syncControls();
    const svg = $("plot");
    svg.replaceChildren();
    const W = Math.round(Math.min(Math.max(svg.clientWidth || svg.parentElement.clientWidth, 340), 1100));
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    L = layout(W);
    if (view.vae) {
      view.states = state.vae.noise ? view.statesNoise : view.statesMean;
      view.w1 = state.vae.noise ? view.w1Noise : view.w1Mean;
      view.deterministic = !state.vae.noise;
    }
    svg.classList.toggle("drag", ganGen() || vaeSingle() || Boolean(view.ar2d));
    if (view.arseq) renderSeq(svg);
    else if (view.ar) renderTokens(svg);
    else if (view.ar2d) render2d(svg);
    else if (view.gan) renderGan(svg);
    else if (view.compare) renderCompare(svg);
    else if (view.vae && state.vae.mode === "infer") renderVaeInfer(svg);
    else if (vaeSingle()) renderVaeSingle(svg);
    else renderPaths(svg);
    renderLegend();
    plotDescription();
    renderGuide();
    updateCursor();
  }

  // Fixed Observe / Explain / Try box and the Diagnostics list.
  function renderGuide() {
    const c = content();
    $("guide").innerHTML = [["Observe", c.observe], ["Explain", c.explain], ["Try", c.try]]
      .map(([k, v]) => `<dt>${k.toUpperCase()}</dt><dd>${v}</dd>`).join("");
    const items = [];
    if (view.compare) {
      for (const p of view.panels) items.push(`${p.label} (${p.kind}): W<sub>1</sub> = ${fixed(p.w1, 3)}.`);
      items.push("Each panel uses 4,000 samples; below ≈ 0.1 is sampling noise.");
    } else if (isTraining()) {
      items.push("Training view: no samples are generated, so there is no distance to the target.");
    } else {
      const w1 = view.gan ? view.w1s[cursor] : view.w1;
      items.push(`Distance of the ${view.gan && !ganGen() ? "samples at this training step" : "finished samples"} to the target: W<sub>1</sub> = ${fixed(w1, 3)} (4,000 samples; below ≈ 0.1 is sampling noise).`);
    }
    if (state.family === "ar") {
      items.push(view.arseq
        ? "Randomness: one random draw per value. The W<sub>1</sub> above is for the last value x₈, whose distribution is the target."
        : view.ar
          ? "Randomness: one random draw per token. The W<sub>1</sub> above includes the 0.1 resolution of the tokens."
          : "Randomness: one random draw per variable. The W<sub>1</sub> above is for x₁ only.");
    }
    if (state.family === "gan") {
      items.push("Randomness: the initial draw only; the generator is deterministic.");
      if (ganGen()) items.push("The trained generator finds both modes with about the right weights, but each mode is too narrow: G is almost flat over ranges of z. See Advanced → Training dynamics.");
    }
    if (state.family === "vae" && state.vae.mode === "generate") {
      items.push(state.vae.noise ? "Randomness: the initial draw of z and the decoder's own draw of x." : "Randomness: the initial draw of z only (decoder mean).");
      items.push(`W<sub>1</sub> with decoder noise ${fixed(view.w1Noise, 3)}, decoder mean only ${fixed(view.w1Mean, 3)}: the mean alone already places the samples in both modes.`);
    }
    if (state.family === "fm" && !isTraining()) {
      items.push("Randomness: the initial draw only; the sampler itself is deterministic.");
      const err = Toy.referenceError(finalColumn(view), fmReference());
      items.push(`Error against an accurate reference (same start, RK4 with 200 steps): mean |x − x<sub>ref</sub>| = ${fixed(err, 3)} per sample.`);
    }
    if (state.family === "diffusion" && !isTraining()) {
      const s = state.diffusion;
      items.push(s.sampler === "ddpm"
        ? "Randomness: the initial draw <em>and</em> fresh noise in every sampling step."
        : "Randomness: the initial draw only; the sampler itself is deterministic.");
      if (s.sampler !== "ddpm") {
        const err = Toy.referenceError(finalColumn(view), diffusionReference());
        items.push(`Error against an accurate reference (same start, 1000-step ODE): mean |x − x<sub>ref</sub>| = ${fixed(err, 3)} per sample.`);
      } else {
        items.push("The stochastic sampler cannot be compared sample by sample with a reference; compare the distributions (W<sub>1</sub>) instead.");
      }
    }
    $("diag-list").innerHTML = items.map((t) => `<li>${t}</li>`).join("");
  }

  // Diffusion, flow matching (and the VAE's stages): paths through the generative process.
  function renderPaths(svg) {
    const defs = el("defs", {}, svg);
    const clip = el("clipPath", { id: "revealed" }, defs);
    const clipRect = el("rect", { x: L.xM0 - 2, y: 0, width: 0, height: H }, clip);
    const panel = el("clipPath", { id: "panel" }, defs);
    el("rect", { x: L.xM0 - 4, y: M.top - 4, width: L.xM1 - L.xM0 + 8, height: PH + 8 }, panel);
    arrowMarker(defs, "arrow-muted", "#8a919c");
    arrowMarker(defs, "arrow-noise", "#d55e00");
    arrowMarker(defs, "arrow-target", "#1f4e9c");

    const title = view.vae
      ? (L.narrow ? "decoder" : "generation: z → decoder → x (three stages, no transport)")
      : view.forward
      ? (L.narrow ? "forward noising ←" : "forward noising (training): data → noise, read right to left")
      : state.family === "fm" && state.fm.mode === "pairs"
        ? (L.narrow ? "training pairs" : "training pairs: straight lines between random noise and data samples")
        : "generative process";
    titles(svg, "noise", title, view.straight ? "data" : simple() || L.narrow ? "result" : "samples vs. target");
    valueAxis(svg);

    // Only the continuous processes (diffusion, flow matching) have a density along the process.
    if (state.showDensity && view.density) {
      const href = heatImage(view.density);
      el("image", { href, x: L.xM0, y: M.top, width: L.xM1 - L.xM0, height: PH, preserveAspectRatio: "none" }, svg);
    }
    if (state.family === "fm" && state.fm.field) drawField(svg);

    // Axis labels of the middle panel.
    for (const c of [0, view.cols - 1]) {
      text(svg, sxCol(c), H - 14, columnLabel(c), { "text-anchor": c === 0 ? "start" : "end" });
    }
    if (view.vae) text(svg, sxCol(1), H - 14, columnLabel(1), { "text-anchor": "middle" });

    // Paths: faded everywhere, strong in the revealed part (up to the slider).
    const cols = drawnColumns();
    const drawn = drawnCount();
    const emphasis = trace !== null || (state.family === "fm" && state.fm.mode === "pairs");
    const faded = el("g", { "clip-path": "url(#panel)", fill: "none", "stroke-width": 1.1, "stroke-opacity": 0.16 }, svg);
    const strong = el("g", { "clip-path": "url(#revealed)", fill: "none", "stroke-width": 1.3, "stroke-opacity": emphasis ? 0.25 : 0.75 }, svg);
    // VAE: dashed connections and a dot per stage, because a stage is one function evaluation,
    // not a movement.
    const stages = Boolean(view.vae);
    const lineAttrs = stages ? { "stroke-dasharray": "2 3" } : {};
    for (let i = 0; i < drawn; i++) {
      const pts = pathPoints(i, cols);
      const col = color(view.colorKey[i]);
      el("polyline", { points: pts, stroke: col, "stroke-linejoin": "round", ...lineAttrs }, faded);
      el("polyline", { points: pts, stroke: col, "stroke-linejoin": "round", ...lineAttrs }, strong);
      if (stages) for (const c of cols) el("circle", { cx: sxCol(c), cy: sy(value(c, i)), r: 1.8, fill: col, stroke: "none" }, strong);
      // VAE: the decoder's output is a distribution, N(mean, std^2): drawn as an error bar.
      if (view.vae) el("line", { x1: sxCol(1), x2: sxCol(1), y1: sy(view.mean[i] - view.std[i]), y2: sy(view.mean[i] + view.std[i]), stroke: col, "stroke-width": 1, "stroke-opacity": 0.8 }, strong);
    }
    const showTrace = trace !== null && !view.straight;
    if (showTrace) {
      // The highlighted sample is revealed up to the slider, only faintly beyond.
      const pts = pathPoints(trace, cols);
      el("polyline", { points: pts, fill: "none", stroke: "#111", "stroke-width": 1.2, "stroke-opacity": 0.2, "clip-path": "url(#panel)" }, svg);
      el("polyline", { points: pts, fill: "none", stroke: "#fff", "stroke-width": 5, "clip-path": "url(#revealed)" }, svg);
      el("polyline", { points: pts, fill: "none", stroke: "#111", "stroke-width": 2.2, "clip-path": "url(#revealed)" }, svg);
    }
    if (state.family === "fm" && state.fm.mode === "pairs") drawProbe(svg);

    // Left strip: noise density, pointing left from the panel.
    const nx = (d) => L.xL1 - (Math.min(d, DMAX) / DMAX) * (L.xL1 - L.xL0);
    const noisePts = stripPoints(Toy.NOISE, nx);
    el("polygon", { points: `${L.xL1},${sy(Toy.Y_MIN)} ${noisePts.join(" ")} ${L.xL1},${sy(Toy.Y_MAX)}`, fill: "var(--noise-fill)", stroke: "var(--noise)", "stroke-width": 1.5 }, svg);
    el("line", { x1: L.xL1, x2: L.xL1, y1: M.top, y2: M.top + PH, stroke: "#9aa1ab" }, svg);
    if (!view.forward) {
      for (let i = 0; i < drawn; i++) el("circle", { cx: L.xL1 - 3, cy: sy(value(0, i)), r: 1.8, fill: color(view.colorKey[i]) }, svg);
    }

    // Right strip: the target is static, the histogram follows the slider.
    const hist = targetStrip(svg);
    const marg = el("polyline", { fill: "none", stroke: "var(--target)", "stroke-width": 1.6, "stroke-dasharray": "4 3" }, svg);

    // Slider: cursor line, markers, label.
    const cursorLine = el("line", { y1: M.top, y2: M.top + PH, stroke: "#111", "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
    const markers = el("g", {}, svg);
    const cursorText = text(svg, 0, M.top - 8, "", { "text-anchor": "middle" });
    dyn = { clipRect, hist, marg, cursorLine, markers, cursorText, drawn };
  }

  // Flow matching: arrows of the (exact) velocity field u(x, t), all of the same length.
  function drawField(svg) {
    const g = el("g", { stroke: "#8a919c", "stroke-width": 1.2, "marker-end": "url(#arrow-muted)", opacity: 0.8 }, svg);
    const scaleY = PH / (Toy.Y_MAX - Toy.Y_MIN);
    const scaleT = L.xM1 - L.xM0;
    for (let t = 0.05; t < 1; t += 0.1) {
      for (let x = -3; x <= 10; x += 1) {
        const u = Toy.fmVelocity(x, t);
        const dx = scaleT;
        const dy = -u * scaleY;
        const len = Math.hypot(dx, dy);
        const x0 = sxT(t);
        const y0 = sy(x);
        el("line", { x1: x0, y1: y0, x2: x0 + (dx / len) * 13, y2: y0 + (dy / len) * 13 }, g);
      }
    }
  }

  // Flow matching, training pairs: training lines that pass near the probe point, their average
  // direction (orange) and the exact velocity u(x, t) (blue, dashed).
  function drawProbe(svg) {
    const pairs = getProbePairs();
    const { x, t } = fmProbe;
    const g = el("g", { "clip-path": "url(#panel)" }, svg);
    let shown = 0;
    for (let i = 0; i < pairs.x0.length && shown < 30; i++) {
      if (Math.abs((1 - t) * pairs.x0[i] + t * pairs.x1[i] - x) < 0.1) {
        el("line", { x1: sxT(0), y1: sy(pairs.x0[i]), x2: sxT(1), y2: sy(pairs.x1[i]), stroke: "#333", "stroke-width": 1, "stroke-opacity": 0.45 }, g);
        shown++;
      }
    }
    const avg = Toy.fmPairAverage(pairs, x, t, 0.1);
    const u = Toy.fmVelocity(x, t);
    const dt = 0.16;
    el("line", { x1: sxT(t), y1: sy(x), x2: sxT(t + dt), y2: sy(x + dt * avg.mean), stroke: "var(--noise)", "stroke-width": 3, "marker-end": "url(#arrow-noise)" }, g);
    el("line", { x1: sxT(t), y1: sy(x), x2: sxT(t + dt), y2: sy(x + dt * u), stroke: "var(--target)", "stroke-width": 2, "stroke-dasharray": "4 3", "marker-end": "url(#arrow-target)" }, g);
    el("circle", { cx: sxT(t), cy: sy(x), r: 5, fill: "#fff", stroke: "#111", "stroke-width": 2 }, svg);
    drawProbeInset(avg, u);
  }

  function drawProbeInset(avg, u) {
    const svg = $("fm-probe");
    svg.replaceChildren();
    const w = 280;
    const h = 200;
    const m = { l: 10, r: 10, t: 10, b: 34 };
    const lo = -4;
    const hi = 13;
    const px = (v) => m.l + ((v - lo) / (hi - lo)) * (w - m.l - m.r);
    const bins = Toy.histogram(avg.directions, lo, hi, 34);
    const maxD = Math.max(...bins.map((b) => b.density), 1e-9);
    const py = (d) => h - m.b - (d / maxD) * (h - m.t - m.b);
    el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b, fill: "#fff", stroke: "#e2e5ea" }, svg);
    for (const b of bins) {
      if (b.density > 0) el("rect", { x: px(b.y0) + 0.5, y: py(b.density), width: Math.max(px(b.y1) - px(b.y0) - 1, 1), height: h - m.b - py(b.density), fill: "#c9ccd2", stroke: "var(--hist)", "stroke-width": 0.6 }, svg);
    }
    for (let v = -4; v <= 12; v += 4) text(svg, px(v), h - m.b + 14, minus(v), { "text-anchor": "middle" });
    text(svg, w / 2, h - 3, "direction x₁ − x₀ of the nearby training lines", { "text-anchor": "middle" });
    if (Number.isFinite(avg.mean)) el("line", { x1: px(avg.mean), x2: px(avg.mean), y1: m.t, y2: h - m.b, stroke: "var(--noise)", "stroke-width": 3 }, svg);
    el("line", { x1: px(u), x2: px(u), y1: m.t, y2: h - m.b, stroke: "var(--target)", "stroke-width": 2, "stroke-dasharray": "4 3" }, svg);
    const d = avg.directions;
    const range = d.length ? `from ${num(Math.min(...d), 1)} to ${num(Math.max(...d), 1)}` : "";
    $("fm-probe-desc").innerHTML = avg.count
      ? `${avg.count.toLocaleString("en-US")} of 40,000 random training pairs pass within ±0.1 of x = ${num(fmProbe.x)} at t = ${fixed(fmProbe.t)}. Their directions x₁ − x₀ range ${range}: for the same input (x, t), the network is asked for very different outputs. With the squared-error loss it learns their <strong>average</strong> <span style="color: var(--noise)">${num(avg.mean)}</span>, which matches the exact velocity <span style="color: var(--target)">u(x, t) = ${num(u)}</span>. The average can be a direction that hardly any single training line has.`
      : `No training pair passes near x = ${num(fmProbe.x)} at t = ${fixed(fmProbe.t)}: the data never go there, so the velocity there is not learned from data.`;
  }

  // ---- autoregressive, 1-D: two tokens -----------------------------------------------------------

  function renderTokens(svg) {
    const gaugesOn = state.ar.gauges;
    titles(svg, gaugesOn ? "u" : "", L.narrow ? "two tokens" : "token 1: c ~ p(c)   →   token 2: f ~ p(f | c)", simple() || L.narrow ? "result" : "samples vs. target");
    const wM = L.xM1 - L.xM0;
    const A0 = L.xM0;
    const A1 = L.xM0 + wM * 0.42;
    const B0 = L.xM0 + wM * 0.58;
    const B1 = L.xM1;
    valueAxis(svg, A0, A1);
    el("rect", { x: B0, y: M.top, width: B1 - B0, height: PH, fill: "#fff", stroke: "#e2e5ea" }, svg);
    text(svg, A0, H - 14, L.narrow ? "token 1" : "token 1: coarse value c", { "text-anchor": "start" });
    text(svg, B1, H - 14, L.narrow ? "token 2 (zoom)" : "token 2: fine value f (zoomed in)", { "text-anchor": "end" });
    // Random numbers: two gauges from 0 (bottom) to 1 (top).
    const gw = (L.xL1 - L.xL0 - 10) / 2;
    const gauges = [L.xL0 + 2, L.xL0 + 6 + gw].map((x) => ({ x, w: gw }));
    if (gaugesOn) gauges.forEach((g, j) => text(svg, g.x + g.w / 2, H - 14, `u${sub(j + 1)}`, { "text-anchor": "middle" }));
    const hist = targetStrip(svg);
    const groups = {
      wedge: el("g", {}, svg), barsA: el("g", {}, svg), barsB: el("g", {}, svg), gauge: el("g", {}, svg), marks: el("g", {}, svg),
    };
    dyn = { hist, A0, A1, B0, B1, gauges, gaugesOn, ...groups };
  }

  function updateTokens() {
    const s = view.sample;
    const t = s.tables;
    const i = arIndex;
    const stage = cursor;
    const c = s.c[i];
    const f = s.f[i];
    const shownC = arOverrideC ?? (stage >= 1 ? c : null);
    const { A0, A1, B0, B1 } = dyn;
    const showLabels = A1 - A0 > 120;

    // Token 1: p(c) as bars along the value axis.
    dyn.barsA.replaceChildren();
    const maxPc = Math.max(...t.pc);
    for (let cc = 0; cc < Toy.AR.coarse; cc++) {
      const y0 = sy(Toy.arEdge(cc + 1));
      const y1 = sy(Toy.arEdge(cc));
      const w = (t.pc[cc] / maxPc) * (A1 - A0 - (showLabels ? 34 : 4));
      const chosen = stage >= 1 && cc === c && arOverrideC === null;
      el("rect", {
        x: A0 + 1, y: y0 + 1, width: Math.max(w, 0.5), height: y1 - y0 - 2,
        fill: chosen ? "var(--target)" : cc === arOverrideC ? "#9db3db" : "#c7d3ea", stroke: "var(--target)", "stroke-width": cc === shownC ? 1.5 : 0.5,
      }, dyn.barsA);
      if (showLabels && t.pc[cc] >= 0.001) text(dyn.barsA, A0 + w + 4, (y0 + y1) / 2 + 4, `${fixed(100 * t.pc[cc], t.pc[cc] < 0.01 ? 1 : 0)}%`, {});
    }

    // Token 2: p(f | c) for the chosen (or clicked) coarse bin, zoomed to the full height.
    dyn.wedge.replaceChildren();
    dyn.barsB.replaceChildren();
    const fy = (ff) => M.top + PH - ((ff + 1) / Toy.AR.fine) * PH;
    if (shownC === null) {
      text(dyn.barsB, (B0 + B1) / 2, M.top + PH / 2, "waits for token 1", { "text-anchor": "middle" });
    } else {
      el("polygon", {
        points: `${A1},${sy(Toy.arEdge(shownC + 1))} ${B0},${M.top} ${B0},${M.top + PH} ${A1},${sy(Toy.arEdge(shownC))}`,
        fill: "#e8edf6", stroke: "#c7d3ea",
      }, dyn.wedge);
      const pf = t.pf[shownC];
      const maxPf = Math.max(...pf);
      const showB = B1 - B0 > 110;
      for (let ff = 0; ff < Toy.AR.fine; ff++) {
        const y0 = fy(ff);
        const hgt = PH / Toy.AR.fine;
        const w = (pf[ff] / maxPf) * (B1 - B0 - (showB ? 92 : 4));
        const chosen = stage === 2 && arOverrideC === null && ff === f;
        el("rect", { x: B0 + 1, y: y0 + 1.5, width: Math.max(w, 0.5), height: hgt - 3, fill: chosen ? "var(--target)" : "#c7d3ea", stroke: "var(--target)", "stroke-width": 0.5 }, dyn.barsB);
        if (showB) text(dyn.barsB, B0 + w + 4, y0 + hgt / 2 + 4, `x ≈ ${num(Toy.arValue(shownC, ff))}: ${fixed(100 * pf[ff], 0)}%`, {});
      }
    }

    // Random numbers: each gauge is cut into the cumulative probabilities of its token; the
    // segment that u falls into is the chosen token (inverse-cdf sampling).
    dyn.gauge.replaceChildren();
    const gy = (u) => M.top + PH - u * PH;
    const gauge = (g, p, u, chosenIdx) => {
      let acc = 0;
      p.forEach((pk, k) => {
        el("rect", { x: g.x, y: gy(acc + pk), width: g.w, height: Math.max(gy(acc) - gy(acc + pk), 0), fill: k === chosenIdx ? "#9db3db" : k % 2 ? "#eef0f3" : "#dfe3e9" }, dyn.gauge);
        acc += pk;
      });
      el("rect", { x: g.x, y: M.top, width: g.w, height: PH, fill: "none", stroke: "#9aa1ab" }, dyn.gauge);
      if (u !== null) {
        el("line", { x1: g.x - 2, x2: g.x + g.w + 2, y1: gy(u), y2: gy(u), stroke: "#111", "stroke-width": 2 }, dyn.gauge);
      }
    };
    if (dyn.gaugesOn) gauge(dyn.gauges[0], t.pc, stage >= 1 && arOverrideC === null ? s.u1[i] : null, stage >= 1 && arOverrideC === null ? c : null);
    if (!dyn.gaugesOn) { /* random numbers hidden */ } else if (shownC !== null) gauge(dyn.gauges[1], t.pf[shownC], stage === 2 && arOverrideC === null ? s.u2[i] : null, stage === 2 && arOverrideC === null ? f : null);
    else el("rect", { x: dyn.gauges[1].x, y: M.top, width: dyn.gauges[1].w, height: PH, fill: "none", stroke: "#d5d9e0" }, dyn.gauge);

    // Right strip: all samples after the current token (coarse values after token 1).
    dyn.marks.replaceChildren();
    $("ar-next").textContent = stage >= 2 ? "Next token (new value)" : "Next token";
    if (simple()) {
      // One value at a time: earlier values in grey, the current one in black.
      dyn.hist.replaceChildren();
      for (const x of arDone) el("circle", { cx: L.xR0 + 5, cy: sy(x), r: 3.5, fill: "#666" }, dyn.marks);
    } else if (stage === 0) dyn.hist.replaceChildren();
    else if (stage === 1) drawHistogram(dyn.hist, s.coarse, Toy.AR.lo, Toy.AR.lo + Toy.AR.coarse, Toy.AR.coarse);
    else if (stage === 2) drawHistogram(dyn.hist, s.x, Toy.AR.lo, Toy.AR.lo + Toy.AR.coarse, 60);
    if (stage >= 1 && arOverrideC === null) {
      const y = stage === 2 ? sy(s.x[i]) : sy(s.coarse[i]);
      el("circle", { cx: L.xR0, cy: y, r: 4.5, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, dyn.marks);
    }
  }

  // ---- autoregressive, sequence: x1 → x2 → … → x8 ------------------------------------------------

  function renderSeq(svg) {
    titles(svg, "", L.narrow ? "x₁ → … → x₈" : "x₁ → x₂ → … → x₈: each value drawn given all previous ones", simple() || L.narrow ? "result" : "x₈ vs. target");
    valueAxis(svg);
    const n = Toy.SEQ.n;
    const step = (L.xM1 - L.xM0) / n;
    const px = (k) => L.xM0 + (k + 0.5) * step;
    for (let k = 0; k < n; k++) {
      el("line", { x1: px(k), x2: px(k), y1: M.top, y2: M.top + PH, stroke: "#eef0f3" }, svg);
      text(svg, px(k), H - 14, `x${sub(k + 1)}`, { "text-anchor": "middle" });
    }
    const hist = targetStrip(svg);
    dyn = { px, step, hist, g: el("g", {}, svg) };
  }

  // A density drawn sideways from column x (density times `scale` pixels), only where it is
  // visibly above zero, so that columns do not turn into long vertical lines.
  function sideDensity(group, x, mix, scale, attrs) {
    const ys = [];
    for (let y = Toy.Y_MIN; y <= Toy.Y_MAX + 1e-9; y += 0.04) ys.push(y);
    const w = ys.map((y) => Toy.pdf(y, mix) * scale);
    const first = w.findIndex((v) => v > 0.4);
    const last = w.length - 1 - [...w].reverse().findIndex((v) => v > 0.4);
    if (first < 0) return;
    const pts = [];
    for (let i = first; i <= last; i++) pts.push(`${(x + w[i]).toFixed(1)},${sy(ys[i]).toFixed(1)}`);
    if (attrs.fill && attrs.fill !== "none") {
      el("polygon", { points: `${x},${sy(ys[first]).toFixed(1)} ${pts.join(" ")} ${x},${sy(ys[last]).toFixed(1)}`, ...attrs }, group);
    } else {
      el("polyline", { points: pts.join(" "), fill: "none", ...attrs }, group);
    }
  }

  function updateSeq() {
    const { px, step, g } = dyn;
    g.replaceChildren();
    const n = Toy.SEQ.n;
    const cur = view.samples[seqIndex];
    const k = seqStep;
    const line = (xs, len) => Array.from(xs).slice(0, len).map((v, j) => `${px(j).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
    // Many sequences, the histogram of their last values and the plain distribution of each position.
    if (!simple()) {
      for (let i = 0; i < N_DRAW; i++) {
        const xs = view.samples[i];
        el("polyline", { points: line(xs, n), fill: "none", stroke: color(Toy.cdf(xs[n - 1], Toy.TARGET)), "stroke-opacity": 0.35 }, g);
      }
      for (let j = 0; j < n; j++) {
        const mix = Toy.seqMarginal(j);
        sideDensity(g, px(j), mix, (step * 0.8) / Math.max(...mix.means.map((m) => Toy.pdf(m, mix))), { stroke: "#6b7280", "stroke-width": 1.2, "stroke-dasharray": "4 3" });
      }
      drawHistogram(dyn.hist, Float64Array.from(view.samples, (x) => x[n - 1]), Toy.Y_MIN, Toy.Y_MAX, 64);
    } else {
      dyn.hist.replaceChildren();
    }
    // Finished sequences in grey.
    for (const idx of seqDone) {
      const xs = view.samples[idx];
      el("polyline", { points: line(xs, n), fill: "none", stroke: "#888", "stroke-width": 1.2, "stroke-opacity": 0.5 }, g);
      el("circle", { cx: L.xR0 + 5, cy: sy(xs[n - 1]), r: 3.5, fill: "#666" }, g);
    }
    // Distributions the values were drawn from (faint), and the next one (blue) with the plain
    // distribution of that position (dashed) for comparison.
    for (let j = 0; j <= Math.min(k, n - 1); j++) {
      const cond = Toy.seqConditional(Array.from(cur).slice(0, j));
      const peak = Math.max(...cond.means.map((m) => Toy.pdf(m, cond)));
      if (j < k) {
        sideDensity(g, px(j), cond, (step * 0.8) / peak, { stroke: "#9aa1ab", "stroke-width": 1 });
        continue;
      }
      const marg = Toy.seqMarginal(j);
      const scale = (step * 0.85) / Math.max(peak, ...marg.means.map((m) => Toy.pdf(m, marg)));
      sideDensity(g, px(j), cond, scale, { fill: "var(--target-fill)", stroke: "var(--target)", "stroke-width": 1.6, "fill-opacity": 0.85 });
      sideDensity(g, px(j), marg, scale, { stroke: "#6b7280", "stroke-width": 1.4, "stroke-dasharray": "4 3" });
    }
    // The current sequence.
    if (k > 0) {
      const pts = line(cur, k);
      el("polyline", { points: pts, fill: "none", stroke: "#fff", "stroke-width": 5 }, g);
      el("polyline", { points: pts, fill: "none", stroke: "#111", "stroke-width": 2.2 }, g);
      for (let j = 0; j < k; j++) el("circle", { cx: px(j), cy: sy(cur[j]), r: j === k - 1 ? 5 : 3.5, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, g);
      if (k === n) el("circle", { cx: L.xR0, cy: sy(cur[n - 1]), r: 5, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, g);
    }
    $("seq-next").textContent = k >= n ? "Next value (new sequence)" : "Next value";
  }

  // ---- autoregressive, 2-D companion ------------------------------------------------------------

  function render2d(svg) {
    const top = { y0: M.top, y1: M.top + 62 };
    const J0 = M.top + 74;
    const J1 = M.top + PH;
    const sx1 = (x1) => L.xM0 + ((x1 - X1.lo) / (X1.hi - X1.lo)) * (L.xM1 - L.xM0);
    const sy2 = (x2) => J1 - ((x2 - X2.lo) / (X2.hi - X2.lo)) * (J1 - J0);
    titles(svg, "", L.narrow ? "p(x₁) p(x₂ | x₁)" : "joint p(x₁, x₂) = p(x₁) · p(x₂ | x₁)", "p(x₂ | x₁)");

    // Joint density as a heat map (one normalization for the whole map).
    const cw = 220;
    const ch = 160;
    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(cw, ch);
    const vals = new Float64Array(cw * ch);
    let max = 0;
    for (let r = 0; r < ch; r++) {
      for (let q = 0; q < cw; q++) {
        const x1 = X1.lo + ((q + 0.5) / cw) * (X1.hi - X1.lo);
        const x2 = X2.hi - ((r + 0.5) / ch) * (X2.hi - X2.lo);
        vals[r * cw + q] = Toy.ar2dPdf(x1, x2);
        max = Math.max(max, vals[r * cw + q]);
      }
    }
    for (let k = 0; k < vals.length; k++) {
      img.data[4 * k] = 31;
      img.data[4 * k + 1] = 78;
      img.data[4 * k + 2] = 156;
      img.data[4 * k + 3] = Math.round(170 * Math.pow(vals[k] / max, 0.5));
    }
    ctx.putImageData(img, 0, 0);
    el("rect", { x: L.xM0, y: J0, width: L.xM1 - L.xM0, height: J1 - J0, fill: "#fff", stroke: "#e2e5ea" }, svg);
    el("image", { href: canvas.toDataURL(), x: L.xM0, y: J0, width: L.xM1 - L.xM0, height: J1 - J0, preserveAspectRatio: "none" }, svg);
    for (let v = 0; v <= 10; v += 2) {
      text(svg, L.xM0 - 6, sy2(v) + 4, minus(v), { "text-anchor": "end" });
      text(svg, sx1(v), H - 14, minus(v), { "text-anchor": "middle" });
    }
    text(svg, L.xM1, H - 2, "x₁", { "text-anchor": "end", "font-weight": 600 });
    text(svg, L.xM0 - 22, J0 + 10, "x₂", { "text-anchor": "middle", "font-weight": 600 });

    // Top: p(x1) and a histogram of the sampled x1.
    const tmax = 0.3;
    const ty = (d) => top.y1 - (Math.min(d, tmax) / tmax) * (top.y1 - top.y0);
    el("rect", { x: L.xM0, y: top.y0, width: L.xM1 - L.xM0, height: top.y1 - top.y0, fill: "#fff", stroke: "#e2e5ea" }, svg);
    const n2 = simple() ? ar2dShown : Math.max(ar2dShown, 1000);
    const shownX1 = view.x1.subarray(0, n2);
    if (!simple()) {
      for (const b of Toy.histogram(shownX1, X1.lo, X1.hi, 52)) {
        if (b.density > 0) el("rect", { x: sx1(b.y0) + 0.5, y: ty(b.density), width: Math.max(sx1(b.y1) - sx1(b.y0) - 1, 1), height: top.y1 - ty(b.density), fill: "#c9ccd2", stroke: "var(--hist)", "stroke-width": 0.6 }, svg);
      }
    }
    const tp = [];
    for (let x1 = X1.lo; x1 <= X1.hi + 1e-9; x1 += 0.05) tp.push(`${sx1(x1).toFixed(1)},${ty(Toy.pdf(x1, Toy.TARGET)).toFixed(1)}`);
    el("polyline", { points: tp.join(" "), fill: "none", stroke: "var(--target)", "stroke-width": 1.8 }, svg);
    text(svg, L.xM0 + 4, top.y0 + 13, "p(x₁)", {});

    // Samples drawn so far.
    const dots = el("g", { fill: "#444", "fill-opacity": 0.55 }, svg);
    for (let i = 0; i < n2; i++) el("circle", { cx: sx1(view.x1[i]), cy: sy2(view.x2[i]), r: simple() ? 3 : 1.7 }, dots);
    if (ar2dLast !== null) el("circle", { cx: sx1(view.x1[ar2dLast]), cy: sy2(view.x2[ar2dLast]), r: 6, fill: "none", stroke: "var(--noise)", "stroke-width": 2.5 }, svg);

    // Right strip: conditional density of x2 for the slider's x1.
    el("line", { x1: L.xR0, x2: L.xR0, y1: J0, y2: J1, stroke: "#9aa1ab" }, svg);
    const condFill = el("polygon", { fill: "var(--target-fill)", "fill-opacity": 0.6 }, svg);
    const cond = el("polyline", { fill: "none", stroke: "var(--target)", "stroke-width": 2 }, svg);
    const slice = el("g", { fill: "var(--noise)" }, svg);
    const line = el("line", { y1: top.y0, y2: J1, stroke: "#111", "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
    const marker = el("circle", { r: 4, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, svg);
    dyn = { sx1, sy2, ty, cond, condFill, slice, line, marker };
  }

  function update2d() {
    const x1 = colToX1(cursor);
    const { sx1, sy2 } = dyn;
    dyn.line.setAttribute("x1", sx1(x1));
    dyn.line.setAttribute("x2", sx1(x1));
    dyn.marker.setAttribute("cx", sx1(x1));
    dyn.marker.setAttribute("cy", dyn.ty(Toy.pdf(x1, Toy.TARGET)));
    const cmax = 0.9;
    const mix = Toy.ar2dConditional(x1);
    const pts = stripPoints(mix, (d) => rx(d, cmax), X2.lo, X2.hi, sy2);
    dyn.cond.setAttribute("points", pts.join(" "));
    dyn.condFill.setAttribute("points", `${L.xR0},${sy2(X2.lo)} ${pts.join(" ")} ${L.xR0},${sy2(X2.hi)}`);
    // Samples whose x1 is close to the slider: their x2 follow p(x2 | x1).
    dyn.slice.replaceChildren();
    for (let i = 0; i < (simple() ? ar2dShown : Math.max(ar2dShown, 1000)); i++) {
      if (Math.abs(view.x1[i] - x1) < 0.15) {
        el("circle", { cx: sx1(view.x1[i]), cy: sy2(view.x2[i]), r: 2.6 }, dyn.slice);
        el("circle", { cx: L.xR0 + 4, cy: sy2(view.x2[i]), r: 2.6 }, dyn.slice);
      }
    }
  }

  // ---- GAN: the generator as a function --------------------------------------------------------

  const Z_LO = -3.5;
  const Z_HI = 3.5;

  // The middle panel plots x = G(z) with z on the horizontal axis: a GAN evaluates one function,
  // there is no path between z and x.
  function renderGan(svg) {
    titles(svg, "", "", ganGen() || L.narrow ? "result" : "samples vs. target");
    const szz = (z) => L.xM0 + ((z - Z_LO) / (Z_HI - Z_LO)) * (L.xM1 - L.xM0);
    valueAxis(svg);
    // Noise density along the z axis, at the bottom of the panel.
    const base = M.top + PH;
    const nd = (d) => base - (d / 0.4) * 44;
    const pts = [];
    for (let z = Z_LO; z <= Z_HI + 1e-9; z += 0.05) pts.push(`${szz(z).toFixed(1)},${nd(Toy.pdf(z, Toy.NOISE)).toFixed(1)}`);
    el("polygon", { points: `${szz(Z_LO)},${base} ${pts.join(" ")} ${szz(Z_HI)},${base}`, fill: "var(--noise-fill)", stroke: "var(--noise)", "stroke-width": 1.2, "fill-opacity": 0.7 }, svg);
    for (let z = -3; z <= 3; z += 1) text(svg, szz(z), H - 14, minus(z), { "text-anchor": "middle" });
    text(svg, L.xM1, H - 2, "noise z", { "text-anchor": "end", "font-weight": 600 });
    text(svg, L.xM0 - 22, M.top - 6, "x", { "text-anchor": "middle", "font-weight": 600 });
    const curve = el("polyline", { fill: "none", stroke: "#111", "stroke-width": 2.4 }, svg);
    const dots = el("g", {}, svg);
    const traceG = el("g", {}, svg);
    // Right strip: samples vs. target, and the discriminator D(x) on its own scale (0 to 1).
    const hist = targetStrip(svg);
    let disc = null;
    if (!ganGen()) {
      const half = L.xR0 + 0.5 * (L.xR1 - L.xR0);
      el("line", { x1: half, x2: half, y1: M.top, y2: M.top + PH, stroke: "#8c510a", "stroke-width": 0.8, "stroke-dasharray": "1 3" }, svg);
      text(svg, half, M.top - 6, "D = ½", { "text-anchor": "middle" });
      disc = el("polyline", { fill: "none", stroke: "#8c510a", "stroke-width": 1.8, "stroke-dasharray": "5 3" }, svg);
    }
    dyn = { szz, base, curve, dots, traceG, hist, disc, title: text(svg, (L.xM0 + L.xM1) / 2, L.narrow ? 34 : 16, "", { "text-anchor": "middle", class: "title" }) };
  }

  function updateGan() {
    const s = view.snaps[cursor];
    const samples = view.samples[cursor];
    const G = (z) => Toy.gridInterp(MODELS.zGrid, s.g, z);
    const gen = ganGen();
    dyn.title.textContent = gen
      ? (L.narrow ? "x = G(z)" : "generator x = G(z): one evaluation per sample")
      : L.narrow ? `x = G(z), step ${s.step}` : `generator x = G(z) after ${s.step} training steps`;
    const pts = [];
    for (let z = Z_LO; z <= Z_HI + 1e-9; z += 0.02) pts.push(`${dyn.szz(z).toFixed(1)},${sy(Math.min(Math.max(G(z), Toy.Y_MIN), Toy.Y_MAX)).toFixed(1)}`);
    dyn.curve.setAttribute("points", pts.join(" "));
    dyn.dots.replaceChildren();
    const faint = trace !== null || gen;
    const population = !gen || state.showDist;
    for (let i = 0; i < (population ? N_DRAW : 0); i++) {
      const z = view.z[i];
      if (z < Z_LO || z > Z_HI) continue;
      const col = color(view.colorKey[i]);
      el("circle", { cx: dyn.szz(z), cy: dyn.base - 3, r: 1.8, fill: col }, dyn.dots);
      el("circle", { cx: dyn.szz(z), cy: sy(samples[i]), r: 2.6, fill: col, "fill-opacity": faint ? 0.3 : 0.9 }, dyn.dots);
    }
    dyn.traceG.replaceChildren();
    if (gen) {
      // Earlier samples (grey), then the selected z with its dashed mapping to x.
      for (const z of ganHistory) {
        el("circle", { cx: dyn.szz(z), cy: sy(G(z)), r: 3.5, fill: "#666" }, dyn.traceG);
        el("circle", { cx: L.xR0 + 5, cy: sy(G(z)), r: 3.5, fill: "#666" }, dyn.traceG);
      }
      const x = G(ganZ);
      const attrs = { stroke: "#111", "stroke-width": 1.5, "stroke-dasharray": "4 3" };
      el("line", { x1: dyn.szz(ganZ), x2: dyn.szz(ganZ), y1: dyn.base, y2: sy(x), ...attrs }, dyn.traceG);
      el("line", { x1: dyn.szz(ganZ), x2: L.xR0, y1: sy(x), y2: sy(x), ...attrs }, dyn.traceG);
      el("circle", { cx: dyn.szz(ganZ), cy: sy(x), r: 5, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, dyn.traceG);
      el("circle", { cx: L.xR0, cy: sy(x), r: 5, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, dyn.traceG);
      el("circle", { cx: dyn.szz(ganZ), cy: dyn.base - 4, r: 8, fill: "var(--noise)", stroke: "#fff", "stroke-width": 2, style: "cursor: ew-resize" }, dyn.traceG);
    } else if (trace !== null) {
      const z = view.z[trace];
      const x = samples[trace];
      const attrs = { stroke: "#111", "stroke-width": 1.5, "stroke-dasharray": "4 3" };
      el("line", { x1: dyn.szz(z), x2: dyn.szz(z), y1: dyn.base, y2: sy(x), ...attrs }, dyn.traceG);
      el("line", { x1: dyn.szz(z), x2: L.xR0, y1: sy(x), y2: sy(x), ...attrs }, dyn.traceG);
      el("circle", { cx: dyn.szz(z), cy: sy(x), r: 5, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, dyn.traceG);
    }
    if (population) drawHistogram(dyn.hist, samples, Toy.Y_MIN, Toy.Y_MAX, 64);
    else dyn.hist.replaceChildren();
    if (dyn.disc) {
      const dp = [];
      for (let x = MODELS.xGrid.lo; x <= MODELS.xGrid.hi + 1e-9; x += 0.1) dp.push(`${(L.xR0 + Toy.ganDiscriminator(MODELS, s, x) * (L.xR1 - L.xR0)).toFixed(1)},${sy(x).toFixed(1)}`);
      dyn.disc.setAttribute("points", dp.join(" "));
    }
    plotDescription();
    renderGuide();
  }

  // ---- VAE, lecture view: one z, its decoder distribution, repeated draws of x -------------------

  function renderVaeSingle(svg) {
    titles(svg, "prior", L.narrow ? "decoder" : "decoder: z → N(μθ(z), σθ(z)²), then draw x", "result");
    valueAxis(svg);
    const w = L.xM1 - L.xM0;
    const c0 = L.xM0;
    const c1 = L.xM0 + 0.4 * w;
    const c2 = L.xM1;
    text(svg, c0, H - 14, L.narrow ? "z" : "z ~ N(0, 1)", { "text-anchor": "start" });
    text(svg, c1, H - 14, L.narrow ? "decoder" : "decoder distribution", { "text-anchor": "middle" });
    text(svg, c2, H - 14, "x", { "text-anchor": "end" });
    const nx = (d) => L.xL1 - (Math.min(d, DMAX) / DMAX) * (L.xL1 - L.xL0);
    const pts = stripPoints(Toy.NOISE, nx);
    el("polygon", { points: `${L.xL1},${sy(Toy.Y_MIN)} ${pts.join(" ")} ${L.xL1},${sy(Toy.Y_MAX)}`, fill: "var(--noise-fill)", stroke: "var(--noise)", "stroke-width": 1.5 }, svg);
    el("line", { x1: L.xL1, x2: L.xL1, y1: M.top, y2: M.top + PH, stroke: "#9aa1ab" }, svg);
    targetStrip(svg);
    dyn = { c0, c1, c2, w, g: el("g", {}, svg) };
  }

  function updateVaeSingle() {
    const { c0, c1, c2, w, g } = dyn;
    g.replaceChildren();
    const d = Toy.vaeDecoder(MODELS, vaeZ);
    const dec = { weights: [1], means: [d.mean], stds: [d.std] };
    const peak = Toy.pdf(d.mean, dec);
    const width = 0.3 * w;
    const pts = [];
    for (let y = d.mean - 4 * d.std; y <= d.mean + 4 * d.std + 1e-9; y += d.std / 20) pts.push(`${(c1 + (Toy.pdf(y, dec) / peak) * width).toFixed(1)},${sy(y).toFixed(1)}`);
    el("polygon", { points: `${c1},${sy(d.mean - 4 * d.std)} ${pts.join(" ")} ${c1},${sy(d.mean + 4 * d.std)}`, fill: "#d9dde5", stroke: "#444", "stroke-width": 1.5 }, g);
    el("line", { x1: c1 - 8, x2: c1 + width, y1: sy(d.mean), y2: sy(d.mean), stroke: "#444", "stroke-dasharray": "3 2" }, g);
    text(g, c1 + 4, sy(d.mean + 4 * d.std) - 8, `μθ(z) = ${num(d.mean)}, σθ(z) = ${fixed(d.std)}`, {});
    el("line", { x1: c0, y1: sy(vaeZ), x2: c1, y2: sy(d.mean), stroke: "var(--noise)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }, g);
    vaeDraws.forEach((x, k) => {
      const last = k === vaeDraws.length - 1;
      el("line", { x1: c1, y1: sy(d.mean), x2: c2, y2: sy(x), stroke: "#555", "stroke-opacity": last ? 0.7 : 0.2, "stroke-dasharray": "2 3" }, g);
      el("circle", { cx: c2, cy: sy(x), r: last ? 5 : 3, fill: last ? "#111" : "#888", stroke: "#fff", "stroke-width": last ? 1.5 : 0 }, g);
      el("circle", { cx: L.xR0 + 4, cy: sy(x), r: last ? 5 : 3, fill: last ? "#111" : "#888", stroke: "#fff", "stroke-width": last ? 1.5 : 0 }, g);
    });
    el("circle", { cx: L.xL1, cy: sy(vaeZ), r: 8, fill: "var(--noise)", stroke: "#fff", "stroke-width": 2, style: "cursor: ns-resize" }, g);
    el("circle", { cx: c0, cy: sy(vaeZ), r: 4, fill: "var(--noise)" }, g);
  }

  // ---- VAE inference: x → q(z | x) → reconstructions -------------------------------------------

  function renderVaeInfer(svg) {
    titles(svg, "q(z | x)", L.narrow ? "encoder, decoder" : "inference: encoder x → q(z | x), decoder z → reconstruction", L.narrow ? "x" : "x and reconstructions");
    valueAxis(svg);
    const enc = Toy.vaeEncoder(MODELS, vaeInferX);
    const c0 = L.xM0;
    const c1 = (L.xM0 + L.xM1) / 2;
    const c2 = L.xM1;
    text(svg, c0, H - 14, "z ~ q(z | x)", { "text-anchor": "start" });
    text(svg, c1, H - 14, "decoder μθ(z) ± σθ(z)", { "text-anchor": "middle" });
    text(svg, c2, H - 14, "reconstruction", { "text-anchor": "end" });
    // Left strip: the prior N(0, 1) (outline) and the encoder's q(z | x) (filled, own scale).
    const qmax = Toy.pdf(enc.mean, { weights: [1], means: [enc.mean], stds: [enc.std] });
    const qx = (d) => L.xL1 - (d / qmax) * (L.xL1 - L.xL0);
    const nx = (d) => L.xL1 - (Math.min(d, DMAX) / DMAX) * (L.xL1 - L.xL0);
    el("polyline", { points: stripPoints(Toy.NOISE, nx).join(" "), fill: "none", stroke: "var(--noise)", "stroke-width": 1.2, "stroke-dasharray": "3 2" }, svg);
    const q = { weights: [1], means: [enc.mean], stds: [enc.std] };
    const qp = stripPoints(q, qx);
    el("polygon", { points: `${L.xL1},${sy(Toy.Y_MIN)} ${qp.join(" ")} ${L.xL1},${sy(Toy.Y_MAX)}`, fill: "#e4d7f0", stroke: "#6a3d9a", "stroke-width": 1.5 }, svg);
    el("line", { x1: L.xL1, x2: L.xL1, y1: M.top, y2: M.top + PH, stroke: "#9aa1ab" }, svg);
    // Encoder arrow from x to the mean of q(z | x).
    const defs = el("defs", {}, svg);
    arrowMarker(defs, "arrow-q", "#6a3d9a");
    el("line", { x1: c2 - 6, y1: sy(vaeInferX), x2: c0 + 8, y2: sy(enc.mean), stroke: "#6a3d9a", "stroke-width": 2.5, "stroke-opacity": 0.85, "marker-end": "url(#arrow-q)" }, svg);
    text(svg, (c1 + c2) / 2, (sy(vaeInferX) + sy(enc.mean)) / 2 - 8, "encoder", { "text-anchor": "middle", fill: "#6a3d9a" });
    // Samples of q, decoded.
    const m = 40;
    const ez = Toy.normals(m, Toy.rng(state.seed + 101));
    const ex = Toy.normals(2000, Toy.rng(state.seed + 202));
    const g = el("g", {}, svg);
    for (let j = 0; j < m; j++) {
      const z = enc.mean + enc.std * ez[j];
      const d = Toy.vaeDecoder(MODELS, z);
      const xr = d.mean + d.std * ex[j];
      el("line", { x1: c0, y1: sy(z), x2: c1, y2: sy(d.mean), stroke: "#6a3d9a", "stroke-opacity": 0.35, "stroke-dasharray": "2 3" }, g);
      el("line", { x1: c1, y1: sy(d.mean), x2: c2, y2: sy(xr), stroke: "#555", "stroke-opacity": 0.25, "stroke-dasharray": "2 3" }, g);
      el("line", { x1: c1, x2: c1, y1: sy(d.mean - d.std), y2: sy(d.mean + d.std), stroke: "#555", "stroke-opacity": 0.5 }, g);
      el("circle", { cx: c0, cy: sy(z), r: 2.4, fill: "#6a3d9a" }, g);
      el("circle", { cx: c1, cy: sy(d.mean), r: 2.2, fill: "#555" }, g);
      el("circle", { cx: c2, cy: sy(xr), r: 2.2, fill: "#555", "fill-opacity": 0.7 }, g);
    }
    // Right strip: reconstructions of x (many samples of q and of the decoder) vs. the target.
    const hist = targetStrip(svg);
    const rec = new Float64Array(2000);
    const ez2 = Toy.normals(2000, Toy.rng(state.seed + 303));
    for (let j = 0; j < 2000; j++) {
      const d = Toy.vaeDecoder(MODELS, enc.mean + enc.std * ez2[j]);
      rec[j] = d.mean + d.std * ex[j];
    }
    drawHistogram(hist, rec, Toy.Y_MIN, Toy.Y_MAX, 64);
    el("path", { d: `M${L.xR0 + 2},${sy(vaeInferX)} l10,-6 v12 z`, fill: "#111" }, svg);
    text(svg, L.xR0 + 14, sy(vaeInferX) + 4, "x", { "font-weight": 600 });
    dyn = { enc, rec };
  }

  // ---- Compare: final distributions of three settings, side by side (stacked on phones) ------------

  function renderCompare(svg) {
    const lo = -3;
    const hi = 12;
    const ymax = 0.5;
    const n = view.panels.length;
    const gap = 26;
    const ph = L.narrow ? 150 : PH;
    const pw = L.narrow ? L.W - 46 : (L.W - 40 - gap * (n - 1)) / n;
    const height = L.narrow ? n * (ph + 76) + 10 : H;
    svg.setAttribute("viewBox", `0 0 ${L.W} ${height}`);
    view.panels.forEach((p, k) => {
      const x0 = L.narrow ? 36 : 30 + k * (pw + gap);
      const y0 = L.narrow ? 50 + k * (ph + 76) : M.top;
      const x1 = x0 + pw;
      const y1 = y0 + ph;
      const sx = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
      const syd = (d) => y1 - (Math.min(d, ymax) / ymax) * ph;
      text(svg, x0, y0 - 26, p.label, { class: "title" });
      text(svg, x0, y0 - 10, `${p.kind} · W₁ = ${fixed(p.w1, 2)}`, {});
      el("rect", { x: x0, y: y0, width: pw, height: ph, fill: "#fff", stroke: "#e2e5ea" }, svg);
      for (let v = 0; v <= 12; v += 4) text(svg, sx(v), y1 + 14, minus(v), { "text-anchor": "middle" });
      const tp = [];
      for (let v = lo; v <= hi + 1e-9; v += 0.05) tp.push(`${sx(v).toFixed(1)},${syd(Toy.pdf(v, Toy.TARGET)).toFixed(1)}`);
      el("polygon", { points: `${sx(lo)},${y1} ${tp.join(" ")} ${sx(hi)},${y1}`, fill: "var(--target-fill)", "fill-opacity": 0.8 }, svg);
      el("polyline", { points: tp.join(" "), fill: "none", stroke: "var(--target)", "stroke-width": 1.6 }, svg);
      const hp = [];
      for (const b of Toy.histogram(p.x, lo, hi, 75)) hp.push(`${sx(b.y0).toFixed(1)},${syd(b.density).toFixed(1)}`, `${sx(b.y1).toFixed(1)},${syd(b.density).toFixed(1)}`);
      el("polyline", { points: hp.join(" "), fill: "none", stroke: "#333", "stroke-width": 1.6, "stroke-linejoin": "round" }, svg);
    });
    dyn = {};
  }

  // ---- legend -----------------------------------------------------------------------------------

  function renderLegend() {
    const items = [];
    if (ganGen()) {
      items.push(["noise", "noise z ~ N(0, 1); drag the orange handle"], ["gcurve", "generator x = G(z)"], ["dots", "earlier samples"], ["target", "target p<sub>data</sub>(x)"]);
      if (state.showDist) items.push(["paths", "120 samples (z, G(z))"], ["hist", "histogram of 4,000 generated samples"]);
    } else if (vaeSingle()) {
      items.push(["noise", "prior N(0, 1); drag the orange handle"], ["dec", "decoder distribution N(μθ(z), σθ(z)²)"], ["dots", "x drawn from it (latest in black)"], ["target", "target p<sub>data</sub>(x)"]);
    } else if (view.gan) {
      items.push(["noise", "noise z ~ N(0, 1)"], ["gcurve", "generator x = G(z)"], ["paths", "samples (z, G(z)), colored by z"],
        ["target", "target p<sub>data</sub>(x)"], ["hist", "histogram of 4,000 generated samples"], ["disc", "discriminator D(x): 0 = “fake”, 1 = “real”"]);
    } else if (view.vae && state.vae.mode === "infer") {
      items.push(["q", "encoder q(z | x)"], ["noise", "prior N(0, 1)"], ["dots", "z ~ q(z | x), decoder means, reconstructions"],
        ["target", "target p<sub>data</sub>(x)"], ["hist", "reconstructions of x"]);
    } else if (view.compare) {
      items.push(["target", "target p<sub>data</sub>(x)"], ["gcurve", "histogram of 4,000 samples of the setting"]);
    } else if (view.vae) {
      items.push(["noise", "latent z ~ N(0, 1)"], ["paths", "samples: z → decoder μθ(z) ± σθ(z) → x"], ["target", "target p<sub>data</sub>(x)"],
        ["hist", "histogram at the slider position"]);
    } else if (view.arseq) {
      items.push(["cond", "distribution of the next value, given the values so far"], ["margdash", "distribution of that position without context"],
        ["gcurve", "current sequence"], ["dots", "finished sequences"], ["target", "target p<sub>data</sub>(x) = distribution of x₈"]);
      if (!simple()) items.push(["paths", "120 sequences"], ["hist", "x₈ of 4,000 sequences"]);
    } else if (view.ar && simple()) {
      items.push(["tokbar", "p(c) and p(f | c); dark: chosen token"], ["target", "target p<sub>data</sub>(x)"], ["dots", "finished values (current in black)"]);
      if (state.ar.gauges) items.push(["uniform", "random numbers u₁, u₂, cut into the token probabilities"]);
    } else if (view.ar2d && simple()) {
      items.push(["heat", "joint density p(x₁, x₂)"], ["dots", "values drawn so far"], ["target", "p(x₁) (top), p(x₂ | x₁) at the chosen x₁ (right)"],
        ["probe", "values with x₁ near the chosen x₁"]);
    } else if (view.ar) {
      items.push(["uniform", "random numbers u₁, u₂, cut into the token probabilities"], ["tokbar", "p(c) and p(f | c); dark: chosen token"],
        ["target", "target p<sub>data</sub>(x)"], ["hist", "all samples after the current token"]);
    } else if (view.ar2d) {
      items.push(["heat", "joint density p(x₁, x₂)"], ["dots", "samples"], ["target", "p(x₁) (top), p(x₂ | x₁) at the slider (right)"],
        ["probe", "samples with x₁ near the slider"]);
    } else if (view.straight) {
      items.push(["noise", "noise N(0, 1)"], ["target", "target p<sub>data</sub>(x)"], ["paths", "training lines between random noise–data pairs"],
        ["probe", "average direction of the lines near the probe"], ["marg", "exact velocity u(x, t)"]);
      if (state.fm.field) items.push(["field", "direction of the velocity field u(x, t)"]);
    } else if (simple()) {
      items.push(["noise", "noise N(0, 1)"], ["target", "target p<sub>data</sub>(x)"], ["paths", `${N_SIMPLE} samples, colored by their starting value`],
        ["gcurve", "highlighted sample (click another path to follow it)"]);
      if (state.showDensity && view.density) items.push(["heat", "exact density along the process"]);
    } else {
      items.push(["noise", "noise N(0, 1)"], ["target", "target p<sub>data</sub>(x)"], ["paths", "samples, colored by their starting value"],
        ["hist", "histogram at the slider position"], ["marg", "exact distribution at the slider position"]);
      if (state.showDensity && view.density) items.push(["heat", "exact density along the process"]);
      if (state.family === "fm" && state.fm.field) items.push(["field", "direction of the velocity field u(x, t)"]);
    }
    $("legend").innerHTML = items.map(([cls, label]) => `<li><span class="swatch ${cls}"></span>${label}</li>`).join("");
  }

  // Everything that changes when only the slider moves.
  function updateCursor() {
    if (!view || !dyn) return;
    cursor = Math.min(Math.max(cursor, 0), view.cols - 1);
    const slider = $("cursor");
    slider.max = String(view.cols - 1);
    slider.value = String(cursor);
    $("cursor-out").textContent = columnLabel(cursor);
    slider.setAttribute("aria-valuetext", columnLabel(cursor));
    if (view.arseq) updateSeq();
    else if (view.ar) updateTokens();
    else if (view.ar2d) update2d();
    else if (view.gan) updateGan();
    else if (vaeSingle()) updateVaeSingle();
    else if (view.compare || (view.vae && state.vae.mode === "infer")) { /* static */ }
    else updatePaths();
    $("fm-inset").hidden = !(state.family === "fm" && state.fm.mode === "pairs");
    describeTrace();
  }

  function updatePaths() {
    const cx = sxCol(cursor);
    if (view.forward) {
      dyn.clipRect.setAttribute("x", cx - 2);
      dyn.clipRect.setAttribute("width", L.xM1 - cx + 6);
    } else {
      dyn.clipRect.setAttribute("x", L.xM0 - 4);
      dyn.clipRect.setAttribute("width", cx - L.xM0 + 6);
    }
    dyn.cursorLine.setAttribute("x1", cx);
    dyn.cursorLine.setAttribute("x2", cx);
    const atEnd = cursor === endColumn();
    dyn.cursorLine.setAttribute("visibility", atEnd ? "hidden" : "visible");
    dyn.cursorText.setAttribute("x", Math.min(Math.max(cx, L.xM0 + 50), L.xM1 - 50));
    dyn.cursorText.textContent = atEnd ? "" : columnLabel(cursor);

    // Markers at the slider position.
    dyn.markers.replaceChildren();
    const faint = trace !== null;
    for (let i = 0; i < dyn.drawn; i++) {
      el("circle", { cx, cy: sy(value(cursor, i)), r: 2.2, fill: color(view.colorKey[i]), "fill-opacity": faint ? 0.3 : 0.9 }, dyn.markers);
    }
    const showTrace = trace !== null && !view.straight;
    if (showTrace) el("circle", { cx, cy: sy(value(cursor, trace)), r: 4.5, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, dyn.markers);

    if (simple()) {
      // Follow a sample: no population statistics, only the highlighted sample's current value.
      dyn.hist.replaceChildren();
      dyn.marg.setAttribute("visibility", "hidden");
      if (showTrace) el("circle", { cx: L.xR0, cy: sy(value(cursor, trace)), r: 4.5, fill: "#111", stroke: "#fff", "stroke-width": 1.5 }, dyn.markers);
      return;
    }
    // Histogram of all samples at the slider position.
    drawHistogram(dyn.hist, view.states.subarray(cursor * view.n, (cursor + 1) * view.n), Toy.Y_MIN, Toy.Y_MAX, 64);
    // Exact distribution at the slider position (dashed), when it differs from the target.
    if (cursor !== view.cols - 1 && view.marginal) {
      dyn.marg.setAttribute("points", stripPoints(view.marginal(cursor), (d) => rx(d)).join(" "));
      dyn.marg.setAttribute("visibility", "visible");
    } else {
      dyn.marg.setAttribute("visibility", "hidden");
    }
  }

  // ---- clicks -------------------------------------------------------------------------------

  // Click: depends on the view. Paths: trace the closest drawn sample. Flow-matching pairs: move
  // the probe. Flow density evaluation: choose x. AR tokens: show p(f | c) for a clicked bar.
  function pick(e) {
    const svg = $("plot");
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * L.W;
    const y = ((e.clientY - r.top) / r.height) * H;
    if (y < M.top - 6 || y > M.top + PH + 6) return;
    if (view.arseq) return;
    if (view.ar) {
      if (x < dyn.A0 || x > dyn.A1) return;
      const cc = Math.floor(unSy(y) - Toy.AR.lo);
      if (cc < 0 || cc >= Toy.AR.coarse) return;
      arOverrideC = cc === arOverrideC ? null : cc;
      updateCursor();
      return;
    }
    if (view.compare) return;
    if (view.vae && state.vae.mode === "infer") {
      if (x < L.xM0 || x > L.xR1) return;
      vaeInferX = Math.min(Math.max(Math.round(unSy(y) * 20) / 20, -1.5), 11);
      render();
      return;
    }
    if (ganGen() || vaeSingle()) return;
    if (view.gan) {
      let bestI = null;
      let bestD = 14;
      for (let i = 0; i < N_DRAW; i++) {
        const z = view.z[i];
        if (z < Z_LO || z > Z_HI) continue;
        const d = Math.min(Math.hypot(dyn.szz(z) - x, sy(view.samples[cursor][i]) - y), Math.hypot(dyn.szz(z) - x, dyn.base - 3 - y));
        if (d < bestD) { bestD = d; bestI = i; }
      }
      trace = bestI === trace ? null : bestI;
      updateCursor();
      return;
    }
    if (view.ar2d) {
      if (x < L.xM0 || x > L.xM1) return;
      stop();
      cursor = x1ToCol(X1.lo + ((x - L.xM0) / (L.xM1 - L.xM0)) * (X1.hi - X1.lo));
      updateCursor();
      return;
    }
    if (state.family === "fm" && state.fm.mode === "pairs") {
      if (x < L.xM0 || x > L.xM1) return;
      fmProbe = { x: Math.round(unSy(y) * 20) / 20, t: Math.min(Math.max(Math.round(((x - L.xM0) / (L.xM1 - L.xM0)) * 100) / 100, 0.02), 0.98) };
      render();
      return;
    }
    if (x < L.xL0 || x > L.xM1 + 4) return;
    let c = 0;
    if (x > L.xM0) {
      let best = Infinity;
      for (let j = 0; j < view.cols; j++) {
        const d = Math.abs(sxCol(j) - x);
        if (d < best) { best = d; c = j; }
      }
    } else if (view.forward) {
      return;
    }
    let bestI = null;
    let bestD = 12;
    for (let i = 0; i < drawnCount(); i++) {
      const d = Math.abs(sy(value(c, i)) - y);
      if (d < bestD) { bestD = d; bestI = i; }
    }
    trace = bestI === trace ? null : bestI;
    traceAuto = false;
    render();
  }

  // Diffusion with the stochastic sampler: remember where the traced sample ended in each run.
  function recordRun() {
    const key = JSON.stringify([state.seed, state.diffusion.steps, trace]);
    if (key !== runs.key) runs = { key, ends: [] };
    if (!runs.ends.some((r) => r.run === state.samplerSeed)) runs.ends.push({ run: state.samplerSeed, x: value(endColumn(), trace) });
  }

  function describeTrace() {
    const box = $("trace");
    const clearButton = () => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = "Clear (Esc)";
      box.append(" ", b);
    };
    if (view.arseq) {
      const n = Toy.SEQ.n;
      const cur = Array.from(view.samples[seqIndex]);
      if (seqStep === 0) {
        box.textContent = "Press “Next value” to draw x₁.";
      } else if (seqStep >= n) {
        box.textContent = `Sequence complete: x₈ = ${num(cur[n - 1])}. Press “Next value” to start a new sequence.`;
      } else {
        const post = Toy.seqPosterior(cur.slice(0, seqStep));
        const right = post[1] >= post[0];
        const pct = Math.round(100 * Math.max(...post));
        const sure = pct >= 80
          ? `The values so far point towards the ${right ? "right" : "left"} mode (${right ? 7 : 2}) with ${pct}% probability.`
          : `The values so far do not decide the direction yet (${Math.round(100 * post[0])}% left, ${Math.round(100 * post[1])}% right).`;
        const drawn = seqStep === 1 ? "x₁" : `x₁ … x${sub(seqStep)}`;
        box.textContent = `${drawn} = ${cur.slice(0, seqStep).map((v) => num(v)).join(", ")}. ${sure}`;
      }
      return;
    }
    if (view.ar) {
      const s = view.sample;
      const i = arIndex;
      if (arOverrideC !== null) {
        box.textContent = `Showing p(f | c) for the coarse bin [${minus(Toy.arEdge(arOverrideC))}, ${minus(Toy.arEdge(arOverrideC + 1))}) you clicked. Click it again or press “Next token” to continue.`;
        return;
      }
      const u = state.ar.gauges; // name the random numbers only when they are shown
      const parts = [`Value ${i + 1}:`];
      if (cursor >= 1) parts.push(`${u ? `u₁ = ${fixed(s.u1[i])} → ` : "token 1: "}c = ${s.c[i]} (x in [${minus(Toy.arEdge(s.c[i]))}, ${minus(Toy.arEdge(s.c[i] + 1))}))`);
      if (cursor >= 2) parts.push(`${u ? `then u₂ = ${fixed(s.u2[i])} → ` : "→ token 2: "}f = ${s.f[i]} → x = ${num(s.x[i])}`);
      if (cursor === 0) parts.push("no token chosen yet. Press “Next token”.");
      box.textContent = parts.join(" ");
      return;
    }
    if (view.ar2d) {
      if (ar2dLast !== null) {
        const i = ar2dLast;
        const x1 = view.x1[i];
        box.textContent = `Last sample: x₁ = ${num(x1)} from p(x₁), then x₂ = ${num(view.x2[i])} from p(x₂ | x₁) = N(${num(Toy.AR2D.mean(x1))}, ${num(Toy.AR2D.std(x1))}²). Values drawn: ${ar2dShown}.`;
      } else {
        box.textContent = `Values drawn: ${ar2dShown}. Drag in the plot to choose x₁ and watch p(x₂ | x₁) on the right, or press “Next value”.`;
      }
      return;
    }
    if (view.compare) {
      box.textContent = "Each panel: 4,000 samples of one setting (dark line) vs. the target (blue).";
      return;
    }
    if (view.vae && state.vae.mode === "infer") {
      const rm = dyn.rec.reduce((a, b) => a + b, 0) / dyn.rec.length;
      box.textContent = `Encoder: q(z | x = ${num(vaeInferX)}) = N(${num(dyn.enc.mean)}, ${num(dyn.enc.std)}²). Decoding samples of q gives reconstructions with mean ${num(rm)}. Click in the plot to choose another x.`;
      return;
    }
    if (ganGen()) {
      const x = Toy.gridInterp(MODELS.zGrid, view.snaps[cursor].g, ganZ);
      box.textContent = `z = ${num(ganZ)} → x = G(z) = ${num(x)}.${ganHistory.length ? ` Earlier samples in grey: ${ganHistory.length}.` : ""} Drag the orange handle or press “Generate another sample”.`;
      return;
    }
    if (vaeSingle()) {
      const d = Toy.vaeDecoder(MODELS, vaeZ);
      const list = vaeDraws.slice(-6).map((x) => num(x)).join(", ");
      box.textContent = vaeDraws.length
        ? `z = ${num(vaeZ)} → decoder N(${num(d.mean)}, ${fixed(d.std)}²). Drawn x (latest last): ${list}${vaeDraws.length > 6 ? ", …" : ""}.`
        : `z = ${num(vaeZ)} → decoder N(${num(d.mean)}, ${fixed(d.std)}²). Press “Sample x again” to draw x from it.`;
      return;
    }
    if (view.gan) {
      // Training dynamics: what G does at this training step.
      const s = view.snaps[cursor];
      const losses = s.lossD === null ? "" : ` Losses before this snapshot: D ${fixed(s.lossD)}, G ${fixed(s.lossG)}.`;
      box.innerHTML = ganNote() + losses;
      return;
    }
    if (trace === null || view.straight) {
      box.textContent = state.family === "fm" && state.fm.mode === "pairs"
        ? "Click in the plot to probe another point (x, t)."
        : "Click a path (or a dot on the noise axis) to trace one sample.";
      return;
    }
    const start = value(startColumn(), trace);
    const now = value(cursor, trace);
    const end = value(endColumn(), trace);
    if (view.vae) {
      box.textContent = `Traced sample: z = ${num(view.z[trace])} → decoder N(${num(view.mean[trace])}, ${num(view.std[trace])}²) → x = ${num(value(2, trace))}.`;
      clearButton();
      return;
    }
    let s = view.forward
      ? `Traced data sample x₀ = ${num(start)} → at the slider ${num(now)} → at t = T ${num(end)}.`
      : `Highlighted sample: starts at ${num(start)}, now at ${num(now)}${cursor === endColumn() ? "" : `, ends at ${num(end)}`}.`;
    if (state.family === "diffusion" && state.diffusion.mode === "reverse" && state.diffusion.sampler === "ddpm") {
      recordRun();
      if (runs.ends.length > 1) s += ` Same start, end points over sampler runs: ${runs.ends.map((r) => `#${r.run}: ${num(r.x)}`).join(", ")}.`;
    }
    box.textContent = s;
    clearButton();
  }

  // ---- text alternatives ------------------------------------------------------------------------

  // Description of the plot for screen readers.
  function plotDescription() {
    const name = content().name;
    $("plot-desc").textContent = view.compare
      ? `${name}: three panels with the final distributions of ${view.panels.map((p) => p.label).join(", ")} vs. the bimodal target.`
      : view.gan
        ? `${name}: the generator x = G(z)${ganGen() ? "" : ` after ${view.snaps[cursor].step} training steps`} and its samples vs. the bimodal target.`
        : view.vae
          ? `${name}: ${state.vae.mode === "infer" ? "encoding x into q(z | x) and decoding it again" : "a latent value z, its decoder distribution and values drawn from it"}.`
          : view.arseq
            ? `${name}: a sequence of ${Toy.SEQ.n} values drawn one after the other; the distribution of the next value given the values so far is shown at its position.`
          : view.ar
            ? `${name}: sampling a value as two tokens, a coarse and a fine one, from exact probabilities of the bimodal target with modes at 2 and 7.`
            : view.ar2d
              ? `${name}: joint density p(x1, x2) = p(x1) p(x2 | x1) with ${ar2dShown} values drawn; the conditional density of x2 at the chosen x1 is shown on the right.`
              : `${name}: ${drawnCount()} sample paths from standard-normal noise (left) to the bimodal target with modes at 2 and 7 (right).`;
  }

  // What the GAN does at the current snapshot, from simple statistics of its samples.
  function ganNote() {
    const x = view.samples[cursor];
    const step = view.snaps[cursor].step;
    const n = x.length;
    let left = 0;
    let gap = 0;
    let sum = 0;
    let sum2 = 0;
    let rightN = 0;
    let rightSum = 0;
    let rightSum2 = 0;
    for (const v of x) {
      if (v < 4.5) left++;
      if (v > 3.5 && v < 5.5) gap++;
      sum += v;
      sum2 += v * v;
      if (v >= 4.5) { rightN++; rightSum += v; rightSum2 += v * v; }
    }
    const sd = Math.sqrt(sum2 / n - (sum / n) ** 2);
    const rightSd = rightN > 10 ? Math.sqrt(rightSum2 / rightN - (rightSum / rightN) ** 2) : 0;
    let p;
    if (step === 0) p = ("Step 0: the untrained generator maps every z to almost the same value. The slider runs through the training; press play.");
    else if (sd < 0.5) p = (`Step ${step}: almost all samples have nearly the <strong>same value</strong> (spread ${fixed(sd)}). The discriminator will learn to reject it, and the generator jumps elsewhere: training oscillates.`);
    else if (gap / n > 0.5) p = (`Step ${step}: most samples (${Math.round((100 * gap) / n)}%) fall <strong>between</strong> the modes, where real data is rare. Neither mode is produced.`);
    else if (left / n < 0.05 || left / n > 0.95) p = (`Step ${step}: <strong>mode collapse</strong>. The generator only produces the ${left / n < 0.05 ? "right" : "left"} mode. Each sample looks realistic, but the other mode is missing.`);
    else if (gap / n > 0.15) p = (`Step ${step}: ${Math.round((100 * gap) / n)}% of the samples fall between the modes (target: 5%). G still connects the modes with a steep but not steep enough step.`);
    else if (rightSd < 0.8) p = (`Step ${step}: both modes, with roughly the right weights (left mode ${Math.round((100 * left) / n)}%, target 40%). But each mode is squeezed into a few narrow spikes: G is almost flat over ranges of z (spread of the right mode ${fixed(rightSd)}, target 1.00). The samples look realistic but vary too little.`);
    else if (view.w1s[cursor] < 0.15) p = (`Step ${step}: close to the target.`);
    else p = (`Step ${step}: both modes are present, but their positions and shapes are off (W<sub>1</sub> = ${fixed(view.w1s[cursor])}). Move the slider: the generator keeps changing, GAN training does not settle smoothly.`);
    return p;
  }

  // Optional: an embedding page on the same site can add `data-auto-height` to the iframe;
  // the iframe then follows the height of the demo instead of using a fixed height.
  function followHeight() {
    let frame = null;
    try {
      if (window.frameElement && window.frameElement.hasAttribute("data-auto-height")) frame = window.frameElement;
    } catch (error) {
      // cross-origin parent: keep the fixed height
    }
    if (!frame) return;
    new ResizeObserver(() => {
      const borderBox = window.parent.getComputedStyle(frame).boxSizing === "border-box";
      const chrome = borderBox ? frame.offsetHeight - window.innerHeight : 0;
      frame.style.height = Math.ceil(document.body.getBoundingClientRect().height) + chrome + "px";
    }).observe(document.body);
  }

  document.body.classList.toggle("embed", EMBED);
  fillSelect("tau", TEMPERATURES);
  fillSelect("diff-steps", DIFF_STEPS);
  fillSelect("fm-steps", FM_STEPS);
  bindEvents();
  changed();
  followHeight();
  // Redraw when the width of the plot changes (window resize, iframe resize).
  let lastWidth = $("plot").clientWidth;
  new ResizeObserver(() => {
    const width = $("plot").clientWidth;
    if (Math.abs(width - lastWidth) > 2) {
      lastWidth = width;
      render();
    }
  }).observe($("plot").parentElement);
})();
