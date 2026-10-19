// Interface of the demo: controls, plot, statistics. Computation is in dist.js.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";

  // Generator parameters (the target lives in Dist.TARGET and is never changed here).
  const DEFAULTS = Object.freeze({
    kind: 1,
    means: [4, 8],
    stds: [1.5, 1],
    weight: 0.5, // weight of the first component in the mixture
    count: 300,
    seed: 1,
    showTrain: false,
  });
  const PRESETS = {
    collapse: { kind: 1, means: [7, 8], stds: [0.6, 1], weight: 0.5 },
    broad: { kind: 1, means: [4.5, 8], stds: [4, 1], weight: 0.5 },
  };

  const state = structuredClone(DEFAULTS);
  const train = Dist.trainingData();
  const llTarget = Dist.logLikelihood(train, Dist.TARGET);
  let noise = null;
  let noiseKey = "";
  let lastAction = "";

  function model() {
    if (state.kind === 1) return { means: [state.means[0]], stds: [state.stds[0]], weights: [1] };
    return { means: state.means.slice(), stds: state.stds.slice(), weights: [state.weight, 1 - state.weight] };
  }

  // Noise is drawn once per seed and sample count and reused while the parameters change.
  function currentNoise() {
    const key = `${state.count}:${state.seed}`;
    if (key !== noiseKey) {
      noise = Dist.noise(state.count, state.seed);
      noiseKey = key;
    }
    return noise;
  }

  const fixed = (x, d = 2) => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const percent = (x) => `${Math.round(x * 100)}%`;
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  // ---- controls -------------------------------------------------------------------------

  function syncControls() {
    $("kind").value = String(state.kind);
    $("second").hidden = state.kind === 1;
    for (const k of [0, 1]) {
      $(`mean${k}`).value = state.means[k];
      $(`std${k}`).value = state.stds[k];
      $(`mean${k}-out`).textContent = fixed(state.means[k], 1);
      $(`std${k}-out`).textContent = fixed(state.stds[k], 2);
    }
    $("weight").value = state.weight;
    $("weight-out").textContent = fixed(state.weight, 2);
    $("count").value = String(state.count);
    $("seed").value = state.seed;
    $("show-train").checked = state.showTrain;
    $("legend-train").hidden = !state.showTrain;
  }

  function bindEvents() {
    $("kind").addEventListener("change", (e) => {
      state.kind = Number(e.target.value);
      lastAction = "";
      update();
    });
    for (const k of [0, 1]) {
      $(`mean${k}`).addEventListener("input", (e) => { state.means[k] = Number(e.target.value); lastAction = ""; update(); });
      $(`std${k}`).addEventListener("input", (e) => { state.stds[k] = Number(e.target.value); lastAction = ""; update(); });
    }
    $("weight").addEventListener("input", (e) => { state.weight = Number(e.target.value); lastAction = ""; update(); });
    $("count").addEventListener("change", (e) => { state.count = Number(e.target.value); update(); });
    $("seed").addEventListener("change", (e) => {
      const v = Math.round(Number(e.target.value));
      state.seed = Number.isFinite(v) && v >= 1 ? Math.min(v, 99999) : DEFAULTS.seed;
      update();
    });
    $("new-seed").addEventListener("click", () => {
      state.seed = (state.seed % 99999) + 1;
      update();
    });
    $("show-train").addEventListener("change", (e) => { state.showTrain = e.target.checked; update(); });
    $("fit").addEventListener("click", () => {
      const fitted = state.kind === 1 ? Dist.fitGaussian(train) : Dist.fitMixture(train);
      // Round to the slider steps so that the sliders show the fitted values.
      for (let k = 0; k < fitted.means.length; k++) {
        state.means[k] = clamp(Math.round(fitted.means[k] * 10) / 10, -2, 11);
        state.stds[k] = clamp(Math.round(fitted.stds[k] * 20) / 20, 0.2, 5);
      }
      if (state.kind === 2) state.weight = Math.round(fitted.weights[0] * 100) / 100;
      lastAction = "fit";
      update();
    });
    for (const button of document.querySelectorAll("[data-preset]")) {
      button.addEventListener("click", () => {
        const p = PRESETS[button.dataset.preset];
        state.kind = p.kind;
        state.means = p.means.slice();
        state.stds = p.stds.slice();
        state.weight = p.weight;
        lastAction = button.dataset.preset;
        update();
      });
    }
    $("reset").addEventListener("click", () => {
      Object.assign(state, structuredClone(DEFAULTS));
      lastAction = "";
      update();
    });
  }

  // ---- plot -----------------------------------------------------------------------------

  // The plot is drawn in CSS pixels of its container, so that text keeps its size on small screens.
  let W = 760;
  const H = 290;
  const M = { left: 44, right: 12, top: 10, bottom: 62 };
  const PLOT_H = H - M.top - M.bottom;
  const sx = (x) => M.left + ((x - Dist.X_MIN) / (Dist.X_MAX - Dist.X_MIN)) * (W - M.left - M.right);

  function el(name, attrs, parent) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.append(node);
    return node;
  }

  function densityPath(fn, sy) {
    const pts = [];
    for (let i = 0; i <= 400; i++) {
      const x = Dist.X_MIN + (i / 400) * (Dist.X_MAX - Dist.X_MIN);
      pts.push(`${sx(x).toFixed(1)},${sy(fn(x)).toFixed(1)}`);
    }
    return pts;
  }

  function draw(m, samples, hist) {
    const svg = $("plot");
    svg.replaceChildren();
    W = Math.round(Math.min(Math.max(svg.clientWidth || svg.parentElement.clientWidth, 320), 900));
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    // Vertical scale: fits target, generator and histogram, but is capped so that very narrow
    // generators do not squeeze the target.
    let yMax = 0.3;
    for (let x = Dist.X_MIN; x <= Dist.X_MAX; x += 0.02) yMax = Math.max(yMax, Dist.pdf(x, Dist.TARGET), Dist.pdf(x, m));
    for (const h of hist) yMax = Math.max(yMax, h.density);
    yMax = Math.min(yMax * 1.1, 2.2);
    const sy = (y) => M.top + PLOT_H - (Math.min(y, yMax) / yMax) * PLOT_H;
    const baseline = sy(0);

    const clip = el("clipPath", { id: "area" }, el("defs", {}, svg));
    el("rect", { x: M.left, y: M.top, width: W - M.left - M.right, height: PLOT_H + 1 }, clip);

    const g = el("g", { "clip-path": "url(#area)" }, svg);

    // Target density: filled area.
    const target = densityPath((x) => Dist.pdf(x, Dist.TARGET), sy);
    el("polygon", { points: `${sx(Dist.X_MIN)},${baseline} ${target.join(" ")} ${sx(Dist.X_MAX)},${baseline}`, fill: "var(--target-fill)", "fill-opacity": 0.75 }, g);

    // Histogram of generated samples.
    for (const h of hist) {
      if (h.density <= 0) continue;
      el("rect", {
        x: sx(h.x0) + 0.5, y: sy(h.density), width: Math.max(sx(h.x1) - sx(h.x0) - 1, 1), height: baseline - sy(h.density),
        fill: "var(--model-fill)", "fill-opacity": 0.8, stroke: "#f0a776", "stroke-width": 1,
      }, g);
    }

    // Target density: outline on top of the histogram.
    el("polyline", { points: target.join(" "), fill: "none", stroke: "var(--target)", "stroke-width": 2 }, g);

    // Generator density.
    el("polyline", { points: densityPath((x) => Dist.pdf(x, m), sy).join(" "), fill: "none", stroke: "var(--model)", "stroke-width": 3 }, g);

    // Axes.
    el("line", { x1: M.left, y1: baseline, x2: W - M.right, y2: baseline, stroke: "#9aa1ab" }, svg);
    for (let x = -2; x <= 12; x += 2) {
      el("line", { x1: sx(x), y1: baseline, x2: sx(x), y2: baseline + 4, stroke: "#9aa1ab" }, svg);
      el("text", { x: sx(x), y: baseline + 16, "text-anchor": "middle" }, svg).textContent = x;
    }
    const yStep = yMax > 1.2 ? 0.5 : yMax > 0.6 ? 0.2 : 0.1;
    for (let y = yStep; y < yMax; y += yStep) {
      el("line", { x1: M.left - 4, y1: sy(y), x2: M.left, y2: sy(y), stroke: "#9aa1ab" }, svg);
      el("text", { x: M.left - 7, y: sy(y) + 4, "text-anchor": "end" }, svg).textContent = fixed(y, 1);
    }
    el("text", { x: 12, y: M.top + PLOT_H / 2, transform: `rotate(-90 12 ${M.top + PLOT_H / 2})`, "text-anchor": "middle" }, svg).textContent = "density";

    // Rugs: generated samples (at most 300 shown) and, optionally, training data.
    const rugY = baseline + 24;
    const shown = samples.length > 300 ? samples.subarray(0, 300) : samples;
    for (const x of shown) {
      if (x < Dist.X_MIN || x > Dist.X_MAX) continue;
      el("line", { x1: sx(x), y1: rugY, x2: sx(x), y2: rugY + 10, stroke: "var(--model)", "stroke-opacity": 0.7 }, svg);
    }
    if (state.showTrain) {
      for (const x of train.subarray(0, 300)) {
        if (x < Dist.X_MIN || x > Dist.X_MAX) continue;
        el("line", { x1: sx(x), y1: rugY + 14, x2: sx(x), y2: rugY + 24, stroke: "var(--target)", "stroke-opacity": 0.6 }, svg);
      }
    }
    if (samples.length > 300) {
      el("text", { x: W - M.right, y: H - 2, "text-anchor": "end" }, svg).textContent = `ticks: first 300 of ${samples.length.toLocaleString("en-US")} samples`;
    }
  }

  // ---- statistics and messages ------------------------------------------------------------

  function describe(m, scores, ll) {
    const parts = [];
    let tone = "";
    if (lastAction === "fit" && state.kind === 1) {
      parts.push("Maximum likelihood places the single Gaussian between the two modes: it has to assign some density to every training sample. Coverage is high, but many generated samples fall where real data is rare.");
      tone = "warn";
    } else if (lastAction === "fit") {
      parts.push("With two components, maximum likelihood recovers both modes. Its log-likelihood is close to that of the target itself (it can even be slightly higher, since it is fitted to exactly these training samples).");
      tone = "good";
    } else if (lastAction === "collapse") {
      parts.push("Mode collapse: every generated sample looks realistic (high fidelity), but the left mode is never generated. The log-likelihood is very low, because the generator assigns almost no density to the training samples of the left mode.");
      tone = "warn";
    } else if (lastAction === "broad") {
      parts.push("Too broad: the generator covers all real data, but many samples are unrealistic.");
      tone = "warn";
    } else if (scores.fidelity >= 0.9 && scores.coverage >= 0.9) {
      parts.push("The generator reproduces the target well: high fidelity and high coverage.");
      tone = "good";
    } else {
      if (scores.coverage < 0.8) parts.push("Low coverage: parts of the real data are rarely or never generated.");
      if (scores.fidelity < 0.8) parts.push("Low fidelity: many generated samples fall where real data is unlikely.");
      if (!parts.length) parts.push("Adjust the generator to increase both fidelity and coverage.");
      tone = parts.length ? "warn" : "";
    }
    if (state.count <= 30) parts.push(`With only ${state.count} samples, the histogram and the fidelity fluctuate strongly between seeds, although the generator itself is unchanged.`);
    const note = $("note");
    note.textContent = parts.join(" ");
    note.className = `note ${tone}`;

    const desc = state.kind === 1
      ? `Single Gaussian with mean ${fixed(m.means[0], 1)} and standard deviation ${fixed(m.stds[0])}`
      : `Mixture of Gaussians with means ${fixed(m.means[0], 1)} and ${fixed(m.means[1], 1)}, standard deviations ${fixed(m.stds[0])} and ${fixed(m.stds[1])}, weights ${fixed(m.weights[0])} and ${fixed(m.weights[1])}`;
    $("plot-desc").textContent = `${desc}, compared with the bimodal target with modes at 2 and 7. ${state.count} generated samples. Fidelity ${percent(scores.fidelity)}, coverage ${percent(scores.coverage)}, average log-likelihood ${fixed(ll)}.`;
  }

  function update() {
    syncControls();
    const m = model();
    const samples = Dist.sample(m, currentNoise());
    const hist = Dist.histogram(samples, Dist.X_MIN, Dist.X_MAX, 50);
    const scores = Dist.compare(samples, train, m);
    const ll = Dist.logLikelihood(train, m);

    draw(m, samples, hist);
    $("stat-fidelity").textContent = percent(scores.fidelity);
    $("stat-coverage").textContent = percent(scores.coverage);
    $("meter-fidelity").style.width = percent(scores.fidelity);
    $("meter-coverage").style.width = percent(scores.coverage);
    const llEl = $("stat-ll");
    llEl.textContent = ll < -99 ? "< −99" : fixed(ll).replace("-", "−");
    const small = document.createElement("small");
    small.textContent = `target itself: ${fixed(llTarget).replace("-", "−")} (higher is better)`;
    llEl.append(small);
    describe(m, scores, ll);
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
      // Border and padding of the iframe count towards its height if the page uses border-box.
      const borderBox = window.parent.getComputedStyle(frame).boxSizing === "border-box";
      const chrome = borderBox ? frame.offsetHeight - window.innerHeight : 0;
      frame.style.height = Math.ceil(document.body.getBoundingClientRect().height) + chrome + "px";
    }).observe(document.body);
  }

  bindEvents();
  update();
  followHeight();
  // Redraw when the width of the plot changes (window resize, iframe resize).
  let lastWidth = 0;
  new ResizeObserver(() => {
    const width = $("plot").clientWidth;
    if (Math.abs(width - lastWidth) > 2) {
      lastWidth = width;
      update();
    }
  }).observe($("plot").parentElement);
})();
