// Interface of the demo: controls, plot, statistics. Computation is in fid.js.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";

  // Generator parameters (the real features are fixed in fid.js and never changed here).
  const DEFAULTS = Object.freeze({
    shift: [0.8, 0.6],
    spread: 1.4,
    clusters: [true, true, true],
    count: 300,
    seed: 1,
    showGauss: true,
    manifold: "none",
  });
  const PRESETS = {
    good: { shift: [0, 0], spread: 1, clusters: [true, true, true] },
    shift: { shift: [1.5, 1.1], spread: 1, clusters: [true, true, true] },
    collapse: { shift: [0, 0], spread: 1, clusters: [false, true, false] },
    broad: { shift: [0, 0], spread: 2, clusters: [true, true, true] },
    narrow: { shift: [0, 0], spread: 0.4, clusters: [true, true, true] },
  };

  const state = structuredClone(DEFAULTS);
  let lastAction = "";
  let noise = null;
  let noiseKey = "";
  let real = [];
  let realKey = 0;

  // Noise is drawn once per seed and sample count and reused while the parameters change.
  function currentNoise() {
    const key = `${state.count}:${state.seed}`;
    if (key !== noiseKey) {
      noise = Fid.noise(state.count, state.seed);
      noiseKey = key;
    }
    return noise;
  }

  function currentReal() {
    if (realKey !== state.count) {
      real = Fid.realFeatures(state.count);
      realKey = state.count;
    }
    return real;
  }

  const fixed = (x, d = 2) => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const percent = (x) => `${Math.round(x * 100)}%`;
  const params = () => ({ shift: state.shift, spread: state.spread, clusters: state.clusters });

  // ---- controls -------------------------------------------------------------------------

  function syncControls() {
    for (const [k, id] of [[0, "shift-x"], [1, "shift-y"]]) {
      $(id).value = state.shift[k];
      $(`${id}-out`).textContent = fixed(state.shift[k], 1);
    }
    $("spread").value = state.spread;
    $("spread-out").textContent = fixed(state.spread, 2);
    for (const k of [0, 1, 2]) $(`cluster${k}`).checked = state.clusters[k];
    $("count").value = String(state.count);
    $("seed").value = state.seed;
    $("show-gauss").checked = state.showGauss;
    $("manifold").value = state.manifold;
    $("legend-gauss").hidden = !state.showGauss;
    for (const button of document.querySelectorAll("[data-preset]")) {
      button.classList.toggle("active", button.dataset.preset === lastAction);
    }
  }

  function bindEvents() {
    for (const [k, id] of [[0, "shift-x"], [1, "shift-y"]]) {
      $(id).addEventListener("input", (e) => { state.shift[k] = Number(e.target.value); lastAction = ""; update(); });
    }
    $("spread").addEventListener("input", (e) => { state.spread = Number(e.target.value); lastAction = ""; update(); });
    for (const k of [0, 1, 2]) {
      $(`cluster${k}`).addEventListener("change", (e) => { state.clusters[k] = e.target.checked; lastAction = ""; update(); });
    }
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
    $("show-gauss").addEventListener("change", (e) => { state.showGauss = e.target.checked; update(); });
    $("manifold").addEventListener("change", (e) => { state.manifold = e.target.value; update(); });
    for (const button of document.querySelectorAll("[data-preset]")) {
      button.addEventListener("click", () => {
        Object.assign(state, structuredClone(PRESETS[button.dataset.preset]));
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

  // The plot is drawn in CSS pixels of its container, so that text keeps its size on small
  // screens. Both axes share one scale, so circles stay circles.
  let W = 520;
  let H = 470;
  let scale = 1;
  const sx = (x) => (x - Fid.X_MIN) * scale;
  const sy = (y) => H - (y - Fid.Y_MIN) * scale;

  function el(name, attrs, parent) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.append(node);
    return node;
  }

  function drawGaussian(g, color, parent) {
    const e = Fid.ellipse(g, 2);
    const deg = (-e.angle * 180) / Math.PI; // y axis points down in SVG
    el("ellipse", {
      cx: sx(e.cx), cy: sy(e.cy), rx: e.rx * scale, ry: e.ry * scale,
      transform: `rotate(${deg.toFixed(2)} ${sx(e.cx).toFixed(1)} ${sy(e.cy).toFixed(1)})`,
      fill: "none", stroke: color, "stroke-width": 2.5,
    }, parent);
    const [x, y] = [sx(g.mean[0]), sy(g.mean[1])];
    el("path", { d: `M${x - 6},${y - 6}L${x + 6},${y + 6}M${x - 6},${y + 6}L${x + 6},${y - 6}`, stroke: "#fff", "stroke-width": 5 }, parent);
    el("path", { d: `M${x - 6},${y - 6}L${x + 6},${y + 6}M${x - 6},${y + 6}L${x + 6},${y - 6}`, stroke: color, "stroke-width": 2.5 }, parent);
  }

  function drawPoints(points, color, inside, parent) {
    const r = state.count >= 1000 ? 2 : state.count >= 300 ? 2.6 : 3.4;
    points.forEach(([x, y], i) => {
      const out = inside && !inside[i];
      el("circle", {
        cx: sx(x).toFixed(1), cy: sy(y).toFixed(1), r: out ? r + 1.2 : r,
        fill: out ? "#fff" : color, "fill-opacity": out ? 1 : 0.6,
        stroke: out ? color : "none", "stroke-width": out ? 1.6 : 0,
      }, parent);
    });
  }

  function draw(gen, pr) {
    const svg = $("plot");
    svg.replaceChildren();
    W = Math.round(Math.min(Math.max(svg.clientWidth || svg.parentElement.clientWidth, 280), 760));
    scale = W / (Fid.X_MAX - Fid.X_MIN);
    H = Math.round((Fid.Y_MAX - Fid.Y_MIN) * scale);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const clip = el("clipPath", { id: "area" }, el("defs", {}, svg));
    el("rect", { x: 0, y: 0, width: W, height: H }, clip);
    el("rect", { x: 0.5, y: 0.5, width: W - 1, height: H - 1, fill: "#fff", stroke: "var(--line)" }, svg);
    const g = el("g", { "clip-path": "url(#area)" }, svg);

    // Balls of the selected set: their union approximates its manifold.
    if (state.manifold !== "none") {
      const [centers, radii, color] = state.manifold === "real"
        ? [real, pr.realRadii, "var(--target-fill)"]
        : [gen, pr.genRadii, "var(--model-fill)"];
      const balls = el("g", { opacity: 0.55 }, g);
      centers.forEach(([x, y], i) => {
        el("circle", { cx: sx(x).toFixed(1), cy: sy(y).toFixed(1), r: (radii[i] * scale).toFixed(1), fill: color }, balls);
      });
    }

    // Points; with balls shown, the points of the other set outside the manifold are hollow.
    drawPoints(real, "var(--target)", state.manifold === "gen" ? pr.realInside : null, g);
    drawPoints(gen, "var(--model)", state.manifold === "real" ? pr.genInside : null, g);

    if (state.showGauss && gen.length > 1) {
      drawGaussian(Fid.meanCov(real), "var(--target)", g);
      drawGaussian(Fid.meanCov(gen), "var(--model)", g);
    }

    if (!gen.length) {
      el("text", { x: W / 2, y: 24, "text-anchor": "middle" }, svg).textContent = "Select at least one cluster.";
    }
  }

  // ---- statistics and messages ------------------------------------------------------------

  const MESSAGES = {
    good: ["The generator reproduces the real features: FID close to 0, high precision and recall.", "good"],
    shift: ["Shifted: many generated points lie outside the real manifold (low precision). FID is high, almost entirely due to the mean term.", "warn"],
    collapse: ["Mode collapse: every generated point looks realistic (high precision), but two of three clusters are never generated (low recall). FID is high as well, but on its own it does not tell whether fidelity or coverage is the problem: compare with Shifted.", "warn"],
    broad: ["Too broad: the generator covers all real features (high recall), but many generated points are unrealistic (lower precision).", "warn"],
    narrow: ["Too narrow: each cluster is generated with too little variation. Recall drops clearly, but FID barely changes: a single Gaussian fitted to three clusters hardly notices the narrower clusters. Precision and recall reveal what FID misses.", "warn"],
  };

  function describe(f, pr) {
    const parts = [];
    let tone = "";
    if (MESSAGES[lastAction]) {
      [parts[0], tone] = MESSAGES[lastAction];
    } else if (pr.precision >= 0.85 && pr.recall >= 0.85) {
      parts.push("High precision and recall: the generated features match the real ones well.");
      tone = "good";
    } else {
      if (pr.precision < 0.85) parts.push("Low precision: many generated points lie outside the real manifold (low fidelity).");
      if (pr.recall < 0.85) parts.push("Low recall: parts of the real features are not covered by the generated ones (low coverage).");
      tone = "warn";
    }
    if (state.count <= 50) parts.push(`With only ${state.count} points per set, all values fluctuate between seeds, and FID is biased upwards: even a perfect generator does not reach 0.`);
    const note = $("note");
    note.textContent = parts.join(" ");
    note.className = `note ${tone}`;

    const active = ["left", "right", "top"].filter((_, k) => state.clusters[k]);
    $("plot-desc").textContent = `Generated clusters: ${active.join(", ") || "none"}, shifted by (${fixed(state.shift[0], 1)}, ${fixed(state.shift[1], 1)}), spread ${fixed(state.spread)} times the real spread. ${state.count} points per set. FID ${fixed(f.fid)}, precision ${percent(pr.precision)}, recall ${percent(pr.recall)}.`;
  }

  function update() {
    syncControls();
    const realPts = currentReal();
    const gen = Fid.sample(params(), currentNoise());
    const f = gen.length > 1 ? Fid.fid(realPts, gen) : null;
    const pr = Fid.precisionRecall(realPts, gen);

    draw(gen, pr);
    const fidEl = $("stat-fid");
    fidEl.textContent = f ? fixed(f.fid) : "–";
    if (f) {
      const small = document.createElement("small");
      small.textContent = `mean term ${fixed(f.meanTerm)} + covariance term ${fixed(f.covTerm)}`;
      fidEl.append(small);
    }
    $("stat-precision").textContent = percent(pr.precision);
    $("stat-recall").textContent = percent(pr.recall);
    $("meter-precision").style.width = percent(pr.precision);
    $("meter-recall").style.width = percent(pr.recall);
    $("legend-out").hidden = state.manifold === "none";
    $("legend-out-text").textContent = state.manifold === "real"
      ? "generated point outside the real manifold"
      : "real point outside the generated manifold";
    if (f) describe(f, pr);
    else {
      $("note").textContent = "Select at least one cluster.";
      $("note").className = "note";
    }
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
