// Interface: the same stage (inputs | network | output) for every model family, one progress
// slider, "Play" and "New sample". Computation in sampling.js, texts in config.js.
(function () {
  "use strict";

  const S = globalThis.Sampling, D = globalThis.IMAGE_DATA, C = globalThis.CONFIG;
  const W = D.width, H = D.height, P = D.codebook.patch, K = D.codebook.k;
  const GW = W / P, N_TOKENS = GW * (H / P), N_IMAGES = D.images.length;
  const codebook = Uint8Array.from(atob(D.codebook.data), (c) => c.charCodeAt(0));
  const codeColors = Array.from({ length: K }, (_, k) => {
    const rgb = [0, 0, 0];
    for (let i = 0; i < P * P; i++) for (let c = 0; c < 3; c++) rgb[c] += codebook[(k * P * P + i) * 3 + c] / (P * P);
    return `rgb(${rgb.map(Math.round).join(",")})`;
  });
  const COLOR = { ink: "#1f2328", model: "#d55e00", modelLight: "#f0b48a", target: "#1f4e9c", muted: "#5b6472", line: "#d5d9e0" };

  const params = new URLSearchParams(window.location.search);
  const EMBED = params.get("embed") === "1";
  const state = {
    family: C.families.includes(params.get("family")) ? params.get("family") : "ar",
    mode: { vae: "generate", diffusion: "reverse" },
    seed: 0,
    step: 0,
    playing: false,
    history: [0], // seeds of the samples shown in the strip (GAN, VAE)
    showPaths: params.get("paths") === "1", // optional: pixel paths and "Run again" (diffusion, flow matching)
    run: 0, // "Run again (same noise)": new fresh noise for diffusion, nothing changes for flow matching
    previousPaths: null, // pixel paths of the previous run
  };
  const modes = C[state.family].modes;
  if (modes && params.get("mode") in modes) state.mode[state.family] = params.get("mode");

  const $ = (id) => document.getElementById(id);
  let images = []; // one Float32Array per image
  const cache = new Map();
  const cached = (key, fn) => {
    if (!cache.has(key)) cache.set(key, fn());
    return cache.get(key);
  };

  // ---- model state -----------------------------------------------------------------------

  const imageIndex = (seed = state.seed) => seed % N_IMAGES;
  const noise = () => cached(`noise${state.seed}`, () => S.normals(9973 + state.seed, 3 * W * H));
  const reverseStates = () =>
    cached(`reverse${state.seed}_${state.run}`, () => S.diffusionReverse(images[imageIndex()], 9973 + state.seed, 31337 + 101 * state.run + state.seed));
  // forward process as a Markov chain: the reverse construction, read backwards (x_0, ..., x_T)
  const forwardStates = () =>
    cached(`forward${state.seed}`, () => S.diffusionReverse(images[imageIndex()], 9973 + state.seed, 4242 + state.seed).reverse());
  const mode = () => state.mode[state.family];
  const iterative = () => state.family === "diffusion" || state.family === "fm";
  const pathsOn = () => state.showPaths && iterative();
  const canRerun = () => pathsOn() && (state.family === "fm" || mode() === "reverse");

  // positions (x, y) of the pixels whose values are traced, and their colors
  const PIXELS = [[40, 25], [128, 72], [210, 128]];
  const PIXEL_COLORS = ["#d55e00", "#1f4e9c", "#009e73"];

  function maxStep() {
    return { ar: N_TOKENS, diffusion: S.T_DIFFUSION, fm: S.N_FLOW }[state.family] || 0;
  }

  // image of diffusion or flow matching after `step` steps
  function stateAt(step) {
    if (state.family === "fm") return S.flow(images[imageIndex()], noise(), step / S.N_FLOW);
    return (mode() === "reverse" ? reverseStates() : forwardStates())[step];
  }

  // value (green channel) of each traced pixel at every step
  function pixelPaths() {
    return cached(`paths${state.family}${mode()}${state.seed}_${state.run}`, () => {
      const steps = Array.from({ length: maxStep() + 1 }, (_, s) => s);
      return PIXELS.map(([x, y]) => {
        const q = 3 * (y * W + x) + 1;
        if (state.family === "fm") return steps.map((s) => (1 - s / S.N_FLOW) * noise()[q] + (s / S.N_FLOW) * images[imageIndex()][q]);
        return steps.map((s) => stateAt(s)[q]);
      });
    });
  }

  function timeLabel(step) {
    if (state.family === "ar") return `token ${step} / ${N_TOKENS}`;
    if (state.family === "fm") return `t = ${(step / S.N_FLOW).toFixed(2)}`;
    return `t = ${mode() === "reverse" ? S.T_DIFFUSION - step : step}`;
  }

  function evaluations() {
    const f = state.family;
    if (f === "ar" || f === "fm") return state.step;
    if (f === "diffusion") return mode() === "reverse" ? state.step : 0;
    if (f === "gan") return 1;
    return mode() === "generate" ? 1 : 2;
  }

  // ---- stage -----------------------------------------------------------------------------

  const th = "<sub>θ</sub>";
  const prior = "~ <i>N</i>(<b>0</b>, <b>I</b>)";
  const node = (id, caption, cls) =>
    `<figure class="node ${cls}"><canvas id="${id}" width="${W}" height="${H}"></canvas><figcaption id="${id}Caption">${caption}</figcaption></figure>`;
  const bars = (caption, cls = "bars") => `<figure class="node ${cls}"><canvas id="bars"></canvas><figcaption>${caption}</figcaption></figure>`;
  const net = (label, sub, cls = "") => `<div class="net ${cls}">${label}<small>${sub}</small></div>`;
  const arrow = '<div class="arrow" aria-hidden="true">→</div>';
  const paths = () => (pathsOn() ? '<figure class="node paths"><canvas id="paths"></canvas><figcaption>value of the 3 marked pixels</figcaption></figure>' : "");

  function stageHTML() {
    const f = state.family;
    if (f === "ar") {
      return node("out", "", "big") + arrow + net(`<i>p</i>${th}`, "network") + arrow
        + bars(`<i>p</i>${th}(<i>x</i><sub><i>i</i></sub> | <i>x</i><sub>1</sub>, …, <i>x</i><sub><i>i</i>−1</sub>) over ${K} tokens`);
    }
    if (f === "gan") {
      return bars(`<b>z</b> ${prior}`, "latent") + arrow + net(`<i>g</i>${th}`, "generator") + arrow
        + node("out", `<b>x</b> = <i>g</i>${th}(<b>z</b>)`, "big");
    }
    if (f === "vae") {
      const off = mode() === "generate" ? "off" : "";
      return node("in", "input <b>x</b>", `small ${off}`) + arrow + net("<i>q</i><sub>ψ</sub>", "encoder", off) + arrow
        + bars(mode() === "generate" ? `<b>z</b> ${prior}` : "<b>z</b> = <b>μ</b> + <b>σ</b> ⊙ <b>ε</b>", "latent") + arrow
        + net(`<i>p</i>${th}`, "decoder") + arrow + node("out", "output", "big");
    }
    if (f === "diffusion" && mode() === "forward") {
      return node("in", "data <b>x</b><sub>0</sub>", "small") + arrow + net("+ noise", "fixed, no network", "fixed") + arrow
        + node("out", "", "big") + paths();
    }
    if (f === "diffusion") {
      return node("in", `<b>x</b><sub><i>T</i></sub> ${prior}`, "small") + arrow
        + net(`<i>f</i>${th}(<b>x</b><sub><i>t</i></sub>, <i>t</i>)`, "denoiser, once per step") + arrow + node("out", "", "big") + paths();
    }
    return node("in", `<b>x</b><sub>0</sub> ${prior}`, "small") + arrow
      + net(`<i>v</i>${th}(<b>x</b><sub><i>t</i></sub>, <i>t</i>)`, "velocity, once per step") + arrow + node("out", "", "big") + paths();
  }

  // ---- drawing ---------------------------------------------------------------------------

  // traced pixels over the steps: the band marks the value range of images, the dashed lines
  // are the previous run with the same starting noise
  function drawPaths() {
    const { ctx: b, w, h } = fit($("paths"));
    const n = maxStep(), l = 4, r = 4, top = 6, bottom = 16;
    const X = (s) => l + (s / n) * (w - l - r);
    const Y = (v) => top + ((2.6 - Math.max(-2.6, Math.min(2.6, v))) / 5.2) * (h - top - bottom);
    b.fillStyle = "#eef1f5";
    b.fillRect(X(0), Y(1), X(n) - X(0), Y(-1) - Y(1));
    b.fillStyle = COLOR.muted;
    b.textBaseline = "bottom";
    b.textAlign = "left";
    b.fillText(timeLabel(0), X(0), h);
    b.textAlign = "right";
    b.fillText(timeLabel(n), X(n), h);
    const line = (values, upTo) => {
      b.beginPath();
      for (let s = 0; s <= upTo; s++) s ? b.lineTo(X(s), Y(values[s])) : b.moveTo(X(s), Y(values[s]));
      b.stroke();
    };
    if (state.previousPaths) {
      b.setLineDash([3, 3]);
      b.globalAlpha = 0.5;
      state.previousPaths.forEach((values, k) => ((b.strokeStyle = PIXEL_COLORS[k]), line(values, n)));
      b.setLineDash([]);
      b.globalAlpha = 1;
    }
    b.lineWidth = 1.5;
    pixelPaths().forEach((values, k) => ((b.strokeStyle = PIXEL_COLORS[k]), line(values, state.step)));
    b.lineWidth = 1;
    b.strokeStyle = COLOR.muted;
    b.beginPath(), b.moveTo(X(state.step) + 0.5, top), b.lineTo(X(state.step) + 0.5, h - bottom), b.stroke();
  }

  // circles around the traced pixels on the output image
  function markPixels(canvas) {
    const ctx = canvas.getContext("2d");
    PIXELS.forEach(([x, y], k) => {
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#fff";
      ctx.beginPath(), ctx.arc(x + 0.5, y + 0.5, 5, 0, 2 * Math.PI), ctx.stroke();
      ctx.lineWidth = 2;
      ctx.strokeStyle = PIXEL_COLORS[k];
      ctx.stroke();
    });
  }

  function put(canvas, x) {
    canvas.getContext("2d").putImageData(new ImageData(S.toRGBA(x), W, H), 0, 0);
  }

  // canvas sized to its CSS box, drawing in CSS pixels
  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = "11px system-ui, sans-serif";
    return { ctx, w, h };
  }

  function drawTokens() {
    const x = images[imageIndex()], rgba = S.toRGBA(x);
    for (let t = state.step; t < N_TOKENS; t++) {
      const r0 = Math.floor(t / GW) * P, c0 = (t % GW) * P;
      for (let r = r0; r < r0 + P; r++) rgba.fill(236, 4 * (r * W + c0), 4 * (r * W + c0 + P));
      for (let r = r0; r < r0 + P; r++) for (let c = c0; c < c0 + P; c++) rgba[4 * (r * W + c) + 3] = 255;
    }
    const ctx = $("out").getContext("2d");
    ctx.putImageData(new ImageData(rgba, W, H), 0, 0);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let c = P; c < W; c += P) ctx.moveTo(c, 0), ctx.lineTo(c, H);
    for (let r = P; r < H; r += P) ctx.moveTo(0, r), ctx.lineTo(W, r);
    ctx.stroke();
    if (state.step < N_TOKENS) {
      ctx.strokeStyle = COLOR.model;
      ctx.lineWidth = 2;
      ctx.strokeRect((state.step % GW) * P + 1, Math.floor(state.step / GW) * P + 1, P - 2, P - 2);
    }
    $("outCaption").innerHTML = state.step < N_TOKENS
      ? `tokens <i>x</i><sub>1</sub>, …, <i>x</i><sub>${state.step}</sub>; next token outlined`
      : "all tokens generated";

    const { ctx: b, w, h } = fit($("bars"));
    const base = h - 16, top = 16, bw = (w - 8) / K;
    b.strokeStyle = COLOR.line;
    b.beginPath(), b.moveTo(4, base + 0.5), b.lineTo(w - 4, base + 0.5), b.stroke();
    if (state.step >= N_TOKENS) {
      b.fillStyle = COLOR.muted;
      b.textAlign = "center";
      b.fillText("image complete", w / 2, h / 2);
      return;
    }
    const { probs, token } = S.tokenDistribution(x, W, codebook, P, state.step);
    probs.forEach((p, k) => {
      const bh = p * (base - top), x0 = 4 + k * bw;
      b.fillStyle = k === token ? COLOR.model : COLOR.modelLight;
      b.fillRect(x0 + 1, base - bh, bw - 2, bh);
      b.fillStyle = codeColors[k];
      b.fillRect(x0 + 1, base + 3, bw - 2, 9);
    });
    b.fillStyle = COLOR.model;
    b.textAlign = "center";
    b.fillText(probs[token].toFixed(2), 4 + (token + 0.5) * bw, base - probs[token] * (base - top) - 4);
  }

  // 2-D latent space: the prior N(0, I) (shading, circles at radius 1 and 2), earlier samples
  // numbered as in the strip, the current z; for the VAE encoder also q(z | x) (ellipse at 2 sigma)
  function drawLatent() {
    const { ctx: b, w, h } = fit($("bars"));
    const R = 2.6, cx = w / 2, cy = h / 2, s = Math.min(w, h) / (2 * R + 0.4); // z in [-R, R]
    const px = (z) => [cx + z[0] * s, cy - z[1] * s];
    const shade = b.createRadialGradient(cx, cy, 0, cx, cy, R * s);
    shade.addColorStop(0, "rgba(91, 100, 114, 0.22)");
    shade.addColorStop(1, "rgba(91, 100, 114, 0)");
    b.fillStyle = shade;
    b.beginPath(), b.arc(cx, cy, R * s, 0, 2 * Math.PI), b.fill();
    b.strokeStyle = COLOR.line;
    b.beginPath(), b.moveTo(cx - R * s, cy), b.lineTo(cx + R * s, cy), b.moveTo(cx, cy - R * s), b.lineTo(cx, cy + R * s), b.stroke();
    b.setLineDash([3, 3]);
    for (const r of [1, 2]) b.beginPath(), b.arc(cx, cy, r * s, 0, 2 * Math.PI), b.stroke();
    b.setLineDash([]);
    b.fillStyle = COLOR.muted;
    b.textBaseline = "middle";
    b.fillText("z₁", cx + R * s + 4, cy);
    b.textBaseline = "top";
    b.fillText("z₂", cx + 5, cy - R * s);
    b.textBaseline = "middle";
    b.textAlign = "center";

    const last = state.history.length - 1;
    const { z, enc } = latentFor(state.history[last]);
    if (enc) {
      const [mx, my] = px(enc.mu);
      b.fillStyle = "rgba(31, 78, 156, 0.15)";
      b.strokeStyle = COLOR.target;
      b.beginPath(), b.ellipse(mx, my, 2 * enc.sigma[0] * s, 2 * enc.sigma[1] * s, 0, 0, 2 * Math.PI), b.fill(), b.stroke();
      b.fillStyle = COLOR.target;
      b.beginPath(), b.arc(mx, my, 2.5, 0, 2 * Math.PI), b.fill();
    }
    state.history.slice(0, last).forEach((seed, k) => {
      const [x, y] = px(latentFor(seed).z);
      b.fillStyle = COLOR.modelLight;
      b.beginPath(), b.arc(x, y, 3.5, 0, 2 * Math.PI), b.fill();
      b.fillStyle = COLOR.muted;
      b.fillText(String(k + 1), x, y - 9);
    });
    const [x, y] = px(z);
    b.fillStyle = COLOR.model;
    b.strokeStyle = "#fff";
    b.lineWidth = 1.5;
    b.beginPath(), b.arc(x, y, 5, 0, 2 * Math.PI), b.fill(), b.stroke();
    b.fillStyle = COLOR.ink;
    b.fillText(String(last + 1), x, y - 11);
  }

  function latentFor(seed) {
    if (state.family === "vae" && mode() === "reconstruct") {
      const enc = S.encode(imageIndex(seed), N_IMAGES);
      return { z: S.sampleLatent(enc.mu, enc.sigma, 500 + seed), enc };
    }
    return { z: S.latent(900 + seed), enc: null };
  }

  function outputFor(seed) {
    return images[imageIndex(seed)];
  }

  // ---- strip -----------------------------------------------------------------------------

  let stripKey = "";
  function renderStrip() {
    const f = state.family, iterative = f === "diffusion" || f === "fm";
    const steps = iterative ? Array.from({ length: C.stripSize }, (_, k) => Math.round((k * maxStep()) / (C.stripSize - 1))) : [];
    const key = iterative ? `${f}${mode()}${state.seed}_${state.run}` : f === "ar" ? "ar" : `${f}${mode()}${state.history.join()}`;
    if (key !== stripKey) {
      stripKey = key;
      const items = iterative
        ? steps.map((s) => ({ x: stateAt(s), caption: timeLabel(s) }))
        : f === "ar" ? []
        : state.history.map((seed, k) => ({ x: outputFor(seed), caption: `sample ${k + 1}` }));
      $("strip").innerHTML = items
        .map((it, k) => `<figure><canvas width="${W}" height="${H}"></canvas><figcaption>${it.caption}</figcaption></figure>`)
        .join("");
      $("strip").querySelectorAll("canvas").forEach((cv, k) => put(cv, items[k].x));
    }
    const figs = $("strip").querySelectorAll("figure");
    const current = iterative ? steps.filter((s) => s <= state.step).length - 1 : figs.length - 1;
    figs.forEach((fig, k) => fig.classList.toggle("current", k === current));
  }

  // ---- render ----------------------------------------------------------------------------

  function render() {
    const f = state.family;
    if (f === "ar") drawTokens();
    if (f === "gan" || f === "vae") {
      drawLatent();
      put($("out"), outputFor(state.seed));
      if (f === "vae") put($("in"), images[imageIndex()]);
    }
    if (f === "diffusion" || f === "fm") {
      put($("in"), f === "diffusion" && mode() === "forward" ? images[imageIndex()] : noise());
      put($("out"), stateAt(state.step));
      if (pathsOn()) markPixels($("out")), drawPaths();
      $("outCaption").innerHTML = `<b>x</b><sub><i>t</i></sub>, ${timeLabel(state.step)}`;
    }
    $("slider").value = state.step;
    $("back").disabled = state.step <= 0;
    $("next").disabled = state.step >= maxStep();
    $("sliderOut").textContent = timeLabel(state.step);
    const n = evaluations();
    $("evals").innerHTML = f === "diffusion" && mode() === "forward"
      ? "network evaluations: <b>0</b>"
      : `network evaluations: <b>${n}</b>${f === "gan" || f === "vae" ? " per sample" : ""}`;
    renderStrip();
  }

  function build() {
    stop();
    const f = state.family, cfg = C[f];
    $("stage").innerHTML = stageHTML();
    $("modes").innerHTML = cfg.modes
      ? Object.entries(cfg.modes).map(([m, label]) => `<button data-mode="${m}" aria-pressed="${mode() === m}">${label}</button>`).join("")
      : "";
    $("subtitle").innerHTML = typeof cfg.subtitle === "string" ? cfg.subtitle : cfg.subtitle[mode()];
    $("sliderWrap").hidden = $("play").hidden = !maxStep();
    $("back").hidden = $("next").hidden = f !== "ar";
    $("pathsWrap").hidden = !iterative();
    $("showPaths").checked = state.showPaths;
    $("rerun").hidden = !canRerun();
    state.previousPaths = null;
    $("slider").max = maxStep();
    $("tabs").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.family === f));
    $("openFull").href = `index.html?family=${f}${cfg.modes ? `&mode=${mode()}` : ""}${state.showPaths ? "&paths=1" : ""}`;
    state.history = [state.seed];
    stripKey = "";
    render();
  }

  // ---- playback --------------------------------------------------------------------------

  let raf = 0;
  function play() {
    if (state.step >= maxStep()) state.step = 0;
    state.playing = true;
    $("play").textContent = "Pause";
    const msPerStep = (1000 * C.playSeconds[state.family]) / maxStep();
    let last = performance.now();
    const tick = (now) => {
      if (!state.playing) return;
      const n = Math.floor((now - last) / msPerStep);
      if (n > 0) {
        last += n * msPerStep;
        state.step = Math.min(maxStep(), state.step + n);
        render();
      }
      if (state.step >= maxStep()) return stop();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  function stop() {
    state.playing = false;
    cancelAnimationFrame(raf);
    $("play").textContent = "Play";
  }

  // ---- events ----------------------------------------------------------------------------

  $("play").addEventListener("click", () => (state.playing ? stop() : play()));
  $("slider").addEventListener("input", (e) => {
    stop();
    state.step = Number(e.target.value);
    render();
  });
  $("newSample").addEventListener("click", () => {
    cache.clear();
    state.seed++;
    state.history = [...state.history, state.seed].slice(-C.stripSize);
    state.step = 0;
    state.previousPaths = null;
    stop();
    render();
    if (maxStep()) play();
  });
  // reset: the first image (canonical example), the first step, no earlier samples
  $("reset").addEventListener("click", () => {
    cache.clear();
    Object.assign(state, { seed: 0, step: 0, run: 0, previousPaths: null });
    build();
  });
  const stepBy = (d) => {
    stop();
    state.step = Math.max(0, Math.min(maxStep(), state.step + d));
    render();
  };
  $("back").addEventListener("click", () => stepBy(-1));
  $("next").addEventListener("click", () => stepBy(1));
  $("showPaths").addEventListener("change", (e) => {
    state.showPaths = e.target.checked;
    build();
  });
  $("rerun").addEventListener("click", () => {
    state.previousPaths = pixelPaths();
    for (const key of cache.keys()) if (key.startsWith("reverse")) cache.delete(key);
    state.run++;
    state.step = 0;
    stop();
    render();
    play();
  });
  $("modes").addEventListener("click", (e) => {
    const m = e.target.closest("button")?.dataset.mode;
    if (!m) return;
    state.mode[state.family] = m;
    state.step = 0;
    build();
  });
  $("tabs").addEventListener("click", (e) => {
    const f = e.target.closest("button")?.dataset.family;
    if (!f) return;
    state.family = f;
    state.step = 0;
    build();
  });
  window.addEventListener("resize", () => images.length && render());

  // ---- start -----------------------------------------------------------------------------

  if (EMBED) {
    $("head").hidden = $("tabs").hidden = true;
    $("openFull").hidden = false;
  } else {
    $("tabs").innerHTML = C.families.map((f) => `<button data-family="${f}">${C[f].tab}</button>`).join("");
  }

  // the images are data URIs, so their pixels can be read from file:// as well
  Promise.all(D.images.map((im) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const ctx = Object.assign(document.createElement("canvas"), { width: W, height: H }).getContext("2d");
      ctx.drawImage(img, 0, 0);
      resolve(S.fromRGBA(ctx.getImageData(0, 0, W, H).data));
    };
    img.onerror = reject;
    img.src = im.src;
  }))).then((xs) => {
    images = xs;
    build();
    followHeight();
  });

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
})();
