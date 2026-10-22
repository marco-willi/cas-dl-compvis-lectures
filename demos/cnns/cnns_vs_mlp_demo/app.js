// Interface of the MLP-vs-CNN demo: input canvas, model cards, look inside, self-check.
"use strict";

(function () {
  const NN = window.MnistShiftNN;
  const MANIFEST = window.MNIST_SHIFT_MANIFEST;
  const SAMPLES = window.MNIST_SHIFT_SAMPLES;
  const MAPS = window.MNIST_SHIFT_MAPS;
  const $ = (id) => document.getElementById(id);

  const CANVAS = MANIFEST.canvas; // 56
  const DIGIT = MANIFEST.digit; // 28
  const MAX_OFFSET = MANIFEST.max_offset; // 28
  const ARCHS = [
    { arch: "mlp", name: "MLP" },
    { arch: "cnn_fc", name: "CNN + FC head" },
    { arch: "cnn_gap", name: "CNN + GAP head" },
  ];
  const DEFAULTS = { regime: "tl", sample: 7, dx: 0, dy: 0, net: "cnn_fc", channel: { conv1: null, conv3: null } };
  const COLORS = { ink: "#1f2328", muted: "#667085", line: "#d5d9e0", accent: "#e8910c", good: "#2a7f62", bad: "#b2182b" };
  // Accuracy color scale (0 -> 1).
  const RAMP = [[0, [238, 241, 245]], [0.5, [127, 176, 211]], [1, [11, 79, 138]]];

  const state = structuredClone(DEFAULTS);
  const weights = {}; // id -> decoded tensors
  const loading = {}; // regime -> Promise
  const digits = SAMPLES.items.map((item) => NN.decodeDigit(item, DIGIT));
  const refCache = new Map(); // "id|sample" -> forward pass at offset (0, 0)
  let results = null; // id -> {logits, acts, probs}

  const modelId = (arch, regime = state.regime) => `${arch}_${regime}`;

  // ---------------------------------------------------------------------------
  // Loading: weights are plain <script> files (works from file://); the "anywhere" networks
  // are only loaded when that regime is selected.
  // ---------------------------------------------------------------------------
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const el = document.createElement("script");
      el.src = src;
      el.onload = resolve;
      el.onerror = () => reject(new Error("Could not load " + src));
      document.head.appendChild(el);
    });
  }

  function loadRegime(regime) {
    if (!loading[regime]) {
      loading[regime] = Promise.all(ARCHS.map(({ arch }) => loadScript(`model/weights/${modelId(arch, regime)}.js`)))
        .then(() => {
          for (const { arch } of ARCHS) {
            const id = modelId(arch, regime);
            weights[id] = NN.decodeWeights(window.MNIST_SHIFT_WEIGHTS[id]);
          }
        });
    }
    return loading[regime];
  }

  // ---------------------------------------------------------------------------
  // Computation
  // ---------------------------------------------------------------------------
  function run(id, sample, dx, dy) {
    const out = NN.forward(MANIFEST.models[id].layers, weights[id], NN.place(digits[sample], dx, dy, CANVAS, DIGIT));
    out.probs = NN.softmax(out.logits);
    return out;
  }

  function reference(id) {
    const key = `${id}|${state.sample}`;
    if (!refCache.has(key)) refCache.set(key, run(id, state.sample, 0, 0));
    return refCache.get(key);
  }

  // Relative change ||a - b|| / ||b||.
  function relChange(a, b) {
    let num = 0, den = 0;
    for (let i = 0; i < a.length; i++) { num += (a[i] - b[i]) ** 2; den += b[i] ** 2; }
    return Math.sqrt(num / den);
  }

  // Channel with the largest total activation for the current digit at (0, 0).
  function strongestChannel(act) {
    const [C, H, W] = act.shape;
    let best = 0, bestSum = -1;
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let i = 0; i < H * W; i++) s += act.data[c * H * W + i];
      if (s > bestSum) { bestSum = s; best = c; }
    }
    return best;
  }

  // Sum of |fc1 weights| over neurons and channels, per position of the conv3 map.
  const fcFootprints = {};
  function fcFootprint(id) {
    if (!fcFootprints[id]) {
      const w = weights[id]["fc1.weight"];
      const [O, I] = w.shape;
      const [C, H, W] = reference(id).acts.conv3.shape;
      const out = new Float32Array(H * W);
      for (let o = 0; o < O; o++)
        for (let c = 0; c < C; c++)
          for (let p = 0; p < H * W; p++) out[p] += Math.abs(w.data[o * I + c * H * W + p]);
      fcFootprints[id] = out;
    }
    return fcFootprints[id];
  }

  // ---------------------------------------------------------------------------
  // Drawing helpers
  // ---------------------------------------------------------------------------
  function rampColor(v) {
    v = Math.min(1, Math.max(0, v));
    for (let i = 1; i < RAMP.length; i++) {
      const [t1, c1] = RAMP[i];
      const [t0, c0] = RAMP[i - 1];
      if (v <= t1) {
        const f = (v - t0) / (t1 - t0);
        return c0.map((c, k) => Math.round(c + f * (c1[k] - c)));
      }
    }
    return RAMP[RAMP.length - 1][1];
  }

  // Size the canvas backing store to its CSS size times the device pixel ratio.
  function fit(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return { ctx, w, h, dpr };
  }

  // Draw an n x m grid of values (row-major) into the whole canvas; color(v) -> [r, g, b].
  function drawGrid(canvas, values, rows, cols, color) {
    const { ctx, w, h, dpr } = fit(canvas);
    const img = new ImageData(cols, rows);
    for (let i = 0; i < rows * cols; i++) {
      const [r, g, b] = color(values[i]);
      img.data.set([r, g, b, 255], 4 * i);
    }
    const tmp = document.createElement("canvas");
    tmp.width = cols; tmp.height = rows;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, w, h);
    return { ctx, w, h, dpr };
  }

  function dashedRect(ctx, x, y, w, h, color, lineWidth) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash([lineWidth * 2.5, lineWidth * 1.8]);
    ctx.strokeRect(x + lineWidth / 2, y + lineWidth / 2, w - lineWidth, h - lineWidth);
    ctx.restore();
  }

  const gray = (max) => (v) => { const g = Math.round(255 * Math.min(1, Math.max(0, v / max))); return [g, g, g]; };
  const maxOf = (arr) => arr.reduce((m, v) => (v > m ? v : m), 1e-9);

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------
  function renderInput() {
    if (!$("demo").open) return;
    const canvas = $("input");
    const input = NN.place(digits[state.sample], state.dx, state.dy, CANVAS, DIGIT);
    const { ctx, w, dpr } = drawGrid(canvas, input.data, CANVAS, CANVAS, gray(1));
    const cell = w / CANVAS;
    const [lo, hi] = MANIFEST.regimes[state.regime].offsets;
    dashedRect(ctx, lo * cell, lo * cell, (hi - lo + DIGIT) * cell, (hi - lo + DIGIT) * cell, COLORS.accent, 2 * dpr);
    // Thin frame around the digit's 28 x 28 box.
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = dpr;
    ctx.strokeRect(state.dx * cell + dpr / 2, state.dy * cell + dpr / 2, DIGIT * cell - dpr, DIGIT * cell - dpr);

    const seen = state.dx >= lo && state.dx <= hi && state.dy >= lo && state.dy <= hi;
    $("position").innerHTML = `Position of the digit: <b>(dx, dy) = (${state.dx}, ${state.dy})</b>` +
      (seen ? " — seen during training" : "");
    const item = SAMPLES.items[state.sample];
    $("digit-label").textContent = `“${item.label}” ${state.sample + 1}/${SAMPLES.items.length}`;
  }

  function buildCards() {
    $("models").innerHTML = ARCHS.map(({ arch, name }) => `
      <div class="card" id="card-${arch}">
        <h2>${name}</h2>
        <div class="params" id="params-${arch}"></div>
        <p class="verdict" id="verdict-${arch}"></p>
        <canvas class="bars" id="bars-${arch}" aria-label="Predicted probabilities for the digits 0 to 9"></canvas>
        <canvas class="map" id="map-${arch}" aria-label="Test accuracy for every position of the digit; click to move the digit there"></canvas>
        <p class="map-caption" id="mapcap-${arch}"></p>
      </div>`).join("");
    for (const { arch } of ARCHS) {
      $(`map-${arch}`).addEventListener("click", (event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const n = MAX_OFFSET + 1;
        moveTo(Math.floor(((event.clientX - rect.left) / rect.width) * n), Math.floor(((event.clientY - rect.top) / rect.height) * n));
      });
    }
  }

  function renderBars(canvas, probs, label) {
    const { ctx, w, h, dpr } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const pred = probs.indexOf(Math.max(...probs));
    const font = 11 * dpr;
    const base = h - font * 1.5;
    const slot = w / 10;
    ctx.font = `${font}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let k = 0; k < 10; k++) {
      const bh = Math.max(probs[k] * (base - 2 * dpr), probs[k] > 0.005 ? dpr : 0);
      ctx.fillStyle = k === pred ? (pred === label ? COLORS.good : COLORS.bad) : "#9aa3b2";
      ctx.fillRect(k * slot + slot * 0.18, base - bh, slot * 0.64, bh);
      ctx.fillStyle = k === label ? COLORS.ink : COLORS.muted;
      ctx.font = `${k === label ? "700 " : ""}${font}px system-ui, sans-serif`;
      ctx.fillText(String(k), k * slot + slot / 2, base + font * 0.8);
    }
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(0, base + dpr / 2); ctx.lineTo(w, base + dpr / 2); ctx.stroke();
  }

  function renderMap(canvas, acc) {
    const n = MAX_OFFSET + 1;
    const { ctx, w, dpr } = drawGrid(canvas, acc.flat(), n, n, rampColor);
    const cell = w / n;
    const [lo, hi] = MANIFEST.regimes[state.regime].offsets;
    dashedRect(ctx, lo * cell, lo * cell, (hi - lo + 1) * cell, (hi - lo + 1) * cell, COLORS.accent, 1.5 * dpr);
    // Current position: ring around the cell.
    const r = Math.max(cell * 0.9, 4 * dpr);
    // Keep the ring inside the canvas at the borders.
    const center = (i) => Math.min(w - r - 2 * dpr, Math.max(r + 2 * dpr, (i + 0.5) * cell));
    for (const [color, width] of [["#fff", 4 * dpr], [COLORS.ink, 2 * dpr]]) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.arc(center(state.dx), center(state.dy), r, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }

  function renderCards() {
    const label = SAMPLES.items[state.sample].label;
    for (const { arch } of ARCHS) {
      const id = modelId(arch);
      const meta = MANIFEST.models[id];
      const { probs } = results[id];
      const pred = probs.indexOf(Math.max(...probs));
      $(`params-${arch}`).textContent = `${meta.params.toLocaleString("en-US")} parameters`;
      const verdict = $(`verdict-${arch}`);
      verdict.className = "verdict " + (pred === label ? "good" : "bad");
      verdict.innerHTML = `Prediction <b>${pred}</b> (${Math.round(probs[pred] * 100)} %)`;
      renderBars($(`bars-${arch}`), probs, label);
      const acc = MAPS.maps[id].acc;
      renderMap($(`map-${arch}`), acc);
      $(`mapcap-${arch}`).innerHTML = `Test accuracy at this position: <b>${Math.round(acc[state.dy][state.dx] * 100)} %</b>`;
    }
  }

  function renderInside() {
    if (!$("inside").open) return;
    const id = modelId(state.net);
    const { acts } = results[id];
    const ref = reference(id);
    for (const layer of ["conv1", "conv3"]) {
      if (state.channel[layer] === null) state.channel[layer] = strongestChannel(ref.acts[layer]);
    }
    for (const button of $("ctl-net").children) button.setAttribute("aria-pressed", String(button.dataset.value === state.net));
    $("fm-head").parentElement.classList.toggle("square", state.net === "cnn_fc");

    const input = NN.place(digits[state.sample], state.dx, state.dy, CANVAS, DIGIT);
    const box = (canvas) => {
      const { ctx, w, dpr } = fit(canvas);
      const cell = w / CANVAS;
      dashedRect(ctx, state.dx * cell, state.dy * cell, DIGIT * cell, DIGIT * cell, COLORS.accent, 1.5 * dpr);
    };
    drawGrid($("fm-input"), input.data, CANVAS, CANVAS, gray(1));
    box($("fm-input"));
    for (const layer of ["conv1", "conv3"]) {
      const act = acts[layer];
      const [C, H, W] = act.shape;
      const c = state.channel[layer];
      const plane = act.data.subarray(c * H * W, (c + 1) * H * W);
      drawGrid($(`fm-${layer}`), plane, H, W, gray(maxOf(plane)));
      box($(`fm-${layer}`));
      $(`ch-${layer}`).textContent = `${c + 1}/${C}`;
      $(`cap-${layer}`).innerHTML = `${layer}, channel ${c + 1}: <b>${H} × ${W}</b>` + (layer === "conv3" ? " (after 2 poolings)" : "");
    }
    if (state.net === "cnn_gap") renderGapHead(acts.head.data, ref.acts.head.data);
    else renderFcHead(id, acts.head.data, ref.acts.head.data);
  }

  // GAP output: current values (dark) over the values at the training position (light).
  function renderGapHead(current, atRef) {
    const { ctx, w, h, dpr } = fit($("fm-head"));
    ctx.clearRect(0, 0, w, h);
    const n = current.length;
    const max = Math.max(maxOf(current), maxOf(atRef));
    const pad = 4 * dpr;
    const slot = (w - 2 * pad) / n;
    for (const [values, color, frac] of [[atRef, "#c9ced8", 0.9], [current, COLORS.ink, 0.45]]) {
      ctx.fillStyle = color;
      for (let i = 0; i < n; i++) {
        const bh = (values[i] / max) * (h - 2 * pad);
        ctx.fillRect(pad + i * slot + (slot * (1 - frac)) / 2, h - pad - bh, slot * frac, bh);
      }
    }
    $("cap-head").innerHTML = `GAP output, the input of the classifier: ${n} values, one per conv3 channel ` +
      `(dark: now, light: at the top-left position). Change: <b>${Math.round(relChange(current, atRef) * 100)} %</b>`;
  }

  // FC head: where the weights of the first fully connected layer are large, per position.
  function renderFcHead(id, current, atRef) {
    const foot = fcFootprint(id);
    const side = Math.round(Math.sqrt(foot.length));
    const max = maxOf(foot);
    // Square map in the same coordinate frame as the other maps.
    const { ctx, w, dpr } = drawGrid($("fm-head"), foot, side, side, (v) => rampColor(v / max));
    const cell = w / CANVAS;
    dashedRect(ctx, state.dx * cell, state.dy * cell, DIGIT * cell, DIGIT * cell, COLORS.accent, 1.5 * dpr);
    $("cap-head").innerHTML = `Weights of the first FC layer per position (sum of |w|, dark = large). The FC ` +
      `layer receives all ${current.length.toLocaleString("en-US")} conv3 values, each with its own weights. ` +
      `Change of these values compared with the top-left position: <b>${Math.round(relChange(current, atRef) * 100)} %</b>`;
  }

  // Five training examples per regime: sample digits placed like the training data.
  const EXAMPLES = {
    tl: [0, 11, 22, 33, 44].map((sample) => ({ sample, dx: 0, dy: 0 })),
    all: [[5, 21, 4], [16, 9, 17], [27, 27, 26], [38, 2, 23], [49, 15, 10]].map(([sample, dx, dy]) => ({ sample, dx, dy })),
  };
  function renderExamples() {
    if (!$("data").open) return;
    for (const [regime, items] of Object.entries(EXAMPLES)) {
      const box = $(`examples-${regime}`);
      if (!box.children.length) box.innerHTML = items.map(() => "<canvas></canvas>").join("");
      items.forEach(({ sample, dx, dy }, i) => {
        const canvas = box.children[i];
        canvas.title = `“${SAMPLES.items[sample].label}” at (dx, dy) = (${dx}, ${dy})`;
        drawGrid(canvas, NN.place(digits[sample], dx, dy, CANVAS, DIGIT).data, CANVAS, CANVAS, gray(1));
      });
    }
  }

  function render() {
    renderExamples();
    renderInput();
    if (!results) return;
    if ($("demo").open) renderCards();
    renderInside();
  }

  // ---------------------------------------------------------------------------
  // Updates: at most one computation per animation frame.
  // ---------------------------------------------------------------------------
  let pending = false;
  function update() {
    renderInput();
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      if (!loading[state.regime]) return;
      loadRegime(state.regime).then(() => {
        results = {};
        for (const { arch } of ARCHS) {
          const id = modelId(arch);
          results[id] = run(id, state.sample, state.dx, state.dy);
        }
        $("status").hidden = true;
        render();
      }).catch(showError);
    });
  }

  function moveTo(dx, dy) {
    dx = Math.min(MAX_OFFSET, Math.max(0, dx));
    dy = Math.min(MAX_OFFSET, Math.max(0, dy));
    if (dx === state.dx && dy === state.dy) return;
    state.dx = dx;
    state.dy = dy;
    update();
  }

  function setRegime(regime) {
    state.regime = regime;
    for (const button of $("ctl-regime").children) button.setAttribute("aria-pressed", String(button.dataset.value === regime));
    if (!weights[modelId("mlp", regime)]) {
      $("status").hidden = false;
      $("status").textContent = "Loading networks …";
    }
    loadRegime(regime);
    update();
  }

  function setSample(index) {
    const n = SAMPLES.items.length;
    state.sample = (index + n) % n;
    state.channel = { conv1: null, conv3: null };
    update();
  }

  function showError(error) {
    $("status").hidden = false;
    $("status").className = "status error";
    $("status").textContent = String(error && error.message || error);
  }

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------
  const inputCanvas = $("input");
  let grab = null; // offset of the pointer within the digit, in canvas pixels
  function canvasPixel(event) {
    const rect = inputCanvas.getBoundingClientRect();
    return [((event.clientX - rect.left) / rect.width) * CANVAS, ((event.clientY - rect.top) / rect.height) * CANVAS];
  }
  inputCanvas.addEventListener("pointerdown", (event) => {
    const [x, y] = canvasPixel(event);
    const inside = x >= state.dx && x < state.dx + DIGIT && y >= state.dy && y < state.dy + DIGIT;
    grab = inside ? [x - state.dx, y - state.dy] : [DIGIT / 2, DIGIT / 2];
    inputCanvas.setPointerCapture(event.pointerId);
    inputCanvas.classList.add("dragging");
    moveTo(Math.round(x - grab[0]), Math.round(y - grab[1]));
  });
  inputCanvas.addEventListener("pointermove", (event) => {
    if (!grab) return;
    const [x, y] = canvasPixel(event);
    moveTo(Math.round(x - grab[0]), Math.round(y - grab[1]));
  });
  const endDrag = () => { grab = null; inputCanvas.classList.remove("dragging"); };
  inputCanvas.addEventListener("pointerup", endDrag);
  inputCanvas.addEventListener("pointercancel", endDrag);
  inputCanvas.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 4 : 1;
    const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (!moves[event.key]) return;
    event.preventDefault();
    moveTo(state.dx + moves[event.key][0], state.dy + moves[event.key][1]);
  });

  $("prev").addEventListener("click", () => setSample(state.sample - 1));
  $("next").addEventListener("click", () => setSample(state.sample + 1));
  $("reset").addEventListener("click", () => { state.dx = 0; state.dy = 0; update(); });
  $("ctl-regime").addEventListener("click", (event) => {
    const value = event.target.closest("button")?.dataset.value;
    if (value) setRegime(value);
  });
  $("ctl-net").addEventListener("click", (event) => {
    const value = event.target.closest("button")?.dataset.value;
    if (!value) return;
    state.net = value;
    state.channel = { conv1: null, conv3: null };
    render();
  });
  document.querySelectorAll("[data-step]").forEach((button) => button.addEventListener("click", () => {
    const [layer, delta] = button.dataset.step.split(":");
    const C = results[modelId(state.net)].acts[layer].shape[0];
    state.channel[layer] = (state.channel[layer] + Number(delta) + C) % C;
    renderInside();
  }));
  for (const id of ["data", "demo", "inside"]) $(id).addEventListener("toggle", render);
  new ResizeObserver(() => render()).observe(document.body);

  // ---------------------------------------------------------------------------
  // Self-check against the PyTorch references (model/reference.js)
  // ---------------------------------------------------------------------------
  async function selfTest() {
    const out = $("selftest-result");
    out.className = "status";
    out.textContent = "Running …";
    try {
      if (!window.MNIST_SHIFT_REFERENCE) await loadScript("model/reference.js");
      await Promise.all([loadRegime("tl"), loadRegime("all")]);
      const REF = window.MNIST_SHIFT_REFERENCE;
      const lines = [];
      let ok = true;
      for (const [id, cases] of Object.entries(REF.models)) {
        let err = 0;
        for (const c of cases) {
          const { logits, acts } = run(id, c.sample, c.dx, c.dy);
          c.logits.forEach((v, i) => (err = Math.max(err, Math.abs(v - logits[i]))));
          for (const layer of ["conv1", "conv3"]) {
            const ref = c[layer + "_first20"];
            if (ref) ref.forEach((v, i) => (err = Math.max(err, Math.abs(v - acts[layer].data[i]))));
          }
        }
        const pass = err < REF.tolerance;
        ok = ok && pass;
        lines.push(`${pass ? "✓" : "✗"} ${id}: max. deviation ${err.toExponential(1)}`);
      }
      out.className = "status" + (ok ? "" : " error");
      out.innerHTML = (ok ? "All networks match PyTorch." : "Mismatch!") + "<br>" + lines.join("<br>");
    } catch (error) {
      out.className = "status error";
      out.textContent = String(error && error.message || error);
    }
  }
  $("selftest").addEventListener("click", selfTest);

  // ---------------------------------------------------------------------------
  // Height: an embedding page on the same site can add `data-auto-height` to the iframe; the
  // iframe then follows the height of the demo instead of using a fixed height.
  // ---------------------------------------------------------------------------
  let autoHeight = false;
  try {
    autoHeight = Boolean(window.frameElement && window.frameElement.hasAttribute("data-auto-height"));
  } catch (error) {
    // cross-origin parent: keep the fixed height
  }
  document.documentElement.classList.toggle("auto-height", autoHeight);
  if (autoHeight) {
    const frame = window.frameElement;
    new ResizeObserver(() => {
      const borderBox = window.parent.getComputedStyle(frame).boxSizing === "border-box";
      const chrome = borderBox ? frame.offsetHeight - window.innerHeight : 0;
      frame.style.height = Math.ceil(document.body.getBoundingClientRect().height) + chrome + "px";
    }).observe(document.body);
  }

  // ---------------------------------------------------------------------------
  // Start
  // ---------------------------------------------------------------------------
  $("scale-bar").style.background = `linear-gradient(to right, ${RAMP.map(([t, c]) => `rgb(${c}) ${t * 100}%`).join(", ")})`;
  buildCards();
  const params = new URLSearchParams(location.search);
  if (params.has("inside")) $("inside").open = true;
  setRegime(params.get("regime") === "all" ? "all" : "tl");
  if (params.has("selftest")) { $("verify").open = true; selfTest(); }
})();
