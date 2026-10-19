(() => {
  "use strict";

  const META = window.ATTENTION_META;
  const { grid: G, patchSize: P, imageSize: S, layers: L, heads: H, prefixTokens: R, tokens: N } = META;

  // Inferno color map, linearly interpolated between 9 stops.
  const INFERNO = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#fcffa4"]
    .map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));
  const LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (INFERNO.length - 1);
    const j = Math.min(Math.floor(t), INFERNO.length - 2);
    const f = t - j;
    for (let c = 0; c < 3; c++) LUT[i * 3 + c] = INFERNO[j][c] * (1 - f) + INFERNO[j + 1][c] * f;
  }

  const OVERLAY_ALPHA = 0.7;
  const QUERY_COLOR = "#22d3ee";

  const INITIAL = {
    layer: Math.floor(L / 2), // 0-based; start in a middle layer
    head: -1, // -1 = mean over heads
    query: R + Math.floor(G * 0.6) * G + Math.floor(G * 0.72), // token index; patch on the right cow
    preview: null, // token index while hovering
    shared: false,
    smooth: false,
  };
  const state = { ...INITIAL };

  let quant = null; // Uint8Array (L, H, N, N), each row scaled to its maximum
  let scales = null; // Float32Array (L, H, N), row maxima
  const image = new Image();
  const $ = (id) => document.getElementById(id);

  function decodeBase64(text) {
    const bin = atob(text);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  async function load() {
    if (typeof DecompressionStream === "undefined") throw new Error("This browser cannot decompress the attention data (DecompressionStream is missing). Please use a current browser.");
    const stream = new Blob([decodeBase64(window.ATTENTION_GZ)]).stream().pipeThrough(new DecompressionStream("gzip"));
    quant = new Uint8Array(await new Response(stream).arrayBuffer());
    scales = new Float32Array(decodeBase64(window.ATTENTION_SCALES).buffer);
    if (quant.length !== L * H * N * N) throw new Error("Attention data has an unexpected size.");
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("Could not load " + META.image));
      image.src = META.image;
    });
  }

  // Attention weights of one query token over all N key tokens (sums to ~1).
  function row(layer, head, query) {
    const out = new Float32Array(N);
    const heads = head < 0 ? [...Array(H).keys()] : [head];
    for (const h of heads) {
      const lh = layer * H + h;
      const scale = scales[lh * N + query] / 255 / heads.length;
      const base = (lh * N + query) * N;
      for (let k = 0; k < N; k++) out[k] += quant[base + k] * scale;
    }
    return out;
  }

  function patchMax(weights) {
    let max = 0;
    for (let k = R; k < N; k++) max = Math.max(max, weights[k]);
    return max;
  }

  const isPatch = (token) => token >= R;
  const patchPos = (token) => [Math.floor((token - R) / G), (token - R) % G]; // [row, col]

  // Draws the image with the attention of `weights` (patch keys only) on a canvas.
  const heat = document.createElement("canvas");
  heat.width = heat.height = G;
  const heatCtx = heat.getContext("2d");
  const heatData = heatCtx.createImageData(G, G);

  function drawMap(canvas, weights, max, query, { gridLines = false, hover = null } = {}) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    for (let k = 0; k < G * G; k++) {
      const v = Math.min(255, Math.round((weights[R + k] / max) * 255)) || 0;
      heatData.data.set([LUT[v * 3], LUT[v * 3 + 1], LUT[v * 3 + 2], 255], k * 4);
    }
    heatCtx.putImageData(heatData, 0, 0);

    ctx.globalAlpha = 1;
    ctx.drawImage(image, 0, 0, w, w);
    ctx.globalAlpha = OVERLAY_ALPHA;
    ctx.imageSmoothingEnabled = state.smooth;
    ctx.drawImage(heat, 0, 0, w, w);
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = 1;

    const cell = w / G;
    if (gridLines) {
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < G; i++) {
        ctx.moveTo(Math.round(i * cell) + 0.5, 0); ctx.lineTo(Math.round(i * cell) + 0.5, w);
        ctx.moveTo(0, Math.round(i * cell) + 0.5); ctx.lineTo(w, Math.round(i * cell) + 0.5);
      }
      ctx.stroke();
    }
    const outline = (token, color, width, dash) => {
      if (token === null || !isPatch(token)) return;
      const [r, c] = patchPos(token);
      ctx.setLineDash(dash);
      ctx.lineWidth = width + 2;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.strokeRect(c * cell + 1, r * cell + 1, cell - 2, cell - 2);
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.strokeRect(c * cell + 1, r * cell + 1, cell - 2, cell - 2);
      ctx.setLineDash([]);
    };
    outline(query, QUERY_COLOR, Math.max(1.5, w / 200), []);
    if (hover !== null && hover !== query) outline(hover, "#fff", Math.max(1, w / 300), [4, 3]);
  }

  function queryName(token) {
    if (!isPatch(token)) return `register ${token + 1}`;
    const [r, c] = patchPos(token);
    return `patch (row ${r + 1}, column ${c + 1})`;
  }

  const pct = (x) => (x * 100).toFixed(x < 0.1 ? 1 : 0) + "%";

  function render() {
    const query = state.preview ?? state.query;
    const rows = [...Array(L).keys()].map((l) => row(l, state.head, query));
    const maxes = rows.map(patchMax);
    const shared = Math.max(...maxes);
    const scaleFor = (l) => (state.shared ? shared : maxes[l]);
    const weights = rows[state.layer];

    drawMap($("view"), weights, scaleFor(state.layer), query, { gridLines: true, hover: state.preview === null ? null : state.query });
    rows.forEach((w, l) => drawMap(thumbs[l].canvas, w, scaleFor(l), query));

    // Statistics
    let reg = 0;
    for (let k = 0; k < R; k++) reg += weights[k];
    let mass = 0, dist = 0;
    const [qr, qc] = isPatch(query) ? patchPos(query) : [0, 0];
    for (let k = R; k < N; k++) {
      const [r, c] = patchPos(k);
      mass += weights[k];
      dist += weights[k] * Math.hypot(r - qr, c - qc) * P;
    }
    $("stat-self").textContent = isPatch(query) ? pct(weights[query]) : "–";
    $("stat-reg").textContent = pct(reg);
    $("stat-dist").textContent = isPatch(query) ? Math.round(dist / mass) + " px" : "–";
    $("scale-max").textContent = (state.shared ? "max over layers " : "max ") + pct(scaleFor(state.layer));

    // Controls and labels
    $("layer").value = state.layer + 1;
    $("layer-value").textContent = `${state.layer + 1} / ${L}`;
    $("heads").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.head === state.head)));
    $("queries").querySelectorAll("button").forEach((b) => {
      const pressed = b.id === "query-patch" ? isPatch(state.query) : +b.dataset.token === state.query;
      b.setAttribute("aria-pressed", String(pressed));
    });
    $("scale-layer").setAttribute("aria-pressed", String(!state.shared));
    $("scale-shared").setAttribute("aria-pressed", String(state.shared));
    thumbs.forEach((t, l) => t.button.setAttribute("aria-pressed", String(l === state.layer)));
    const head = state.head < 0 ? "mean over heads" : `head ${state.head + 1}`;
    $("view-caption").textContent = `Query: ${queryName(query)}${state.preview !== null ? " (preview, click to select)" : ""} · layer ${state.layer + 1} · ${head}`;
  }

  // Build the head and register buttons and the layer thumbnails.
  const thumbs = [];
  function build() {
    const heads = $("heads");
    [-1, ...Array(H).keys()].forEach((h) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.head = h;
      b.textContent = h < 0 ? "Mean" : String(h + 1);
      if (h >= 0) b.setAttribute("aria-label", `Head ${h + 1}`);
      b.addEventListener("click", () => { state.head = h; render(); });
      heads.append(b);
    });
    for (let r = 0; r < R; r++) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.token = r;
      b.textContent = `Register ${r + 1}`;
      b.addEventListener("click", () => {
        if (isPatch(state.query)) lastPatch = state.query;
        state.query = r;
        render();
      });
      $("queries").append(b);
    }
    $("registers-identical").hidden = !META.registersIdentical;
    for (let l = 0; l < L; l++) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb";
      button.setAttribute("aria-label", `Show layer ${l + 1}`);
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 128;
      button.append(canvas, `Layer ${l + 1}`);
      button.addEventListener("click", () => { state.layer = l; render(); });
      $("thumbs").append(button);
      thumbs.push({ button, canvas });
    }
    $("layer").max = L;
    $("fine").textContent = `Model: ${META.model} (${L} layers, ${H} heads, ${G} × ${G} patches of ${P} × ${P} px, ${R} register tokens, no [CLS] token). ` +
      "The overlay shows only the attention on image patches; the share on register tokens is listed separately. " +
      "Mean attention distance: average distance between the query and the patches it attends to, weighted by attention.";
  }

  let lastPatch = state.query;

  function tokenAt(event) {
    const rect = $("view").getBoundingClientRect();
    const c = Math.floor(((event.clientX - rect.left) / rect.width) * G);
    const r = Math.floor(((event.clientY - rect.top) / rect.height) * G);
    if (r < 0 || c < 0 || r >= G || c >= G) return null;
    return R + r * G + c;
  }

  function bindEvents() {
    const view = $("view");
    view.addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;
      const t = tokenAt(e);
      if (t !== state.preview) { state.preview = t; render(); }
    });
    view.addEventListener("pointerleave", () => { if (state.preview !== null) { state.preview = null; render(); } });
    view.addEventListener("pointerdown", (e) => {
      const t = tokenAt(e);
      if (t === null) return;
      state.query = lastPatch = t;
      state.preview = null;
      render();
    });
    view.addEventListener("keydown", (e) => {
      const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
      if (!moves[e.key]) return;
      e.preventDefault();
      const [r, c] = patchPos(isPatch(state.query) ? state.query : lastPatch);
      const nr = Math.min(G - 1, Math.max(0, r + moves[e.key][0]));
      const nc = Math.min(G - 1, Math.max(0, c + moves[e.key][1]));
      state.query = lastPatch = R + nr * G + nc;
      render();
    });
    $("query-patch").addEventListener("click", () => { state.query = lastPatch; render(); });
    $("reset").addEventListener("click", () => {
      Object.assign(state, INITIAL);
      lastPatch = INITIAL.query;
      $("smooth").checked = INITIAL.smooth;
      render();
    });
    $("layer").addEventListener("input", (e) => { state.layer = +e.target.value - 1; render(); });
    $("scale-layer").addEventListener("click", () => { state.shared = false; render(); });
    $("scale-shared").addEventListener("click", () => { state.shared = true; render(); });
    $("smooth").addEventListener("change", (e) => { state.smooth = e.target.checked; render(); });
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

  load()
    .then(() => {
      build();
      bindEvents();
      $("app").hidden = false;
      $("strip").hidden = false;
      render();
      followHeight();
    })
    .catch((error) => {
      const box = $("error");
      box.textContent = error.message;
      box.hidden = false;
      followHeight();
    });
})();
