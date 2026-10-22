(() => {
  "use strict";

  const META = window.PATCH_META;
  const [GC, GR] = META.grid; // patches per image: columns, rows
  const [TW, TH] = META.tileSize; // size of one image in the collage, px
  const D = META.dims;
  const IMAGES = META.titles.length;
  const PER_IMAGE = GC * GR;
  const N = IMAGES * PER_IMAGE;

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

  const OVERLAY_ALPHA = 0.65;
  const QUERY_COLOR = "#22d3ee";
  const GAP = 6; // px between images on the canvas

  const INITIAL = {
    query: 12 * GC + 13, // patch index; ear of the elephant in the first image
    preview: null, // patch index while hovering
    perQuery: false,
    smooth: false,
  };
  const state = { ...INITIAL };

  let feats = null; // Float32Array (N, D), rows l2-normalized
  const collage = new Image();
  const $ = (id) => document.getElementById(id);

  function decodeBase64(text) {
    const bin = atob(text);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  async function load() {
    const q = new Int8Array(decodeBase64(window.PATCH_FEATURES).buffer);
    if (q.length !== N * D) throw new Error("Feature data has an unexpected size.");
    feats = new Float32Array(N * D);
    for (let i = 0; i < N; i++) {
      let norm = 0;
      for (let k = 0; k < D; k++) norm += q[i * D + k] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let k = 0; k < D; k++) feats[i * D + k] = q[i * D + k] / norm;
    }
    await new Promise((resolve, reject) => {
      collage.onload = resolve;
      collage.onerror = () => reject(new Error("Could not load " + META.collage));
      collage.src = META.collage;
    });
  }

  // Cosine similarity of patch `query` to all N patches.
  function similarities(query) {
    const out = new Float32Array(N);
    const qo = query * D;
    for (let i = 0; i < N; i++) {
      let s = 0;
      const o = i * D;
      for (let k = 0; k < D; k++) s += feats[qo + k] * feats[o + k];
      out[i] = s;
    }
    return out;
  }

  // Layout: 4 images per row, 2 on narrow screens.
  let cols = 0; // set by layout()
  const tileOrigin = (img) => [(img % cols) * (TW + GAP), Math.floor(img / cols) * (TH + GAP)];
  const sourceOrigin = (img) => [(img % META.collageCols) * TW, Math.floor(img / META.collageCols) * TH];
  const patchCenter = (p) => {
    const img = Math.floor(p / PER_IMAGE);
    const k = p % PER_IMAGE;
    const [x, y] = tileOrigin(img);
    return [x + ((k % GC) + 0.5) * (TW / GC), y + (Math.floor(k / GC) + 0.5) * (TH / GR)];
  };

  function layout() {
    const next = window.innerWidth < 640 ? 2 : 4;
    const canvas = $("view");
    if (next === cols) return;
    cols = next;
    const rows = Math.ceil(IMAGES / cols);
    canvas.width = cols * TW + (cols - 1) * GAP;
    canvas.height = rows * TH + (rows - 1) * GAP;
  }

  // One small canvas per image holds its heat map (GC x GR pixels).
  const heats = [...Array(IMAGES)].map(() => {
    const c = document.createElement("canvas");
    c.width = GC;
    c.height = GR;
    return c;
  });

  function render() {
    const query = state.preview ?? state.query;
    const sims = similarities(query);
    let lo = 0, hi = 1;
    if (state.perQuery) {
      lo = Infinity; hi = -Infinity;
      for (const s of sims) { lo = Math.min(lo, s); hi = Math.max(hi, s); }
    }

    const canvas = $("view");
    const ctx = canvas.getContext("2d");
    const px = canvas.width / canvas.getBoundingClientRect().width || 1; // canvas px per CSS px
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const best = [];
    for (let img = 0; img < IMAGES; img++) {
      // Heat map of this image
      const hctx = heats[img].getContext("2d");
      const data = hctx.createImageData(GC, GR);
      let bi = 0;
      for (let k = 0; k < PER_IMAGE; k++) {
        const s = sims[img * PER_IMAGE + k];
        if (s > sims[img * PER_IMAGE + bi]) bi = k;
        const v = Math.max(0, Math.min(255, Math.round(((s - lo) / (hi - lo)) * 255))) || 0;
        data.data.set([LUT[v * 3], LUT[v * 3 + 1], LUT[v * 3 + 2], 255], k * 4);
      }
      hctx.putImageData(data, 0, 0);
      best.push(img * PER_IMAGE + bi);

      const [x, y] = tileOrigin(img);
      const [sx, sy] = sourceOrigin(img);
      ctx.globalAlpha = 1;
      ctx.drawImage(collage, sx, sy, TW, TH, x, y, TW, TH);
      ctx.globalAlpha = OVERLAY_ALPHA;
      ctx.imageSmoothingEnabled = state.smooth;
      ctx.drawImage(heats[img], x, y, TW, TH);
      ctx.imageSmoothingEnabled = true;
      ctx.globalAlpha = 1;

      // Title (top left) and best match (bottom right) of this image
      const isQueryImage = img === Math.floor(query / PER_IMAGE);
      const label = (text, right, color) => {
        const size = 12 * px;
        const pad = 4 * px;
        const w = ctx.measureText(text).width + 2 * pad;
        const lx = right ? x + TW - w : x;
        const ly = right ? y + TH - size - 2 * pad : y;
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(lx, ly, w, size + 2 * pad);
        ctx.fillStyle = color;
        ctx.fillText(text, lx + pad, ly + pad);
      };
      ctx.font = `600 ${Math.round(12 * px)}px system-ui, sans-serif`;
      ctx.textBaseline = "top";
      label(META.titles[img].replace(" - ", " – "), false, "#fff");
      label(isQueryImage ? "query" : "max " + sims[best[img]].toFixed(2), true, isQueryImage ? QUERY_COLOR : "#fff");
    }

    // Best match per image (ring), query point (cross) and selected query while previewing
    const ring = (p, color, r, dash = []) => {
      const [cx, cy] = patchCenter(p);
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      ctx.lineWidth = 4 * px;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.stroke();
      ctx.lineWidth = 2 * px;
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.setLineDash([]);
    };
    const cross = (p, color, size) => {
      const [cx, cy] = patchCenter(p);
      for (const [lw, c] of [[5 * px, "rgba(0,0,0,0.7)"], [2.5 * px, color]]) {
        ctx.lineWidth = lw;
        ctx.strokeStyle = c;
        ctx.beginPath();
        ctx.moveTo(cx - size, cy); ctx.lineTo(cx + size, cy);
        ctx.moveTo(cx, cy - size); ctx.lineTo(cx, cy + size);
        ctx.stroke();
      }
    };
    best.forEach((p, img) => { if (img !== Math.floor(query / PER_IMAGE)) ring(p, "#fff", 9 * px); });
    if (state.preview !== null && state.preview !== state.query) ring(state.query, QUERY_COLOR, 7 * px, [3 * px, 3 * px]);
    cross(query, QUERY_COLOR, 9 * px);

    $("scale-fixed").setAttribute("aria-pressed", String(!state.perQuery));
    $("scale-query").setAttribute("aria-pressed", String(state.perQuery));
    $("scale-min").textContent = lo.toFixed(2).replace(/^0\.00$/, "0");
    $("scale-max").textContent = hi.toFixed(2).replace(/^1\.00$/, "1");
    const img = Math.floor(query / PER_IMAGE);
    const k = query % PER_IMAGE;
    $("view-caption").textContent = `Query: ${META.titles[img].replace(" - ", " – ")}, patch (row ${Math.floor(k / GC) + 1}, column ${(k % GC) + 1})` +
      (state.preview !== null ? " (preview, click to select)" : "") +
      ". White rings: most similar patch in each other image.";
  }

  function patchAt(event) {
    const canvas = $("view");
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
    const c = Math.floor(x / (TW + GAP));
    const r = Math.floor(y / (TH + GAP));
    const img = r * cols + c;
    const lx = x - c * (TW + GAP);
    const ly = y - r * (TH + GAP);
    if (c < 0 || c >= cols || img < 0 || img >= IMAGES || lx >= TW || ly >= TH) return null;
    return img * PER_IMAGE + Math.floor(ly / (TH / GR)) * GC + Math.floor(lx / (TW / GC));
  }

  function bindEvents() {
    const view = $("view");
    view.addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;
      const p = patchAt(e);
      if (p !== state.preview) { state.preview = p; render(); }
    });
    view.addEventListener("pointerleave", () => { if (state.preview !== null) { state.preview = null; render(); } });
    view.addEventListener("pointerdown", (e) => {
      const p = patchAt(e);
      if (p === null) return;
      state.query = p;
      state.preview = null;
      render();
    });
    view.addEventListener("keydown", (e) => {
      const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
      if (!moves[e.key]) return;
      e.preventDefault();
      const img = Math.floor(state.query / PER_IMAGE);
      const k = state.query % PER_IMAGE;
      const r = Math.min(GR - 1, Math.max(0, Math.floor(k / GC) + moves[e.key][0]));
      const c = Math.min(GC - 1, Math.max(0, (k % GC) + moves[e.key][1]));
      state.query = img * PER_IMAGE + r * GC + c;
      render();
    });
    $("scale-fixed").addEventListener("click", () => { state.perQuery = false; render(); });
    $("scale-query").addEventListener("click", () => { state.perQuery = true; render(); });
    $("smooth").addEventListener("change", (e) => { state.smooth = e.target.checked; render(); });
    $("reset").addEventListener("click", () => {
      Object.assign(state, INITIAL);
      $("smooth").checked = INITIAL.smooth;
      render();
    });
    window.addEventListener("resize", () => { layout(); render(); });
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
      $("fine").textContent = `Model: ${META.model} (ViT-B/16), images resized to ${META.inputSize[0]} × ${META.inputSize[1]} px, ` +
        `so each image has ${GC} × ${GR} patches of ${META.patchSize} × ${META.patchSize} px. The [CLS] token is not used. ` +
        `To keep the download small, the ${META.fullDims}-dimensional patch embeddings are stored as ${META.dims} principal components ` +
        `(8-bit); the similarities differ from the full embeddings by ${META.meanError.toFixed(3)} on average.`;
      $("app").hidden = false;
      layout();
      bindEvents();
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
