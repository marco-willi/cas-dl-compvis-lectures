(() => {
  "use strict";

  const SCENES = window.HERO || []; // { meta, features }, one per photo; all with the same grid
  const [GC, GR] = SCENES.length ? SCENES[0].meta.grid : [0, 0]; // patches: columns, rows
  const D = SCENES.length ? SCENES[0].meta.dims : 0;
  const N = GC * GR;

  // Color scale per query: white at CUT of the way from the lowest similarity of this query to
  // the highest (the query itself), warm above (similar), cool below. 0.7 separates zebras
  // from giraffes (body similarity about 0.7); 0.5 puts both on the warm side.
  const CUT = 0.7;
  const MAP_SCALE = 12; // map canvas px per patch
  const QUERY_COLOR = "#22d3ee";

  // Coolwarm color map (Moreland), linearly interpolated between 10 stops.
  const COOLWARM = ["#3b4cc0", "#5d7ce6", "#82a6fb", "#aac7fd", "#cdd9ec", "#ead4c8", "#f7b89c", "#f18d6f", "#d95847", "#b40426"]
    .map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));
  const LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (COOLWARM.length - 1);
    const j = Math.min(Math.floor(t), COOLWARM.length - 2);
    const f = t - j;
    for (let c = 0; c < 3; c++) LUT[i * 3 + c] = COOLWARM[j][c] * (1 - f) + COOLWARM[j + 1][c] * f;
  }

  const state = { scene: 0, query: 0, preview: null };

  let feats = null; // Float32Array (N, D) of the current scene, rows l2-normalized
  let photo = null; // Image of the current scene
  const heat = document.createElement("canvas"); // GC x GR, one pixel per patch
  const $ = (id) => document.getElementById(id);

  function decodeBase64(text) {
    const bin = atob(text);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function decode(meta, text) {
    const q = new Int8Array(decodeBase64(text).buffer);
    if (meta.grid[0] !== GC || meta.grid[1] !== GR || q.length !== N * D) {
      throw new Error(`Feature data of ${meta.image} has an unexpected size.`);
    }
    const out = new Float32Array(N * D);
    for (let i = 0; i < N; i++) {
      let norm = 0;
      for (let k = 0; k < D; k++) norm += q[i * D + k] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let k = 0; k < D; k++) out[i * D + k] = q[i * D + k] / norm;
    }
    return out;
  }

  // Decodes the features and loads the photo of every scene.
  async function load() {
    if (!SCENES.length) throw new Error("No feature data found.");
    for (const scene of SCENES) {
      scene.feats = decode(scene.meta, scene.features);
      scene.photo = new Image();
      await new Promise((resolve, reject) => {
        scene.photo.onload = resolve;
        scene.photo.onerror = () => reject(new Error("Could not load " + scene.meta.image));
        scene.photo.src = scene.meta.image;
      });
    }
    heat.width = GC;
    heat.height = GR;
  }

  function select(index) {
    const { meta } = SCENES[index];
    state.scene = index;
    state.query = meta.initial[1] * GC + meta.initial[0];
    state.preview = null;
    ({ feats, photo } = SCENES[index]);
    $("view").width = photo.width;
    $("view").height = photo.height;
    $("map").width = GC * MAP_SCALE;
    $("map").height = GR * MAP_SCALE;
    [...$("picker").children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)));
    $("credit").textContent = `Foto: ${meta.credit}, Unsplash. Modell: DINOv3 (ViT-B/16), ${GC} × ${GR} Patches.`;
    render();
  }

  // Draws the similarity of patch `query` to all patches into `heat`.
  function paintHeat(query) {
    const hctx = heat.getContext("2d");
    const data = hctx.createImageData(GC, GR);
    const qo = query * D;
    const sims = new Float32Array(N);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < N; i++) {
      let s = 0;
      const o = i * D;
      for (let k = 0; k < D; k++) s += feats[qo + k] * feats[o + k];
      sims[i] = s;
      lo = Math.min(lo, s);
      hi = Math.max(hi, s);
    }
    const cut = lo + CUT * (hi - lo);
    for (let i = 0; i < N; i++) {
      const s = sims[i];
      const t = s < cut ? 0.5 * (s - lo) / (cut - lo || 1) : 0.5 + 0.5 * (s - cut) / (hi - cut || 1);
      const v = Math.max(0, Math.min(255, Math.round(t * 255)));
      data.data.set([LUT[v * 3], LUT[v * 3 + 1], LUT[v * 3 + 2], 255], i * 4);
    }
    hctx.putImageData(data, 0, 0);
  }

  function drawCross(canvas, query) {
    const ctx = canvas.getContext("2d");
    const { width: W, height: H } = canvas;
    const px = W / canvas.getBoundingClientRect().width || 1; // canvas px per CSS px
    const cx = ((query % GC) + 0.5) * (W / GC);
    const cy = (Math.floor(query / GC) + 0.5) * (H / GR);
    const size = 10 * px;
    for (const [lw, c] of [[5 * px, "rgba(0,0,0,0.7)"], [2.5 * px, QUERY_COLOR]]) {
      ctx.lineWidth = lw;
      ctx.strokeStyle = c;
      ctx.beginPath();
      ctx.moveTo(cx - size, cy); ctx.lineTo(cx + size, cy);
      ctx.moveTo(cx, cy - size); ctx.lineTo(cx, cy + size);
      ctx.stroke();
    }
  }

  function render() {
    const query = state.preview ?? state.query;
    paintHeat(query);

    // Left: the photo, unchanged
    const view = $("view");
    view.getContext("2d").drawImage(photo, 0, 0);
    drawCross(view, query);

    // Right: the similarity map, smoothly upscaled
    const map = $("map");
    const mctx = map.getContext("2d");
    mctx.imageSmoothingEnabled = true;
    mctx.drawImage(heat, 0, 0, map.width, map.height);
    drawCross(map, query);
  }

  function patchAt(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = (event.clientY - rect.top) / rect.height;
    if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return null;
    return Math.floor(fy * GR) * GC + Math.floor(fx * GC);
  }

  function bindPointer(canvas) {
    canvas.addEventListener("pointermove", (e) => {
      const p = patchAt(e);
      if (e.pointerType === "mouse") {
        if (p !== state.preview) { state.preview = p; render(); }
      } else if (e.buttons && p !== null && p !== state.query) {
        state.query = p;
        render();
      }
    });
    canvas.addEventListener("pointerleave", () => { if (state.preview !== null) { state.preview = null; render(); } });
    canvas.addEventListener("pointerdown", (e) => {
      const p = patchAt(e);
      if (p === null) return;
      state.query = p;
      state.preview = null;
      render();
    });
  }

  function bindEvents() {
    // Mouse: hover previews, click selects. Touch and pen: drag moves the query point.
    // Works on the photo and on the map.
    for (const canvas of [$("view"), $("map")]) bindPointer(canvas);
    $("view").addEventListener("keydown", (e) => {
      const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
      if (!moves[e.key]) return;
      e.preventDefault();
      const r = Math.min(GR - 1, Math.max(0, Math.floor(state.query / GC) + moves[e.key][0]));
      const c = Math.min(GC - 1, Math.max(0, (state.query % GC) + moves[e.key][1]));
      state.query = r * GC + c;
      render();
    });
    window.addEventListener("resize", render);
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
      SCENES.forEach((scene, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = scene.meta.label;
        b.addEventListener("click", () => select(i));
        $("picker").append(b);
      });
      $("picker").hidden = SCENES.length < 2;
      $("app").hidden = false;
      bindEvents();
      select(0);
      followHeight();
    })
    .catch((error) => {
      const box = $("error");
      box.textContent = error.message;
      box.hidden = false;
      followHeight();
    });
})();
