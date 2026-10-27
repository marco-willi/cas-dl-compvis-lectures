// Interface of the receptive-field demo. Defaults: config.js, computation: rf.js.
// Notation as in the lecture: input size i, kernel size k_l, stride s_l and padding p_l of
// layer l, output size o_l, receptive field r_0 of the selected activation.
(() => {
  "use strict";

  const CFG = globalThis.RFDemoConfig;
  const RF = globalThis.RF;
  const $ = (id) => document.getElementById(id);

  const ACCENT = [232, 145, 12];
  const INK = "#1f2328";
  const MAX_MAP = 260; // px: largest side of one activation map

  // Outline of an elephant drawn on the input, only as an illustration of image content.
  // Coordinates are fractions of the map side.
  const ELEPHANT = (() => {
    const path = new Path2D();
    // Silhouette: back, head, trunk, legs, rump
    path.moveTo(0.8, 0.38);
    path.bezierCurveTo(0.78, 0.22, 0.55, 0.18, 0.42, 0.24);
    path.bezierCurveTo(0.36, 0.16, 0.22, 0.15, 0.17, 0.26);
    path.bezierCurveTo(0.13, 0.34, 0.13, 0.44, 0.15, 0.52);
    path.bezierCurveTo(0.14, 0.62, 0.12, 0.72, 0.08, 0.78);
    path.bezierCurveTo(0.06, 0.81, 0.09, 0.84, 0.12, 0.81);
    path.bezierCurveTo(0.18, 0.74, 0.21, 0.62, 0.23, 0.55);
    path.lineTo(0.28, 0.58);
    for (const [left, right] of [[0.28, 0.35], [0.38, 0.45]]) {
      path.lineTo(left, 0.7);
      path.lineTo(left, 0.86);
      path.lineTo(right, 0.86);
      path.lineTo(right, 0.7);
    }
    path.bezierCurveTo(0.5, 0.72, 0.56, 0.72, 0.6, 0.7);
    for (const [left, right] of [[0.6, 0.67], [0.7, 0.77]]) {
      path.lineTo(left, 0.7);
      path.lineTo(left, 0.86);
      path.lineTo(right, 0.86);
      path.lineTo(right, 0.7);
    }
    path.bezierCurveTo(0.83, 0.58, 0.83, 0.46, 0.8, 0.38);
    path.closePath();
    // Tail
    path.moveTo(0.815, 0.43);
    path.quadraticCurveTo(0.87, 0.5, 0.86, 0.62);
    // Ear
    path.moveTo(0.3, 0.27);
    path.bezierCurveTo(0.42, 0.25, 0.45, 0.42, 0.38, 0.52);
    path.bezierCurveTo(0.34, 0.57, 0.29, 0.51, 0.29, 0.44);
    // Eye
    path.moveTo(0.225, 0.33);
    path.arc(0.21, 0.33, 0.015, 0, 2 * Math.PI);
    return path;
  })();

  let state; // set by reset()
  let layout = null; // positions of the maps on the canvas, set by draw()

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
  const layers = () => state.layers.slice(0, state.layerCount);
  const sizes = () => RF.sizes(CFG.INPUT_SIZE, layers());

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Typesets TeX with KaTeX (vendored in ../vendor/katex); shows the source if it is missing.
  function tex(source, display = false) {
    const node = el("span", "tex");
    if (globalThis.katex) globalThis.katex.render(source, node, { throwOnError: false, displayMode: display });
    else node.textContent = source;
    return node;
  }

  function segmented(values, isPressed, onClick) {
    const group = el("div", "seg");
    group.setAttribute("role", "group");
    for (const value of values) {
      const button = el("button", "", String(value));
      button.type = "button";
      button.setAttribute("aria-pressed", String(isPressed(value)));
      button.addEventListener("click", () => onClick(value));
      group.append(button);
    }
    return group;
  }

  // ---------------------------------------------------------------------------
  // State changes
  // ---------------------------------------------------------------------------
  function reset() {
    const d = CFG.DEFAULTS;
    state = {
      layerCount: d.layerCount,
      layers: d.layers.map((layer) => ({ ...layer })),
      paths: d.paths,
      pinned: null, // selected activation {r, c} in the last map; null = centre
      hover: null,
    };
    render();
  }

  function update(change) {
    change();
    state.hover = null;
    render();
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------
  function render() {
    // Controls
    $("ctl-layerCount").replaceChildren(
      ...segmented(CFG.LAYER_COUNTS, (n) => n === state.layerCount, (n) => update(() => (state.layerCount = n))).children
    );
    $("paths").checked = state.paths;
    $("layers").replaceChildren(
      ...layers().map((layer, index) => {
        const box = el("div", "layer");
        const label = el("span", "", "kernel");
        const strideLabel = el("span", "", "stride");
        box.append(
          el("b", "", `Layer ${index + 1}`),
          label,
          segmented(CFG.KERNEL_SIZES, (k) => k === layer.k, (k) => update(() => (layer.k = k))),
          strideLabel,
          segmented(CFG.STRIDES, (s) => s === layer.s, (s) => update(() => (layer.s = s)))
        );
        return box;
      })
    );

    const last = sizes()[state.layerCount];
    const centre = Math.floor((last - 1) / 2);
    state.pinned = state.pinned
      ? { r: clamp(state.pinned.r, 0, last - 1), c: clamp(state.pinned.c, 0, last - 1) }
      : { r: centre, c: centre };

    renderFormula();
    renderTable();
    renderSelection();
    draw();
  }

  const selected = () => state.hover || state.pinned;

  function renderFormula() {
    const net = layers();
    const terms = net.map(({ k }, index) => {
      const strides = net.slice(0, index).map((layer) => layer.s);
      const product = strides.length > 1 ? `(${strides.join(" \\cdot ")})` : strides.length ? `${strides[0]}` : "1";
      return `(${k} - 1) \\cdot ${product}`;
    });
    const r = RF.receptiveField(net);
    $("formula").replaceChildren(
      tex(
        String.raw`r_0 = \sum_{l=1}^{L} \left( (k_l - 1) \prod_{i=1}^{l-1} s_i \right) + 1 = ` +
          `${terms.join(" + ")} + 1 = ${r}`,
        true
      )
    );
  }

  function renderTable() {
    const net = layers();
    const size = sizes();
    const head = el("tr");
    for (const label of [
      "Map",
      String.raw`k_l`,
      String.raw`s_l`,
      String.raw`p_l = \lfloor k_l / 2 \rfloor`,
      String.raw`\text{size } o_l = \left\lfloor \frac{o_{l-1} + 2p_l - k_l}{s_l} \right\rfloor + 1`,
      String.raw`\text{receptive field}`,
    ]) {
      const th = el("th");
      th.append(label === "Map" ? label : tex(label));
      head.append(th);
    }
    const rows = [head];
    const input = el("tr");
    for (const text of ["Input", "–", "–", "–", `${CFG.INPUT_SIZE} × ${CFG.INPUT_SIZE}`, "1 × 1"]) input.append(el("td", "", text));
    rows.push(input);
    net.forEach(({ k, s }, index) => {
      const l = index + 1;
      const r = RF.receptiveField(net, l);
      const row = el("tr", l === net.length ? "last" : "");
      for (const text of [`Layer ${l}`, k, s, RF.padding(k), `${size[l]} × ${size[l]}`, `${r} × ${r}`]) {
        row.append(el("td", "", String(text)));
      }
      rows.push(row);
    });
    $("table").replaceChildren(...rows);
  }

  function renderSelection() {
    const { r, c } = selected();
    const i = CFG.INPUT_SIZE;
    const [rowLo, rowHi] = RF.ranges(layers(), r)[0];
    const [colLo, colHi] = RF.ranges(layers(), c)[0];
    const rf = rowHi - rowLo + 1;
    const rows = Math.min(rowHi, i - 1) - Math.max(rowLo, 0) + 1;
    const cols = Math.min(colHi, i - 1) - Math.max(colLo, 0) + 1;
    const node = $("selection");
    node.replaceChildren(
      el("b", "", `Selected activation (r, c) = (${r}, ${c}) in layer ${state.layerCount}`),
      ` depends on input rows ${rowLo} to ${rowHi} and columns ${colLo} to ${colHi} (${rf} × ${rf}).`
    );
    if (rows < rf || cols < rf) {
      node.append(
        " ",
        el(
          "span",
          "warn",
          `Only ${rows} × ${cols} of these lie inside the input (indices 0 to ${i - 1}); the rest is zero-padding.`
        )
      );
    }
    if (rf >= i) node.append(" ", el("span", "note", "The receptive field is at least as large as the input."));
  }

  // ---------------------------------------------------------------------------
  // Canvas: all maps are drawn with the same side length, so that the receptive fields of
  // different layers line up spatially (a map with fewer activations has larger cells).
  // ---------------------------------------------------------------------------
  function computeLayout(width) {
    const count = state.layerCount + 1;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const labelHeight = 2.6 * rem; // two lines of labels above each map
    const stacked = width < 520;
    let side, gap;
    if (stacked) {
      gap = 1.2 * rem;
      side = Math.min(width, 0.8 * MAX_MAP);
    } else {
      gap = clamp(width * 0.035, 14, 48);
      side = Math.min(MAX_MAP, (width - (count - 1) * gap) / count);
    }
    const maps = [];
    for (let l = 0; l < count; l++) {
      const x = stacked ? (width - side) / 2 : (width - (count * side + (count - 1) * gap)) / 2 + l * (side + gap);
      const y = stacked ? l * (side + gap + labelHeight) + labelHeight : labelHeight;
      maps.push({ x, y, side });
    }
    const height = stacked ? count * (side + gap + labelHeight) - gap : labelHeight + side + 4;
    return { maps, width, height, stacked, rem };
  }

  function draw() {
    const canvas = $("maps");
    const width = canvas.parentElement.clientWidth;
    layout = computeLayout(width);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(layout.height * ratio);
    canvas.style.height = layout.height + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, layout.height);

    const net = layers();
    const size = sizes();
    const { r, c } = selected();
    const rowRanges = RF.ranges(net, r);
    const colRanges = RF.ranges(net, c);
    const rowPaths = state.paths ? RF.paths(CFG.INPUT_SIZE, net, r) : null;
    const colPaths = state.paths ? RF.paths(CFG.INPUT_SIZE, net, c) : null;
    const rem = layout.rem;
    const boxes = []; // clipped highlighted rectangle of every map in canvas coordinates

    layout.maps.forEach(({ x, y, side }, l) => {
      const n = size[l];
      const cell = side / n;
      const isLast = l === net.length;

      // Labels
      ctx.fillStyle = INK;
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.font = `600 ${0.8 * rem}px system-ui, sans-serif`;
      ctx.fillText(l === 0 ? "Input" : `Layer ${l}`, x + side / 2, y - 1.45 * rem);
      ctx.font = `${0.72 * rem}px system-ui, sans-serif`;
      ctx.fillStyle = "#667085";
      const detail = l === 0 ? `${n} × ${n}` : `${n} × ${n}, k=${net[l - 1].k}, s=${net[l - 1].s}`;
      ctx.fillText(detail, x + side / 2, y - 0.45 * rem);

      // Background
      ctx.fillStyle = "#eef0f3";
      ctx.fillRect(x, y, side, side);

      // Activations the selected one depends on (clipped to the map)
      const [r0, r1] = rowRanges[l];
      const [c0, c1] = colRanges[l];
      const rLo = Math.max(r0, 0), rHi = Math.min(r1, n - 1);
      const cLo = Math.max(c0, 0), cHi = Math.min(c1, n - 1);
      if (state.paths && !isLast) {
        let max = 0;
        for (let a = rLo; a <= rHi; a++) for (let b = cLo; b <= cHi; b++) max = Math.max(max, rowPaths[l][a] * colPaths[l][b]);
        for (let a = rLo; a <= rHi; a++) {
          for (let b = cLo; b <= cHi; b++) {
            const t = (rowPaths[l][a] * colPaths[l][b]) / max;
            ctx.fillStyle = `rgba(${ACCENT}, ${0.12 + 0.88 * t})`;
            ctx.fillRect(x + b * cell, y + a * cell, cell, cell);
          }
        }
      } else {
        ctx.fillStyle = `rgba(${ACCENT}, ${isLast ? 0.25 : 0.85})`;
        ctx.fillRect(x + cLo * cell, y + rLo * cell, (cHi - cLo + 1) * cell, (rHi - rLo + 1) * cell);
      }
      if (isLast) {
        ctx.fillStyle = INK;
        ctx.fillRect(x + c * cell, y + r * cell, cell, cell);
      }

      // Grid lines (only when the cells are large enough)
      if (cell >= 4) {
        ctx.strokeStyle = cell >= 8 ? "rgba(255, 255, 255, 0.9)" : "rgba(255, 255, 255, 0.6)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let t = 1; t < n; t++) {
          const offset = Math.round(t * cell) + 0.5;
          ctx.moveTo(x + offset, y);
          ctx.lineTo(x + offset, y + side);
          ctx.moveTo(x, y + offset);
          ctx.lineTo(x + side, y + offset);
        }
        ctx.stroke();
      }
      if (l === 0) {
        ctx.save();
        ctx.translate(x, y);
        ctx.scale(side, side);
        ctx.strokeStyle = "rgba(70, 78, 92, 0.6)";
        ctx.lineWidth = 2.5 / side;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke(ELEPHANT);
        ctx.restore();
      }
      ctx.strokeStyle = "#9aa3b2";
      ctx.strokeRect(x + 0.5, y + 0.5, side - 1, side - 1);

      // Receptive field reaching into the zero-padding: dashed frame of the full range
      if (r0 < 0 || c0 < 0 || r1 > n - 1 || c1 > n - 1) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x - 6, y - 6, side + 12, side + 12);
        ctx.clip();
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = `rgb(${ACCENT})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x + c0 * cell, y + r0 * cell, (c1 - c0 + 1) * cell, (r1 - r0 + 1) * cell);
        ctx.restore();
      }

      boxes.push({
        left: x + cLo * cell,
        top: y + rLo * cell,
        right: x + (cHi + 1) * cell,
        bottom: y + (rHi + 1) * cell,
      });
    });

    // Lines connecting the region of each map with the region it feeds in the next map
    ctx.strokeStyle = `rgba(${ACCENT}, 0.7)`;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let l = 0; l < boxes.length - 1; l++) {
      const a = boxes[l];
      const b = l + 1 === net.length ? cellBox(l + 1, r, c) : boxes[l + 1];
      if (layout.stacked) {
        ctx.moveTo(a.left, a.bottom); ctx.lineTo(b.left, b.top);
        ctx.moveTo(a.right, a.bottom); ctx.lineTo(b.right, b.top);
      } else {
        ctx.moveTo(a.right, a.top); ctx.lineTo(b.left, b.top);
        ctx.moveTo(a.right, a.bottom); ctx.lineTo(b.left, b.bottom);
      }
    }
    ctx.stroke();
  }

  // Canvas rectangle of cell (r, c) in map l.
  function cellBox(l, r, c) {
    const { x, y, side } = layout.maps[l];
    const cell = side / sizes()[l];
    return { left: x + c * cell, top: y + r * cell, right: x + (c + 1) * cell, bottom: y + (r + 1) * cell };
  }

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
  // Events
  // ---------------------------------------------------------------------------
  // Activation of the last map under the pointer, or null.
  function cellAt(event) {
    if (!layout) return null;
    const rect = $("maps").getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const { x, y, side } = layout.maps[state.layerCount];
    if (px < x || py < y || px >= x + side || py >= y + side) return null;
    const n = sizes()[state.layerCount];
    return { r: Math.floor(((py - y) / side) * n), c: Math.floor(((px - x) / side) * n) };
  }
  const same = (a, b) => (a && b ? a.r === b.r && a.c === b.c : a === b);

  function refresh() {
    renderSelection();
    draw();
  }

  function bindEvents() {
    const canvas = $("maps");
    // Mouse: hovering previews an activation, clicking selects it. Touch and pen: tap or drag.
    canvas.addEventListener("pointerdown", (event) => {
      const cell = cellAt(event);
      if (!cell) return;
      if (event.pointerType !== "mouse") canvas.setPointerCapture(event.pointerId);
      state.pinned = cell;
      state.hover = null;
      refresh();
    });
    canvas.addEventListener("pointermove", (event) => {
      const cell = cellAt(event);
      canvas.style.cursor = cell ? "pointer" : "default";
      if (event.pointerType === "mouse") {
        if (same(cell, state.hover)) return;
        state.hover = cell;
      } else {
        if (!cell || same(cell, state.pinned)) return;
        state.pinned = cell;
      }
      refresh();
    });
    canvas.addEventListener("pointerleave", () => {
      if (!state.hover) return;
      state.hover = null;
      refresh();
    });
    canvas.addEventListener("keydown", (event) => {
      const step = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[event.key];
      if (!step) return;
      event.preventDefault();
      const n = sizes()[state.layerCount];
      state.pinned = { r: clamp(state.pinned.r + step[0], 0, n - 1), c: clamp(state.pinned.c + step[1], 0, n - 1) };
      state.hover = null;
      refresh();
    });
    $("paths").addEventListener("change", (event) => update(() => (state.paths = event.target.checked)));
    $("reset").addEventListener("click", reset);
    window.addEventListener("resize", draw);
  }

  bindEvents();
  reset();
})();
