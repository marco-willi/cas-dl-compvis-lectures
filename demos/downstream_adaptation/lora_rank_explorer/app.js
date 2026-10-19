// Interface of the LoRA rank explorer. The parameter counts are computed in lora.js.
(() => {
  "use strict";

  // Shape of W. Small on purpose: every rank up to full rank fits on the slider, and the
  // rank at which LoRA stops saving parameters (32) is easy to reach.
  const D_IN = 64;
  const D_OUT = 64;
  const DEFAULTS = { dIn: D_IN, dOut: D_OUT, rank: 8, alpha: 16, modules: 1 };
  const MIN_THICKNESS = 3; // px, so that small ranks stay visible
  const SVG_NS = "http://www.w3.org/2000/svg";

  const state = { ...DEFAULTS };
  const $ = (id) => document.getElementById(id);
  const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
  const int = (value) => value.toLocaleString("en-US");

  function percent(ratio) {
    const value = ratio * 100;
    const digits = value >= 100 ? 1 : value >= 10 ? 2 : 4;
    return value.toLocaleString("en-US", { maximumFractionDigits: digits }) + "%";
  }

  function svg(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // ---------------------------------------------------------------------------
  // Controls
  // ---------------------------------------------------------------------------
  function readNumber(input, fallback, lo, hi, integer) {
    const value = integer ? Math.round(Number(input.value)) : Number(input.value);
    return input.value.trim() !== "" && Number.isFinite(value) ? clamp(value, lo, hi) : fallback;
  }

  function bindEvents() {
    $("rank").addEventListener("input", (event) => {
      state.rank = Number(event.target.value);
      update();
    });
    // Number fields: recompute while typing, write the corrected value back when leaving the field.
    const fields = [
      ["rank-number", "rank", () => [1, Lora.maxRank(state.dIn, state.dOut)], true],
      ["alpha", "alpha", () => [0, 1e6], false],
      ["modules", "modules", () => [1, 999], true],
    ];
    for (const [id, key, range, integer] of fields) {
      $(id).addEventListener("input", (event) => {
        state[key] = readNumber(event.target, state[key], ...range(), integer);
        update(event.target);
      });
      $(id).addEventListener("change", () => update());
    }
    $("reset").addEventListener("click", () => {
      Object.assign(state, DEFAULTS);
      update();
    });
    window.addEventListener("resize", drawDiagram);
  }

  // ---------------------------------------------------------------------------
  // Diagram: W + B · A, drawn to scale relative to the larger dimension of W
  // ---------------------------------------------------------------------------
  function drawDiagram() {
    const { dIn, dOut, rank } = state;
    const root = $("diagram");
    const width = Math.max(260, root.parentElement.clientWidth - 16);
    const font = clamp(width / 42, 11, 15);
    const gap = font * 2.2; // room for the operators between the matrices
    const side = clamp((width - 2 * gap - 8) / 2.35, 70, 190); // length of the larger dimension
    const scale = side / Math.max(dIn, dOut);
    const thick = Math.max(MIN_THICKNESS, rank * scale);
    const w = { width: dIn * scale, height: dOut * scale };
    const b = { width: thick, height: dOut * scale };
    const a = { width: dIn * scale, height: thick };

    const top = font * 1.8;
    const total = w.width + gap + b.width + gap + a.width;
    let x = Math.max(4, (width - total) / 2);
    const height = top + Math.max(w.height, a.height) + font * 3.2;
    root.replaceChildren();
    root.setAttribute("viewBox", `0 0 ${width} ${height}`);
    root.setAttribute("height", height);

    const defs = svg("defs");
    const pattern = svg("pattern", { id: "hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    pattern.append(svg("rect", { width: 6, height: 6, fill: "#ffedd5" }), svg("rect", { width: 2, height: 6, fill: "#fdba74" }));
    defs.append(pattern);
    root.append(defs);

    function matrix(box, name, dims, status, trainable) {
      const center = x + box.width / 2;
      const kind = trainable ? "train" : "frozen";
      root.append(
        svg("text", { x: center, y: top - font * 0.5, "text-anchor": "middle", "font-size": font * 1.1, class: "name-" + kind }, name),
        svg("rect", { x, y: top, width: box.width, height: box.height, "stroke-width": 1.5, class: kind, ...(trainable ? { fill: "url(#hatch)" } : {}) }),
        svg("text", { x: center, y: top + box.height + font * 1.3, "text-anchor": "middle", "font-size": font * 0.9, class: "dim" }, dims),
        svg("text", { x: center, y: top + box.height + font * 2.5, "text-anchor": "middle", "font-size": font * 0.9, class: "name-" + kind }, status)
      );
      x += box.width;
    }
    function operator(symbol) {
      root.append(svg("text", { x: x + gap / 2, y: top + font * 1.2, "text-anchor": "middle", "font-size": font * 1.4 }, symbol));
      x += gap;
    }

    matrix(w, "W", `${dOut} × ${dIn}`, "frozen", false);
    operator("+");
    matrix(b, "B", `${dOut} × ${rank}`, "trainable", true);
    operator("·");
    matrix(a, "A", `${rank} × ${dIn}`, "trainable", true);

    $("diagram-desc").textContent =
      `Frozen pretrained matrix W with ${dOut} rows and ${dIn} columns, plus the product of the trainable ` +
      `factors B with ${dOut} rows and ${rank} columns and A with ${rank} rows and ${dIn} columns.`;
  }

  // ---------------------------------------------------------------------------
  // Parameter counts
  // ---------------------------------------------------------------------------
  function drawBars(result) {
    const largest = Math.max(result.full, result.lora);
    const row = (label, value, kind) => {
      const wrapper = document.createElement("div");
      wrapper.className = "bar-row";
      const name = document.createElement("span");
      name.textContent = label;
      const track = document.createElement("div");
      track.className = "bar-track";
      const bar = document.createElement("div");
      bar.className = "bar " + kind;
      bar.style.width = (value / largest) * 100 + "%";
      track.append(bar);
      wrapper.append(name, track);
      return wrapper;
    };
    $("bars").replaceChildren(row("Full matrix (frozen)", result.full, "frozen"), row("LoRA factors (trainable)", result.lora, "train"));
    $("bars").setAttribute("aria-label", `Bar chart: full matrix ${int(result.full)} parameters, LoRA factors ${int(result.lora)} parameters.`);
  }

  function describe(result) {
    const { dIn, dOut, rank } = state;
    const breakEven = Lora.breakEvenRank(dIn, dOut);
    const limit = `For this shape LoRA has as many parameters as the full matrix at r = ${breakEven.toLocaleString("en-US", { maximumFractionDigits: 1 })}.`;
    const note = $("note");
    note.classList.toggle("warn", result.difference <= 0);
    if (result.difference > 0) {
      note.textContent = `LoRA trains ${percent(result.ratio)} of the parameters of a full update. ${limit} LoRA does not necessarily use fewer parameters: above this rank the ratio exceeds 100%.`;
    } else if (result.difference === 0) {
      note.textContent = `Break-even: at rank ${rank} LoRA has exactly as many parameters as the full matrix. LoRA does not necessarily use fewer parameters.`;
    } else {
      note.textContent = `At rank ${rank} LoRA has more parameters than the full matrix (${percent(result.ratio)}). ${limit} A low-rank update only saves parameters when r is small compared with the dimensions of W.`;
    }
  }

  // `source` is the field the user is typing in; it is not overwritten while it has the focus.
  function update(source) {
    const max = Lora.maxRank(state.dIn, state.dOut);
    state.rank = clamp(state.rank, 1, max);

    const slider = $("rank");
    slider.max = max;
    slider.value = state.rank;
    slider.setAttribute("aria-valuetext", `rank ${state.rank} of at most ${max}`);
    $("rank-number").max = max;
    $("rank-max").textContent = `of max. ${max}`;
    for (const [id, key] of [["rank-number", "rank"], ["alpha", "alpha"], ["modules", "modules"]])
      if ($(id) !== source) $(id).value = state[key];
    $("scale").textContent = `scale α / r = ${(state.alpha / state.rank).toLocaleString("en-US", { maximumFractionDigits: 4 })}`;

    const result = Lora.counts(state.dIn, state.dOut, state.rank, state.modules);
    const times = state.modules > 1 ? ` (${state.modules} matrices)` : "";
    $("stat-full").textContent = int(result.full) + times;
    $("stat-lora").textContent = int(result.lora) + times;
    $("stat-ratio").textContent = percent(result.ratio);
    const diff = $("stat-diff");
    diff.textContent = (result.difference < 0 ? "−" : "") + int(Math.abs(result.difference));
    const small = document.createElement("small");
    small.textContent = result.difference > 0 ? "fewer with LoRA" : result.difference < 0 ? "more with LoRA" : "equal";
    diff.append(small);

    drawDiagram();
    drawBars(result);
    describe(result);
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
})();
