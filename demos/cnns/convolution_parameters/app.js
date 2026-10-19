// Interface of the convolution-parameter demo. Presets: config.js, computation: conv.js,
// exercises: exercises.js. Notation as in the lecture: input size i, output size o, kernel size k,
// padding p, stride s, dilation d; output cell (r, c), kernel entry (u, v).
(() => {
  "use strict";

  const CFG = globalThis.ConvDemoConfig;
  const Conv = globalThis.Conv;
  const Ex = globalThis.ConvExercises;
  const $ = (id) => document.getElementById(id);

  const N = CFG.INPUT_PRESETS[0].values.length; // input size
  const MAX_CELLS = N + 2 * Math.max(...CFG.PADDINGS); // largest grid that can occur
  const MIN_CELL = 11; // px
  const VALUES_FROM = 22; // px: cells of at least this size show their value
  const PARAMS = [
    ["kernelSize", CFG.KERNEL_SIZES],
    ["padding", CFG.PADDINGS],
    ["stride", CFG.STRIDES],
    ["dilation", CFG.DILATIONS],
  ];

  let state; // set by reset()
  let result; // output of Conv.correlate for the current state
  let inputCells = [];
  let outputCells = [];
  let marked = [];
  let taskItems = [];

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  const byId = (list, id) => list.find((entry) => entry.id === id);
  const input = () => byId(CFG.INPUT_PRESETS, state.input).values;
  const params = () => ({ padding: state.padding, stride: state.stride, dilation: state.dilation, relu: state.relu });

  function presetWeights(id, k) {
    const weights = byId(CFG.FILTER_PRESETS, id).weights(k);
    return weights ? weights.map((row) => row.slice()) : null;
  }

  // Rounded number without trailing zeros, e.g. 0.1111 -> "0.111".
  function num(value, digits = 3) {
    const rounded = Number(value.toFixed(digits));
    return String(rounded === 0 ? 0 : rounded);
  }
  const show = (value, digits) => num(value, digits).replace("-", "−");
  const short = (value) => show(value, 2).replace(/^(−?)0\./, "$1.");

  const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
  const NEG = [33, 102, 172], ZERO = [247, 247, 247], POS = [178, 24, 43];

  // Diverging colour scale centred on zero; values beyond +-range are clipped.
  function diverging(value, range) {
    const t = clamp(value / range, -1, 1);
    const end = t < 0 ? NEG : POS;
    const rgb = ZERO.map((z, c) => Math.round(z + (end[c] - z) * Math.abs(t)));
    return { background: `rgb(${rgb})`, color: Math.abs(t) > 0.55 ? "#fff" : "#1f2328" };
  }
  function gray(value) {
    const g = Math.round(255 * clamp(value, 0, 1));
    return { background: `rgb(${g},${g},${g})`, color: g > 140 ? "#000" : "#fff" };
  }
  function paint(el, { background, color }) {
    el.style.background = background;
    el.style.color = color;
  }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Typesets TeX with KaTeX (vendored in ../vendor/katex); shows the source if it is missing.
  function tex(source) {
    const node = el("span", "tex");
    if (globalThis.katex) globalThis.katex.render(source, node, { throwOnError: false });
    else node.textContent = source;
    return node;
  }

  // ---------------------------------------------------------------------------
  // State changes
  // ---------------------------------------------------------------------------
  function reset() {
    const d = CFG.DEFAULTS;
    state = {
      input: d.input,
      filter: d.filter,
      kernelSize: d.kernelSize,
      padding: d.padding,
      stride: d.stride,
      dilation: d.dilation,
      relu: d.relu,
      range: d.colorRange,
      weights: presetWeights(d.filter, d.kernelSize),
      edited: false,
      pinned: null, // selected output cell {i, j}; null = centre
      hover: null,
      notice: "",
    };
    renderKernel();
    render();
  }

  // Padding, stride and dilation leave the weights untouched; a new kernel size needs a
  // new set of weights, which is taken from the selected preset.
  function setParam(name, value) {
    if (state[name] === value) return;
    state.notice = "";
    if (name === "kernelSize") {
      const previous = state.kernelSize;
      const wanted = byId(CFG.FILTER_PRESETS, state.filter);
      let weights = presetWeights(state.filter, value);
      let note = "";
      if (!weights) {
        state.filter = CFG.FALLBACK_FILTER;
        weights = presetWeights(state.filter, value);
        note = ` “${wanted.name}” is not defined for ${value} × ${value}.`;
      }
      state.notice =
        `Kernel size ${previous} → ${value}: weights regenerated from the preset ` +
        `“${byId(CFG.FILTER_PRESETS, state.filter).name}”.${note}${state.edited ? " Your edits were discarded." : ""}`;
      state.kernelSize = value;
      state.weights = weights;
      state.edited = false;
      renderKernel();
    } else {
      state[name] = value;
    }
    state.hover = null;
    render();
  }

  function setFilter(id) {
    state.filter = id;
    state.weights = presetWeights(id, state.kernelSize);
    state.edited = false;
    state.notice = "";
    renderKernel();
    render();
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------
  function buildControls() {
    for (const [name, values] of PARAMS) {
      const group = $("ctl-" + name);
      for (const value of values) {
        const button = el("button", "", String(value));
        button.type = "button";
        button.dataset.value = value;
        button.addEventListener("click", () => setParam(name, value));
        group.append(button);
      }
    }
    for (const preset of CFG.INPUT_PRESETS) $("input-preset").append(new Option(preset.name, preset.id));
    for (const preset of CFG.FILTER_PRESETS) $("filter-preset").append(new Option(preset.name, preset.id));
  }

  // Rebuilds the weight inputs; only called when the weights are replaced as a whole.
  function renderKernel() {
    const k = state.kernelSize;
    const box = $("weights");
    box.replaceChildren();
    box.dataset.k = k;
    box.style.setProperty("--k", k);
    state.weights.forEach((row, u) =>
      row.forEach((weight, v) => {
        const field = el("input");
        field.type = "number";
        field.step = "any";
        field.value = num(weight, k >= 5 ? 2 : 3);
        field.setAttribute("aria-label", `Weight row ${u}, column ${v}`);
        field.addEventListener("input", () => {
          const value = field.valueAsNumber;
          field.classList.toggle("invalid", !Number.isFinite(value));
          if (!Number.isFinite(value)) return; // e.g. while typing "-"
          state.weights[u][v] = value;
          state.edited = true;
          state.notice = "";
          render();
        });
        field.addEventListener("blur", () => {
          if (!Number.isFinite(field.valueAsNumber)) field.value = num(state.weights[u][v]);
          field.classList.remove("invalid");
        });
        box.append(field);
      })
    );
  }

  function render() {
    const { kernelSize: k, padding: p, stride: s, dilation: d } = state;
    result = Conv.correlate(input(), state.weights, params());

    // Controls
    for (const [name] of PARAMS) {
      for (const button of $("ctl-" + name).children) {
        button.setAttribute("aria-pressed", String(Number(button.dataset.value) === state[name]));
      }
    }
    $("relu").checked = state.relu;
    $("input-preset").value = state.input;
    $("filter-preset").value = state.filter;
    for (const option of $("filter-preset").options) option.disabled = !byId(CFG.FILTER_PRESETS, option.value).weights(k);

    // Kernel: tint the weights relative to the largest absolute weight
    const maxWeight = Math.max(...state.weights.flat().map(Math.abs)) || 1;
    [...$("weights").children].forEach((field, index) =>
      paint(field, diverging(state.weights[Math.floor(index / k)][index % k], maxWeight))
    );
    $("kernel-size").replaceChildren("(", tex(`k = ${k}`), `, ${k * k} ${k === 1 ? "weight" : "weights"})`);
    $("kernel-state").textContent = state.edited ? "edited" : "preset weights";
    $("restore").disabled = !state.edited;
    const notice = $("notice");
    if (notice.textContent !== state.notice) {
      notice.textContent = state.notice;
      notice.classList.remove("flash");
      if (state.notice) {
        void notice.offsetWidth; // restart the animation
        notice.classList.add("flash");
      }
    }

    // Input with zero-padding
    const padded = N + 2 * p;
    const inputGrid = $("input-grid");
    inputGrid.replaceChildren();
    inputGrid.style.setProperty("--n", padded);
    inputCells = [];
    for (let r = -p; r < N + p; r++) {
      for (let c = -p; c < N + p; c++) {
        const inside = r >= 0 && r < N && c >= 0 && c < N;
        const cell = el("div", inside ? "cell" : "cell pad", inside ? short(input()[r][c]) : "0");
        if (inside) paint(cell, gray(input()[r][c]));
        else cell.style.color = "#fff";
        inputCells.push(cell);
      }
    }
    inputGrid.append(...inputCells, el("div", "extent"));
    $("input-size").replaceChildren("(", tex(`i = ${N}`), p ? `, padded to ${padded} × ${padded})` : ", no padding)");

    // Output
    const outputGrid = $("output-grid");
    outputGrid.replaceChildren();
    outputGrid.hidden = !result.valid;
    $("message").hidden = result.valid;
    outputCells = [];
    if (result.valid) {
      outputGrid.style.setProperty("--n", result.size);
      result.values.forEach((row, i) =>
        row.forEach((value, j) => {
          const cell = el("div", "cell", short(value));
          cell.dataset.i = i;
          cell.dataset.j = j;
          paint(cell, diverging(value, state.range));
          outputCells.push(cell);
        })
      );
      outputGrid.append(...outputCells);
      $("output-size").replaceChildren("(", tex(`o = ${result.size}`), ")");
    } else {
      const eff = Conv.effectiveKernelSize(k, d);
      $("message").replaceChildren(
        el("b", "", "No output for this configuration."),
        `The effective kernel size (${eff}) is larger than the padded input (i + 2p = ${N} + 2·${p} = ${padded}), ` +
          "so the kernel does not fit anywhere. Increase the padding or reduce kernel size or dilation."
      );
      $("output-size").textContent = "(none)";
    }

    // Colour legend
    const maxAbs = Math.max(0, ...result.values.flat().map(Math.abs));
    $("legend-min").textContent = show(-state.range, 2);
    $("legend-max").textContent = "+" + show(state.range, 2);
    $("clip").textContent =
      maxAbs > state.range * (1 + 1e-9) ? `Largest |value| is ${show(maxAbs, 2)}: colours are clipped.` : "";
    $("rescale").disabled = !result.valid || maxAbs === 0 || Math.abs(maxAbs - state.range) < 1e-9;

    renderFacts();
    renderTasks();
    highlight();
    fit();
  }

  function renderFacts() {
    const { kernelSize: k, padding: p, stride: s, dilation: d } = state;
    const numerator = N + 2 * p - d * (k - 1) - 1;
    const eff = Conv.effectiveKernelSize(k, d);
    const rows = [
      [
        "Output size",
        String.raw`\displaystyle o = \left\lfloor \frac{i + 2p - d\cdot(k-1) - 1}{s} + 1 \right\rfloor = ` +
          String.raw`\left\lfloor \frac{${N} + 2\cdot ${p} - ${d}\cdot(${k}-1) - 1}{${s}} + 1 \right\rfloor = ` +
          String.raw`\left\lfloor \frac{${num(numerator)}}{${s}} + 1 \right\rfloor = ${num(result.size)}`,
        result.valid ? `→ output ${result.size} × ${result.size}` : "≤ 0 → no output",
      ],
      [
        "Effective kernel size",
        String.raw`k_\text{eff} = d\cdot(k-1) + 1 = ${d}\cdot(${k}-1) + 1 = ${eff}`,
        d > 1 && k > 1 ? `→ covers ${eff} × ${eff} input cells, reads ${k * k} of them` : "",
      ],
      ["Number of weights", `k^2 = ${k}^2 = ${k * k}`, "(independent of padding, stride and dilation)"],
    ];
    $("facts").replaceChildren(
      ...rows.map(([term, formula, note], index) => {
        const row = el("div");
        const dd = el("dd", "");
        dd.append(tex(formula), " ", el(index === 0 ? "b" : "span", index === 0 && !result.valid ? "bad" : "", note));
        row.append(el("dt", "", term), dd);
        return row;
      })
    );
  }

  // Ticks the exercises that the current state solves.
  function buildTasks() {
    taskItems = Ex.TASKS.map((task) => {
      const item = el("li");
      const hint = el("details");
      hint.append(el("summary", "", "Hint"), task.hint);
      const body = el("div");
      body.append(el("b", "", task.title), " – " + task.text);
      item.append(el("span", "status"), body, hint);
      return item;
    });
    $("task-list").replaceChildren(...taskItems);
  }

  function renderTasks() {
    const ctx = { ...state, input: input(), result };
    let solved = 0;
    Ex.TASKS.forEach((task, index) => {
      const done = task.check(ctx);
      solved += done;
      taskItems[index].classList.toggle("done", done);
      taskItems[index].querySelector(".status").textContent = done ? "✓" : "";
    });
    $("tasks-progress").textContent = `(${solved} of ${Ex.TASKS.length} solved by the current settings)`;
  }

  // Marks the selected output cell and the input cells it samples, and shows the calculation.
  function highlight() {
    for (const cell of marked) cell.classList.remove("sampled", "selected");
    marked = [];
    const extent = $("input-grid").querySelector(".extent");
    const calc = $("calc");
    if (!result.valid) {
      extent.hidden = true;
      calc.replaceChildren(el("span", "where", "No output cell to inspect."));
      return;
    }
    const { kernelSize: k, padding: p, stride: s, dilation: d } = state;
    const centre = Math.floor((result.size - 1) / 2);
    state.pinned = state.pinned
      ? { i: clamp(state.pinned.i, 0, result.size - 1), j: clamp(state.pinned.j, 0, result.size - 1) }
      : { i: centre, j: centre };
    const { i, j } = state.hover || state.pinned;
    const terms = Conv.terms(input(), state.weights, params(), i, j);

    const padded = N + 2 * p;
    for (const term of terms) marked.push(inputCells[(term.row + p) * padded + term.col + p]);
    for (const cell of marked) cell.classList.add("sampled");
    const selected = outputCells[i * result.size + j];
    selected.classList.add("selected");
    marked.push(selected);

    const eff = Conv.effectiveKernelSize(k, d);
    extent.hidden = false;
    extent.style.left = `calc(var(--cell) * ${j * s})`;
    extent.style.top = `calc(var(--cell) * ${i * s})`;
    extent.style.width = extent.style.height = `calc(var(--cell) * ${eff})`;

    // Calculation: sampled input values x kernel weights -> sum -> optional ReLU
    const indices = (o) => Array.from({ length: k }, (_, u) => num(Conv.sampleIndex(o, u, p, s, d))).join(", ");
    const where = el("div", "where");
    where.append(
      el("b", "", `Output cell (r, c) = (${i}, ${j})`),
      " reads input rows ",
      tex(String.raw`r\cdot s - p + u\cdot d = ${i}\cdot ${s} - ${p} + u\cdot ${d} \in \{${indices(i)}\}`),
      " and columns ",
      tex(String.raw`c\cdot s - p + v\cdot d = ${j}\cdot ${s} - ${p} + v\cdot ${d} \in \{${indices(j)}\}`),
      terms.some((t) => !t.inside) ? `. Indices outside 0…${N - 1} are zero-padding.` : "."
    );
    const mini = (caption, text, isPad) => {
      const figure = el("figure");
      const grid = el("div", "mini");
      grid.dataset.k = k;
      grid.style.setProperty("--k", k);
      grid.append(...terms.map((t) => el("span", isPad(t) ? "pad" : "", text(t))));
      figure.append(grid, el("figcaption", "", caption));
      return figure;
    };
    const sum = result.sums[i][j];
    const total = el("span", "result");
    total.append(tex(String.raw`= \sum_{u,v} x \cdot w =`), " ", el("b", "", show(sum)));
    if (state.relu) total.append(" → ReLU → ", el("b", "", show(Math.max(0, sum))));
    calc.replaceChildren(
      where,
      mini("sampled input values", (t) => show(t.x, 2), (t) => !t.inside),
      el("span", "op", "×"),
      mini("kernel weights", (t) => show(t.w, k >= 5 ? 2 : 3), () => false),
      total
    );
  }

  // ---------------------------------------------------------------------------
  // Layout: one cell size for input and output, derived from the available space
  // ---------------------------------------------------------------------------
  let autoHeight = false;
  try {
    autoHeight = Boolean(window.frameElement && window.frameElement.hasAttribute("data-auto-height"));
  } catch (error) {
    // cross-origin parent: keep the fixed height
  }
  document.documentElement.classList.toggle("auto-height", autoHeight);
  document.documentElement.style.setProperty("--max-cells", MAX_CELLS);

  function fit() {
    const root = document.documentElement;
    const rem = parseFloat(getComputedStyle(root).fontSize);
    const panels = $("panels");
    const stacked = window.matchMedia("(max-width: 620px)").matches;
    const gap = parseFloat(getComputedStyle(panels).columnGap) || 0;
    const width = stacked
      ? Math.min(panels.clientWidth, 26 * rem)
      : (panels.clientWidth - $("kernel-panel").offsetWidth - 2 * gap) / 2;
    let cell = clamp(Math.floor(width / MAX_CELLS), MIN_CELL, Math.round(2.6 * rem));
    root.style.setProperty("--cell", cell + "px");
    // A fixed-height frame (e.g. a slide) should not scroll: shrink the cells if needed.
    if (!autoHeight && !stacked) {
      const overflow = root.scrollHeight - window.innerHeight;
      if (overflow > 0) cell = Math.max(MIN_CELL, cell - Math.ceil(overflow / MAX_CELLS));
      root.style.setProperty("--cell", cell + "px");
    }
    document.body.classList.toggle("show-values", cell >= VALUES_FROM);
  }
  window.addEventListener("resize", fit);

  // Optional: an embedding page on the same site can add `data-auto-height` to the iframe;
  // the iframe then follows the height of the demo instead of using a fixed height.
  if (autoHeight) {
    const frame = window.frameElement;
    new ResizeObserver(() => {
      // Border and padding of the iframe count towards its height if the page uses border-box.
      const borderBox = window.parent.getComputedStyle(frame).boxSizing === "border-box";
      const chrome = borderBox ? frame.offsetHeight - window.innerHeight : 0;
      frame.style.height = Math.ceil(document.body.getBoundingClientRect().height) + chrome + "px";
    }).observe(document.body);
  }

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------
  function cellAt(event) {
    const target = document.elementFromPoint(event.clientX, event.clientY);
    const cell = target && target.closest("#output-grid .cell");
    return cell ? { i: Number(cell.dataset.i), j: Number(cell.dataset.j) } : null;
  }
  const same = (a, b) => (a && b ? a.i === b.i && a.j === b.j : a === b);

  function bindEvents() {
    const grid = $("output-grid");
    // Mouse: hovering previews a cell, clicking selects it. Touch and pen: tap or drag selects.
    grid.addEventListener("pointerdown", (event) => {
      const cell = cellAt(event);
      if (!cell) return;
      if (event.pointerType !== "mouse") grid.setPointerCapture(event.pointerId);
      state.pinned = cell;
      state.hover = null;
      highlight();
    });
    grid.addEventListener("pointermove", (event) => {
      const cell = cellAt(event);
      if (event.pointerType === "mouse") {
        if (same(cell, state.hover)) return;
        state.hover = cell;
      } else {
        if (!cell || same(cell, state.pinned)) return;
        state.pinned = cell;
      }
      highlight();
    });
    grid.addEventListener("pointerleave", () => {
      if (!state.hover) return;
      state.hover = null;
      highlight();
    });
    grid.addEventListener("keydown", (event) => {
      const step = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[event.key];
      if (!step || !result.valid) return;
      event.preventDefault();
      state.pinned = { i: state.pinned.i + step[0], j: state.pinned.j + step[1] };
      state.hover = null;
      highlight();
    });

    $("relu").addEventListener("change", (event) => {
      state.relu = event.target.checked;
      render();
    });
    $("input-preset").addEventListener("change", (event) => {
      state.input = event.target.value;
      render();
    });
    $("filter-preset").addEventListener("change", (event) => setFilter(event.target.value));
    $("restore").addEventListener("click", () => setFilter(state.filter));
    $("rescale").addEventListener("click", () => {
      state.range = Math.max(...result.values.flat().map(Math.abs));
      render();
    });
    $("reset").addEventListener("click", reset);
  }

  buildControls();
  buildTasks();
  bindEvents();
  reset();
})();
