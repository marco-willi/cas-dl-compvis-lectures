// Interface of the softmax demo. Presets: config.js, computation: softmax.js.
(() => {
  "use strict";

  const CFG = globalThis.SoftmaxDemoConfig;
  const S = globalThis.Softmax;
  const $ = (id) => document.getElementById(id);

  const K = CFG.CLASSES.length;
  const LO = CFG.LOGIT_MIN;
  const HI = CFG.LOGIT_MAX;
  const CUSTOM = "custom";

  let logits; // current logits
  let preset; // id of the selected preset, or CUSTOM after an edit
  const cols = { logit: [], prob: [] };
  const fields = [];

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const fmt = (value, digits) => value.toFixed(digits).replace("-", "−");
  // Vertical position of a logit in the chart, as a fraction from the top.
  const fromTop = (z) => (HI - z) / (HI - LO);

  // ---------------------------------------------------------------------------
  // State changes
  // ---------------------------------------------------------------------------
  function setPreset(id) {
    preset = id;
    logits = CFG.PRESETS.find((p) => p.id === id).logits.slice();
    render();
  }

  function setLogits(values) {
    logits = values.map((z) => S.snap(z, CFG.STEP, LO, HI));
    preset = CUSTOM;
    render();
  }

  // ---------------------------------------------------------------------------
  // Building (once)
  // ---------------------------------------------------------------------------
  function build() {
    for (const p of CFG.PRESETS) $("preset").append(new Option(p.name, p.id));
    const custom = new Option("Custom (edited)", CUSTOM);
    custom.hidden = true;
    $("preset").append(custom);

    for (const id of ["logit-chart", "prob-chart", "logit-labels", "prob-labels", "inputs"]) $(id).style.setProperty("--k", K);

    // Axis lines: logits at LO, 0, HI (+ halfway), probabilities at 0, 0.5, 1
    const axis = (chart, top, label, grid) => {
      const line = el("div", grid ? "axis grid" : "axis");
      line.style.top = `${top * 100}%`;
      line.append(el("span", "", label));
      chart.append(line);
    };
    for (const z of [HI, HI / 2, 0, LO / 2, LO]) axis($("logit-chart"), fromTop(z), fmt(z, 0), z !== 0);
    for (const p of [1, 0.5, 0]) axis($("prob-chart"), 1 - p, String(p), p !== 0);

    CFG.CLASSES.forEach((name, k) => {
      for (const kind of ["logit", "prob"]) {
        const col = el("div", "col");
        col.append(el("div", "bar"), el("div", "value"));
        col.dataset.k = k;
        $(kind + "-chart").append(col);
        cols[kind].push(col);
        $(kind + "-labels").append(el("span", "", name));
      }
      const field = el("input");
      field.type = "number";
      field.step = String(CFG.STEP);
      field.min = String(LO);
      field.max = String(HI);
      field.setAttribute("aria-label", `Logit of class ${name}`);
      field.addEventListener("input", () => {
        if (!Number.isFinite(field.valueAsNumber)) return; // e.g. while typing "-"
        const next = logits.slice();
        next[k] = field.valueAsNumber;
        setLogits(next);
      });
      field.addEventListener("blur", () => (field.value = String(logits[k])));
      fields.push(field);
      $("inputs").append(field);
    });
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------
  function render() {
    const probs = S.softmax(logits);
    const top = S.argmax(probs);
    const zero = fromTop(0);

    $("preset").value = preset;
    $("preset").options[$("preset").options.length - 1].hidden = preset !== CUSTOM;

    logits.forEach((z, k) => {
      // Logit bar from the zero line up (positive) or down (negative)
      const col = cols.logit[k];
      const bar = col.firstChild;
      const value = col.lastChild;
      const y = fromTop(z);
      bar.style.top = `${Math.min(y, zero) * 100}%`;
      bar.style.height = `${Math.abs(y - zero) * 100}%`;
      value.textContent = fmt(z, 1);
      // Labels close to the ends of the axis go inside the bar, so they stay within the chart.
      const insideLogit = Math.abs(z) > 0.85 * Math.max(HI, -LO);
      value.style.top =
        z >= 0 === !insideLogit ? `calc(${y * 100}% - 1.15rem)` : `calc(${y * 100}% + 0.15rem)`;
      value.classList.toggle("inside", insideLogit);
      col.classList.toggle("top", k === top);

      // Probability bar from 0 up
      const pcol = cols.prob[k];
      pcol.firstChild.style.height = `${probs[k] * 100}%`;
      pcol.lastChild.textContent = fmt(probs[k], 3);
      const insideProb = probs[k] > 0.88;
      pcol.lastChild.style.bottom = insideProb ? `calc(${probs[k] * 100}% - 1.2rem)` : `calc(${probs[k] * 100}% + 0.15rem)`;
      pcol.lastChild.classList.toggle("inside", insideProb);
      pcol.classList.toggle("top", k === top);

      if (document.activeElement !== fields[k]) fields[k].value = String(z);
    });
    for (const id of ["logit-labels", "prob-labels"]) {
      [...$(id).children].forEach((label, k) => label.classList.toggle("top", k === top));
    }

    // Facts
    const sum = probs.reduce((a, b) => a + b, 0);
    const h = S.entropy(probs);
    // e^z with a superscript exponent
    const power = (z) => {
      const node = el("span", "", "e");
      node.append(el("sup", "", fmt(z, 1)));
      return node;
    };
    const ratio = [power(logits[top]), " / ("];
    logits.forEach((z, k) => ratio.push(...(k ? [" + ", power(z)] : [power(z)])));
    ratio.push(`) = ${fmt(probs[top], 3)}`);
    const rows = [
      ["Prediction", `${CFG.CLASSES[top]} (P = ${fmt(probs[top], 3)})`, ""],
      [`P(${CFG.CLASSES[top]})`, ratio, ""],
      ["Sum of probabilities", fmt(sum, 3), "always 1"],
      ["Entropy H", `${fmt(h, 2)} nats`, `0 = certain, ln ${K} = ${fmt(Math.log(K), 2)} = uniform`],
    ];
    $("facts").replaceChildren(
      ...rows.map(([term, text, note]) => {
        const row = el("div");
        const dd = el("dd");
        dd.append(...(Array.isArray(text) ? text : [text]), " ");
        if (note) dd.append(el("span", "note", `(${note})`));
        row.append(el("dt", "", term), dd);
        return row;
      })
    );
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
  if (autoHeight) {
    const frame = window.frameElement;
    new ResizeObserver(() => {
      const borderBox = window.parent.getComputedStyle(frame).boxSizing === "border-box";
      const chrome = borderBox ? frame.offsetHeight - window.innerHeight : 0;
      frame.style.height = Math.ceil(document.body.getBoundingClientRect().height) + chrome + "px";
    }).observe(document.body);
  }

  // ---------------------------------------------------------------------------
  // Events: dragging in a column of the logit chart sets that logit to the pointer height
  // ---------------------------------------------------------------------------
  function bindEvents() {
    const chart = $("logit-chart");
    let dragging = null; // class index while a drag is in progress

    const valueAt = (event) => {
      const rect = chart.getBoundingClientRect();
      return HI - ((event.clientY - rect.top) / rect.height) * (HI - LO);
    };
    const drag = (event) => {
      const next = logits.slice();
      next[dragging] = valueAt(event);
      setLogits(next);
    };
    chart.addEventListener("pointerdown", (event) => {
      const col = event.target.closest(".col");
      if (!col) return;
      dragging = Number(col.dataset.k);
      chart.setPointerCapture(event.pointerId);
      drag(event);
    });
    chart.addEventListener("pointermove", (event) => {
      if (dragging !== null) drag(event);
    });
    const stop = () => (dragging = null);
    chart.addEventListener("pointerup", stop);
    chart.addEventListener("pointercancel", stop);

    $("preset").addEventListener("change", (event) => setPreset(event.target.value));
    $("shift").addEventListener("click", () => setLogits(logits.map((z) => z + 1)));
    $("double").addEventListener("click", () => setLogits(logits.map((z) => 2 * z)));
    $("halve").addEventListener("click", () => setLogits(logits.map((z) => z / 2)));
    $("reset").addEventListener("click", () => setPreset(CFG.DEFAULT_PRESET));
  }

  build();
  bindEvents();
  setPreset(CFG.DEFAULT_PRESET);
})();
