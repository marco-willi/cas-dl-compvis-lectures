// Augmentation stability demo: manipulate an image and compare EfficientNet-B0 predictions.
"use strict";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const MODEL_DIR = "model/";           // relative to this page, so any site sub-path works
const SAMPLES_URL = "samples/samples.json";

// Larger images are scaled down to these limits once, when they are loaded.
// (Must match tools/prepare_samples.py so that the samples are used unchanged.)
const MAX_SHORT_SIDE = 320;
const MAX_LONG_SIDE = 640;

const FILL = "#000";                  // border fill for areas outside the image
const OCCLUDER = [128, 128, 128];     // colour of the occlusion rectangle
const DEBOUNCE_MS = 120;              // wait this long after the last change before inference
const TOP_K = 5;

// Sliders, in the order in which the manipulations are applied (flip is a checkbox).
const CONTROLS = [
  { key: "zoom",       label: "Zoom",       min: 0.5,  max: 2.5, step: 0.05, value: 1, format: v => v.toFixed(2) + "×" },
  { key: "rotation",   label: "Rotation",   min: -180, max: 180, step: 1,    value: 0, format: v => v + "°" },
  { key: "tx",         label: "Shift x",    min: -50,  max: 50,  step: 1,    value: 0, format: v => v + "%" },
  { key: "ty",         label: "Shift y",    min: -50,  max: 50,  step: 1,    value: 0, format: v => v + "%" },
  { key: "brightness", label: "Brightness", min: 0.2,  max: 2.5, step: 0.05, value: 1, format: v => v.toFixed(2) + "×" },
  { key: "blur",       label: "Blur",       min: 0,    max: 5,   step: 0.25, value: 0, format: v => v.toFixed(2) + " px" },
  { key: "occlusion",  label: "Occlusion",  min: 0,    max: 70,  step: 1,    value: 0, format: v => v === 0 ? "off" : v + "%" },
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const $ = id => document.getElementById(id);

let meta = null;                 // model/meta.json: labels and preprocessing
let source = null;               // { canvas, width, height }: the image all manipulations start from
let original = null;             // prediction for the unmodified image
let params = defaultParams();

function defaultParams() {
  const p = { flip: false, occX: 0.5, occY: 0.5 };
  for (const c of CONTROLS) p[c.key] = c.value;
  return p;
}

// ---------------------------------------------------------------------------
// Status and errors
// ---------------------------------------------------------------------------
function setStatus(text, { progress = null, state = "" } = {}) {
  $("status").className = "status " + state;
  $("status-text").textContent = text;
  if (progress !== null) $("progress").value = progress;
}

function fail(message) {
  setStatus(message, { state: "error" });
}

// ---------------------------------------------------------------------------
// Model: one worker, one inference session, reused for every prediction
// ---------------------------------------------------------------------------
let worker = null;
let nextRequest = 0;
const pending = new Map();       // request id -> { resolve, reject }

async function fetchWithProgress(url, totalBytes, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const total = totalBytes || Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes.buffer;
}

async function loadModel() {
  if (location.protocol === "file:")
    throw new Error("This demo has to be served over HTTP: browsers do not load models from file:// pages. " +
      "Use `quarto preview`, or run `python -m http.server` in the site folder.");
  if (typeof WebAssembly !== "object")
    throw new Error("This browser does not support WebAssembly, which is required to run the model.");

  try {
    meta = await (await fetch(MODEL_DIR + "meta.json")).json();
  } catch (error) {
    throw new Error("Could not load model/meta.json. Run tools/export_model.py to create the model files.");
  }

  const megabytes = bytes => (bytes / 1e6).toFixed(1);
  let model;
  try {
    model = await fetchWithProgress(MODEL_DIR + meta.file, meta.bytes, (received, total) =>
      setStatus(`Downloading model: ${megabytes(received)} / ${megabytes(total)} MB`, { progress: total ? received / total : 0 }));
  } catch (error) {
    throw new Error(`Could not download the model (${error.message}). If the file is missing, run tools/export_model.py; otherwise check your connection and reload the page.`);
  }

  setStatus("Starting ONNX Runtime (WebAssembly)…", { progress: 1 });
  await new Promise((resolve, reject) => {
    worker = new Worker("worker.js");
    worker.onerror = event => reject(new Error("Could not start the inference worker: " + (event.message || "script failed to load")));
    worker.onmessage = ({ data }) => {
      if (data.type === "ready") return resolve();
      if (data.type === "error" && data.stage === "init")
        return reject(new Error("Could not initialise the model: " + data.message));
      const request = pending.get(data.id);
      if (!request) return;
      pending.delete(data.id);
      if (data.type === "result") request.resolve(data);
      else request.reject(new Error(data.message));
    };
    worker.postMessage({ type: "init", model }, [model]);
  });
}

// Run the network on one preprocessed tensor. Requests are processed in order by the worker.
function runModel(tensor) {
  return new Promise((resolve, reject) => {
    const id = nextRequest++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: "run", id, input: tensor });
  }).then(({ logits, ms }) => {
    const probs = softmax(logits);
    const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    return { logits, probs, order, ms };
  });
}

function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const probs = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += probs[i] = Math.exp(logits[i] - max);
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  return probs;
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------
async function decodeImage(blob) {
  // Keep the stored pixel values (no colour management), respect EXIF orientation.
  return createImageBitmap(blob, { colorSpaceConversion: "none", imageOrientation: "from-image" });
}

function setSource(bitmap) {
  const scale = Math.min(1,
    MAX_SHORT_SIDE / Math.min(bitmap.width, bitmap.height),
    MAX_LONG_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#fff"; // transparent uploads are shown on white
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  source = { canvas, ctx, width: canvas.width, height: canvas.height };

  for (const id of ["orig-view", "manip-view", "work"]) {
    const target = id === "work" ? work : $(id);
    target.width = source.width;
    target.height = source.height;
  }
  layout();
}

async function loadSample(sample, button) {
  const response = await fetch(sample.file);
  if (!response.ok) throw new Error(`Could not load ${sample.file} (HTTP ${response.status})`);
  await showImage(await decodeImage(await response.blob()));
  for (const b of $("samples").querySelectorAll("button")) b.setAttribute("aria-pressed", b === button);
}

// Use a new image: show it, predict it, then re-apply the current manipulation.
async function showImage(bitmap) {
  setSource(bitmap);
  original = null;

  const pixels = source.ctx.getImageData(0, 0, source.width, source.height);
  const { tensor, crop } = Preprocess.run(pixels.data, source.width, source.height, meta.preprocessing);
  $("orig-view").getContext("2d").putImageData(pixels, 0, 0);
  $("orig-input").getContext("2d").putImageData(new ImageData(crop, 224, 224), 0, 0);
  $("orig-preds").classList.add("stale");

  const mine = source;
  refresh();
  const result = await runModel(tensor);
  if (mine !== source) return; // another image was chosen in the meantime
  original = result;
  renderPredictions($("orig-preds"), result, false);
  refresh();
}

// ---------------------------------------------------------------------------
// Manipulation: geometry on a canvas, then brightness, blur and occlusion on the pixels
// ---------------------------------------------------------------------------
const work = document.createElement("canvas");
const workCtx = work.getContext("2d", { willReadFrequently: true });

function manipulate(p) {
  const { width: w, height: h } = source;

  // 1-4: flip, zoom, rotation, shift as a single resampling step. Canvas transforms act on a
  // point in reverse order of the calls: scale (flip, zoom) first, then rotate, then translate.
  workCtx.setTransform(1, 0, 0, 1, 0, 0);
  workCtx.fillStyle = FILL;
  workCtx.fillRect(0, 0, w, h);
  workCtx.imageSmoothingEnabled = true;
  workCtx.imageSmoothingQuality = "high";
  workCtx.translate(w / 2 + p.tx / 100 * w, h / 2 + p.ty / 100 * h);
  workCtx.rotate(-p.rotation * Math.PI / 180); // positive angles rotate counter-clockwise
  workCtx.scale(p.flip ? -p.zoom : p.zoom, p.zoom);
  workCtx.drawImage(source.canvas, -w / 2, -h / 2);
  workCtx.setTransform(1, 0, 0, 1, 0, 0);

  const image = workCtx.getImageData(0, 0, w, h);
  const data = image.data;

  // 5: brightness (values above 255 are clipped, as in an overexposed photograph)
  let clipped = 0;
  if (p.brightness !== 1) {
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] * p.brightness, g = data[i + 1] * p.brightness, b = data[i + 2] * p.brightness;
      if (r > 255.5 || g > 255.5 || b > 255.5) clipped++;
      data[i] = r; data[i + 1] = g; data[i + 2] = b; // Uint8ClampedArray rounds and clamps
    }
  }

  // 6: blur
  if (p.blur > 0) gaussianBlur(data, w, h, p.blur);

  // 7: occlusion square, positioned in the frame (it does not move with the image)
  if (p.occlusion > 0) {
    const side = p.occlusion / 100 * Math.min(w, h);
    const x0 = Math.max(0, Math.round(p.occX * w - side / 2)), x1 = Math.min(w, Math.round(p.occX * w + side / 2));
    const y0 = Math.max(0, Math.round(p.occY * h - side / 2)), y1 = Math.min(h, Math.round(p.occY * h + side / 2));
    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++) data.set(OCCLUDER, (y * w + x) * 4);
  }

  return { image, clippedFraction: clipped / (w * h) };
}

// Separable Gaussian blur on RGBA pixels; edge pixels are repeated outwards.
function gaussianBlur(data, w, h, sigma) {
  const radius = Math.ceil(3 * sigma);
  const kernel = new Float32Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) sum += kernel[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

  const tmp = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {          // horizontal pass: data -> tmp
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = -radius; k <= radius; k++) {
        const xx = x + k < 0 ? 0 : x + k >= w ? w - 1 : x + k;
        const s = (y * w + xx) * 4, wgt = kernel[k + radius];
        r += data[s] * wgt; g += data[s + 1] * wgt; b += data[s + 2] * wgt;
      }
      const d = (y * w + x) * 3;
      tmp[d] = r; tmp[d + 1] = g; tmp[d + 2] = b;
    }
  }
  for (let y = 0; y < h; y++) {          // vertical pass: tmp -> data
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = y + k < 0 ? 0 : y + k >= h ? h - 1 : y + k;
        const s = (yy * w + x) * 3, wgt = kernel[k + radius];
        r += tmp[s] * wgt; g += tmp[s + 1] * wgt; b += tmp[s + 2] * wgt;
      }
      const d = (y * w + x) * 4;
      data[d] = r; data[d + 1] = g; data[d + 2] = b;
    }
  }
}

// Apply `p`, update both manipulated previews, and return the tensor for the model.
function renderManipulated(p) {
  const { image, clippedFraction } = manipulate(p);
  const { tensor, crop } = Preprocess.run(image.data, source.width, source.height, meta.preprocessing);
  $("manip-view").getContext("2d").putImageData(image, 0, 0);
  $("manip-input").getContext("2d").putImageData(new ImageData(crop, 224, 224), 0, 0);
  $("manip-note").textContent = clippedFraction > 0.0005
    ? `Brightness clips ${(clippedFraction * 100).toFixed(1)}% of the pixels at white.`
    : p.occlusion > 0 ? "Drag the grey square to move the occlusion." : "";
  return tensor;
}

// ---------------------------------------------------------------------------
// Updating: previews immediately, inference after a short pause, stale results dropped
// ---------------------------------------------------------------------------
let frameRequested = false;
let inferenceTimer = 0;
let latestInference = 0;

function refresh() {
  if (!source || frameRequested) return;
  frameRequested = true;
  requestAnimationFrame(() => {
    frameRequested = false;
    const tensor = renderManipulated(params);
    if (!original) return;

    const ticket = ++latestInference;
    $("manip-preds").classList.add("stale");
    clearTimeout(inferenceTimer);
    inferenceTimer = setTimeout(async () => {
      try {
        const result = await runModel(tensor);
        if (ticket !== latestInference) return; // a newer change is already on its way
        renderPredictions($("manip-preds"), result, true);
        setStatus(`Model ready · WebAssembly · ${result.ms.toFixed(0)} ms per image`, { state: "ready" });
      } catch (error) {
        fail("Inference failed: " + error.message);
      }
    }, DEBOUNCE_MS);
  });
}

// ---------------------------------------------------------------------------
// Predictions
// ---------------------------------------------------------------------------
const percent = p => (p * 100 < 0.1 && p > 0 ? "<0.1" : (p * 100).toFixed(1)) + "%";

function predictionRow(index, prob, isOriginalTop) {
  const row = document.createElement("div");
  row.className = "pred" + (isOriginalTop ? " orig" : "");
  const name = meta.categories[index];
  row.innerHTML = `<span class="name"></span><span class="track"><span class="fill" style="width:${prob * 100}%; display:block"></span></span><span class="pct">${percent(prob)}</span>`;
  row.firstChild.textContent = row.firstChild.title = name;
  return row;
}

function renderPredictions(container, result, isManipulated) {
  container.classList.remove("stale");
  container.replaceChildren();
  const originalTop = original ? original.order[0] : result.order[0];

  const caption = document.createElement("div");
  caption.className = "caption";
  caption.textContent = `Top ${TOP_K} model probabilities (softmax)`;
  container.append(caption);
  for (const index of result.order.slice(0, TOP_K))
    container.append(predictionRow(index, result.probs[index], index === originalTop));

  // Always report the original image's top class, also when it has left the top five.
  const pinned = document.createElement("div");
  pinned.className = "pinned";
  if (isManipulated) {
    const rank = result.order.indexOf(originalTop) + 1;
    pinned.innerHTML = `<div class="caption">Original top class on the manipulated image: rank ${rank} of ${result.order.length}</div>`;
    pinned.append(predictionRow(originalTop, result.probs[originalTop], true));
  } else {
    pinned.innerHTML = `<div class="caption">The <strong style="color:var(--accent)">orange</strong> class is the top prediction here and is tracked on the manipulated image.</div>`;
  }
  container.append(pinned);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
const inputs = {};

function setParam(key, value) {
  params[key] = value;
  syncControls();
  refresh();
}

function syncControls() {
  for (const c of CONTROLS) {
    inputs[c.key].range.value = params[c.key];
    inputs[c.key].output.textContent = c.format(params[c.key]);
  }
  inputs.flip.checked = params.flip;
  $("manip-frame").classList.toggle("movable", params.occlusion > 0);
}

function buildControls() {
  const grid = $("sliders");
  const add = (tag, props = {}) => grid.appendChild(Object.assign(document.createElement(tag), props));

  add("label", { textContent: "Flip", htmlFor: "flip" });
  const flipCell = add("span", { className: "wide" });
  inputs.flip = flipCell.appendChild(Object.assign(document.createElement("input"), { type: "checkbox", id: "flip" }));
  flipCell.append(" horizontal");
  inputs.flip.addEventListener("change", () => setParam("flip", inputs.flip.checked));

  for (const c of CONTROLS) {
    add("label", { textContent: c.label, htmlFor: c.key });
    const range = add("input", { type: "range", id: c.key, min: c.min, max: c.max, step: c.step });
    const output = add("output", { htmlFor: c.key });
    const reset = add("button", { type: "button", className: "reset", textContent: "↺", title: `Reset ${c.label.toLowerCase()}` });
    reset.setAttribute("aria-label", reset.title);
    inputs[c.key] = { range, output };
    range.addEventListener("input", () => setParam(c.key, Number(range.value)));
    reset.addEventListener("click", () => setParam(c.key, c.value));
  }

  $("reset-all").addEventListener("click", () => {
    params = defaultParams();
    syncControls();
    refresh();
  });
  syncControls();
}

// Drag on the manipulated image to move the occlusion square.
function moveOcclusion(event) {
  const rect = $("manip-frame").getBoundingClientRect();
  params.occX = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  params.occY = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
  refresh();
}
$("manip-frame").addEventListener("pointerdown", event => {
  if (params.occlusion === 0) return;
  event.preventDefault();
  $("manip-frame").setPointerCapture(event.pointerId);
  moveOcclusion(event);
});
$("manip-frame").addEventListener("pointermove", event => {
  if (params.occlusion > 0 && $("manip-frame").hasPointerCapture(event.pointerId)) moveOcclusion(event);
});

$("upload").addEventListener("change", async event => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file || !meta || !worker) return;
  try {
    await showImage(await decodeImage(file));
    for (const b of $("samples").querySelectorAll("button")) b.setAttribute("aria-pressed", "false");
  } catch (error) {
    $("manip-note").textContent = `Could not read “${file.name}” as an image.`;
  }
});

// ---------------------------------------------------------------------------
// Layout: size the image frames so that image and model input share one height
// ---------------------------------------------------------------------------
const autoHeight = Boolean(window.frameElement && window.frameElement.hasAttribute("data-auto-height"));

function layout() {
  if (!source) return;
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
  const aspect = source.width / source.height;
  // In a fixed-height frame the images also depend on its height; an auto-height frame adapts instead.
  const maxHeight = autoHeight ? 17 * rem : Math.max(7 * rem, Math.min(17 * rem, 0.25 * window.innerHeight));
  for (const row of document.querySelectorAll(".images")) {
    const available = row.clientWidth - 0.5 * rem; // minus the gap between the two frames
    row.style.setProperty("--img-h", Math.floor(Math.min(maxHeight, available / (aspect + 1))) + "px");
    row.firstElementChild.style.setProperty("--ar", aspect);
  }
  // Dashed marker: the square region kept by resize + center crop, in image coordinates.
  const g = Preprocess.geometry(source.width, source.height, meta.preprocessing);
  for (const marker of document.querySelectorAll(".crop-marker")) {
    marker.style.left = g.left / g.newW * 100 + "%";
    marker.style.top = g.top / g.newH * 100 + "%";
    marker.style.width = meta.preprocessing.crop / g.newW * 100 + "%";
    marker.style.height = meta.preprocessing.crop / g.newH * 100 + "%";
  }
}
window.addEventListener("resize", layout);

// Optional: an embedding page on the same site can add `data-auto-height` to the iframe;
// the iframe then follows the height of the demo instead of using a fixed height.
if (autoHeight) {
  new ResizeObserver(() => {
    window.frameElement.style.height = Math.ceil(document.body.getBoundingClientRect().height) + 2 + "px";
  }).observe(document.body);
}

// ---------------------------------------------------------------------------
// Self-check: compare browser results with the Python references in model/reference.json
// ---------------------------------------------------------------------------
async function selfCheck() {
  const out = $("selftest-out");
  out.textContent = "Running…";
  const reference = await (await fetch(MODEL_DIR + "reference.json")).json();
  const rows = [];

  // 1. Same synthetic tensor as tools/export_model.py -> compare all 1000 logits.
  const input = new Float32Array(3 * 224 * 224);
  let state = reference.synthetic.seed;
  for (let i = 0; i < input.length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    input[i] = state / 4294967296 * 2 - 1;
  }
  const { logits } = await runModel(input);
  let logitDiff = 0;
  reference.synthetic.logits.forEach((v, i) => { logitDiff = Math.max(logitDiff, Math.abs(v - logits[i])); });
  rows.push({ name: "Network (fixed input tensor)", detail: `max. logit difference ${logitDiff.toExponential(1)}`, ok: logitDiff < 1e-2 });

  // 2. Sample images through the browser's preprocessing -> compare top-5 probabilities.
  const samples = await (await fetch(SAMPLES_URL)).json();
  for (const sample of samples) {
    const bitmap = await decodeImage(await (await fetch(sample.file)).blob());
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const result = await runModel(Preprocess.run(pixels, canvas.width, canvas.height, meta.preprocessing).tensor);
    const expected = reference.samples[sample.key].top;
    let probDiff = 0;
    for (const { index, prob } of expected) probDiff = Math.max(probDiff, Math.abs(prob - result.probs[index]));
    rows.push({
      name: `Image + preprocessing: ${sample.label}`,
      detail: `${meta.categories[result.order[0]]} ${percent(result.probs[result.order[0]])}, max. probability difference ${probDiff.toExponential(1)}`,
      ok: result.order[0] === expected[0].index && probDiff < 1e-2,
    });
  }

  out.innerHTML = "<table>" + rows.map(r =>
    `<tr><td class="${r.ok ? "ok" : "bad"}">${r.ok ? "✓" : "✗"}</td><td>${r.name}</td><td>${r.detail}</td></tr>`).join("") +
    "</table>Compared with PyTorch / torchvision (" + meta.weights + ").";
  return rows;
}
$("selftest").addEventListener("click", () => selfCheck().catch(error => { $("selftest-out").textContent = "Self-check failed: " + error.message; }));

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
async function main() {
  buildControls();
  try {
    await loadModel();
  } catch (error) {
    return fail(error.message);
  }
  setStatus("Model ready · WebAssembly", { state: "ready" });

  try {
    const samples = await (await fetch(SAMPLES_URL)).json();
    const buttons = samples.map(sample => {
      const button = document.createElement("button");
      button.type = "button";
      button.title = button.ariaLabel = sample.label;
      button.style.backgroundImage = `url("${sample.file}")`;
      button.addEventListener("click", () => loadSample(sample, button).catch(error => fail(error.message)));
      $("samples").insertBefore(button, $("samples").lastElementChild);
      return button;
    });
    await loadSample(samples[0], buttons[0]);
  } catch (error) {
    return fail("Could not load the sample images: " + error.message);
  }

  if (new URLSearchParams(location.search).has("selftest")) {
    $("verify").open = true;
    window.selfCheckResult = await selfCheck();
  }
}
main();
