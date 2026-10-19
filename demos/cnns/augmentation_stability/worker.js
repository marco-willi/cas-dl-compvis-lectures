// Inference worker: owns the single ONNX Runtime session so the page stays responsive.
"use strict";

importScripts("vendor/ort/ort.wasm.min.js");

// Load the WebAssembly binary from this demo's own folder (works under any site sub-path).
ort.env.wasm.wasmPaths = new URL("vendor/ort/", self.location.href).href;
// GitHub Pages cannot send the headers needed for multi-threaded WebAssembly.
ort.env.wasm.numThreads = 1;

let session = null;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      session = await ort.InferenceSession.create(new Uint8Array(data.model), {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      self.postMessage({ type: "ready" });
    } else if (data.type === "run") {
      const start = performance.now();
      const input = new ort.Tensor("float32", data.input, [1, 3, 224, 224]);
      const output = await session.run({ input });
      self.postMessage({
        type: "result",
        id: data.id,
        logits: output.logits.data,
        ms: performance.now() - start,
      });
    }
  } catch (error) {
    self.postMessage({ type: "error", id: data.id, stage: data.type, message: String(error && error.message || error) });
  }
};
