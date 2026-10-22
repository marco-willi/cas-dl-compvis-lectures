// The exported model files assign to `window`, which Deno 2 does not define.
(globalThis as any).window = globalThis;
