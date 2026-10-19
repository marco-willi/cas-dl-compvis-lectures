// Exercises of the convolution-parameter demo. Each task has a `check(ctx)` that is evaluated
// after every change; pure functions without DOM access, shared by the page and the tests.
//
// ctx = { input, kernelSize, padding, stride, dilation, relu, result }
//   input:  current input matrix (i x i), result: output of Conv.correlate for the current state
globalThis.ConvExercises = (() => {
  "use strict";

  const EPS = 1e-9;

  // Output rows (or columns) r whose kernel extent covers the input indices lo..hi.
  function coversRange(ctx, lo, hi) {
    const eff = Conv.effectiveKernelSize(ctx.kernelSize, ctx.dilation);
    for (let r = 0; r < ctx.result.size; r++) {
      const first = Conv.sampleIndex(r, 0, ctx.padding, ctx.stride, ctx.dilation);
      if (first <= lo && first + eff - 1 >= hi) return true;
    }
    return false;
  }

  // Cells of a bright shape whose upper neighbour is dark (or outside the input).
  function topEdges(input) {
    return input.map((row, r) => row.map((x, c) => x > 0.5 && (r === 0 || input[r - 1][c] < 0.5)));
  }

  const TASKS = [
    {
      id: "same-size",
      title: "Keep the size",
      text: "Use a 5 × 5 kernel and get an output of the same size as the input (9 × 9).",
      hint: "Which padding compensates for the cells lost at the border?",
      check: (ctx) => ctx.kernelSize === 5 && ctx.result.size === ctx.input.length,
    },
    {
      id: "downsample",
      title: "Reduce the resolution",
      text: "Use a 3 × 3 kernel and get an output of 5 × 5.",
      hint: "Without padding, no stride gives exactly 5. Combine stride and padding.",
      check: (ctx) => ctx.kernelSize === 3 && ctx.result.size === 5,
    },
    {
      id: "whole-t",
      title: "See the whole T with 9 weights",
      text:
        "Use a 3 × 3 kernel such that a single output cell reads from an area that covers the " +
        "whole letter T (rows and columns 1 to 7).",
      hint: "The effective kernel size has to be at least 7. Which parameter enlarges it without adding weights?",
      check: (ctx) => ctx.result.valid && ctx.kernelSize === 3 && coversRange(ctx, 1, 7),
    },
    {
      id: "copy",
      title: "Copy the input",
      text: "Make the output identical to the input (same size, same values).",
      hint: "One preset passes the centre value through. Then adjust the padding to the kernel size.",
      check: (ctx) =>
        ctx.result.size === ctx.input.length &&
        ctx.result.values.every((row, r) => row.every((y, c) => Math.abs(y - ctx.input[r][c]) < EPS)),
    },
    {
      id: "top-edge",
      title: "Detect top edges only",
      text:
        "With ReLU switched on, get a 9 × 9 output that is positive exactly at the top edge of the T " +
        "(the bright cells with a dark cell above them) and zero everywhere else.",
      hint:
        "The preset “Horizontal edges” responds to two rows. Edit the weights: compare each cell " +
        "only with the cell directly above it.",
      check: (ctx) => {
        if (!ctx.relu || ctx.result.size !== ctx.input.length) return false;
        const target = topEdges(ctx.input);
        return ctx.result.values.every((row, r) => row.every((y, c) => y > EPS === target[r][c]));
      },
    },
  ];

  return { TASKS, topEdges, coversRange };
})();
