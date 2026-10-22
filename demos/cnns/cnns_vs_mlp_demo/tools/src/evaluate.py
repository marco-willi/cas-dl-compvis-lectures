"""Evaluate all trained runs: own-regime test accuracy, position-accuracy maps, the
float16 check (export seed) and the equivariance table for cnn_gap_tl.

Usage (from experiments/cnn_on_mnist/):
    python evaluate.py                    # maps on the 2000-image subset
    python evaluate.py --map-set full     # maps on all 10k test images
"""

import argparse
import time

import numpy as np
import torch
from common import (
    ARCHS,
    CKPT_DIR,
    EVAL_DIR,
    EXPORT_SEED,
    N_OFFSETS,
    REGIMES,
    SEEDS,
    SPLIT_SEED,
    get_device,
    model_id,
    run_id,
    set_seed,
    write_json,
)
from models import load_checkpoint, predict, round_to_fp16_

from data import eval_subset_indices, fixed_offsets, load_mnist, place, to_float

FP16_MAX_DIFF_PP = 0.2
EQUIV_N_DIGITS = 100
EQUIV_SHIFTS = (1, 2, 3, 4, 8)


def own_regime_acc(model, test_x, test_y, cfg) -> float:
    """Full 10k test set; one fixed offset per image drawn from the model's regime."""
    dx, dy = fixed_offsets(len(test_x), cfg["regime"], cfg["tl_max_offset"], SPLIT_SEED)
    pred, _ = predict(model, place(test_x, dx.to(test_x.device), dy.to(test_x.device)))
    return (pred == test_y).float().mean().item()


@torch.no_grad()
def position_map(model, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy and mean max-softmax confidence per offset; row = dy, column = dx."""
    acc = torch.zeros(N_OFFSETS, N_OFFSETS, device=x.device)
    conf = torch.zeros(N_OFFSETS, N_OFFSETS, device=x.device)
    for dy in range(N_OFFSETS):
        for dx in range(N_OFFSETS):
            pred, c = predict(model, place(x, dx, dy))
            acc[dy, dx] = (pred == y).float().mean()
            conf[dy, dx] = c.mean()
    return acc.cpu().double().numpy(), conf.cpu().double().numpy()


def pattern_stats(acc: np.ndarray) -> dict:
    """Numbers that summarize the expected map patterns (acceptance check 2)."""
    d = np.arange(N_OFFSETS)
    mult4 = (d[None, :] % 4 == 0) & (d[:, None] % 4 == 0)
    return {
        "at_origin": round(float(acc[0, 0]), 4),
        "near_top_left": round(float(acc[:4, :4].mean()), 4),  # dx, dy <= 3
        "far_bottom_right": round(float(acc[21:, 21:].mean()), 4),  # dx, dy >= 21
        "mean": round(float(acc.mean()), 4),
        "min": round(float(acc.min()), 4),
        "dx_dy_multiple_of_4": round(float(acc[mult4].mean()), 4),
        "other_offsets": round(float(acc[~mult4].mean()), 4),
    }


def evaluate_run(model, cfg, test_x, test_y, map_idx) -> dict:
    acc, conf = position_map(model, test_x[map_idx], test_y[map_idx])
    return {
        "test_acc_own_regime": round(own_regime_acc(model, test_x, test_y, cfg), 5),
        "test_acc_all_positions": round(float(acc.mean()), 5),
        "pattern": pattern_stats(acc),
        "acc": acc.round(4).tolist(),
        "conf": conf.round(4).tolist(),
    }


@torch.no_grad()
def equivariance_table(model, digits: torch.Tensor) -> list[dict]:
    """Relative error between f(shift_t(x)) and shift_t(f(x)) for conv1 and conv3 (after ReLU).

    x is the digit at offset (0, 0); shift_t(x) is the same digit at (t, 0), so nothing
    leaves the canvas. In feature space the shift is s = t for conv1 and s = t / 4 for conv3
    (two stride-2 poolings); for t not a multiple of 4 the nearest integer shift is used and
    the row is marked as not exact. A border of s px is cropped on all sides.
    """
    _, f0 = model.forward_with_intermediates(place(digits, 0, 0))
    rows = []
    for layer, factor in (("conv1", 1), ("conv3", 4)):
        for t in EQUIV_SHIFTS:
            _, ft = model.forward_with_intermediates(place(digits, t, 0))
            a, b = ft[layer], f0[layer]
            s = round(t / factor)
            h, w = a.shape[-2:]
            a_c = a[..., s : h - s, s : w - s]  # shifted input, cropped
            b_c = b[..., s : h - s, 0 : w - 2 * s]  # shift_s(f(x)), same region
            num = (a_c - b_c).flatten(1).norm(dim=1)
            den = b_c.flatten(1).norm(dim=1)
            err = (num / den).cpu().numpy()
            rows.append(
                {
                    "layer": layer,
                    "t_px": t,
                    "feature_shift": s,
                    "exact_shift": t % factor == 0,
                    "rel_err_mean": round(float(err.mean()), 6),
                    "rel_err_max": round(float(err.max()), 6),
                }
            )
    return rows


def spread(values: list[float]) -> dict:
    v = np.array(values)
    return {
        "values": [round(float(x), 5) for x in v],
        "mean": round(float(v.mean()), 5),
        "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 5),
        "min": round(float(v.min()), 5),
        "max": round(float(v.max()), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map-set", choices=("subset", "full"), default="subset")
    args = parser.parse_args()

    set_seed(SPLIT_SEED)
    device = get_device()
    images, labels = load_mnist(train=False)
    test_x, test_y = to_float(images).to(device), labels.to(device)
    subset = eval_subset_indices()
    write_json(EVAL_DIR / "eval_subset.json", {"seed": SPLIT_SEED, "indices": subset})
    map_idx = torch.tensor(
        subset if args.map_set == "subset" else list(range(len(labels))), device=device
    )

    t0 = time.time()
    summary = {"map_set": args.map_set, "map_size": len(map_idx), "models": {}}
    for arch in ARCHS:
        for regime in REGIMES:
            mid = model_id(arch, regime)
            per_seed = {}
            for seed in SEEDS:
                rid = run_id(arch, regime, seed)
                path = CKPT_DIR / f"{rid}.pt"
                if not path.exists():
                    print(f"{rid}: missing checkpoint, skipped")
                    continue
                model, cfg = load_checkpoint(path, device)
                res = evaluate_run(model, cfg, test_x, test_y, map_idx)
                if seed == EXPORT_SEED:
                    res["fp16"] = evaluate_run(
                        round_to_fp16_(model), cfg, test_x, test_y, map_idx
                    )
                write_json(EVAL_DIR / f"{rid}.json", {"run": rid, "config": cfg, **res})
                per_seed[seed] = res
                print(
                    f"{rid:16s} own {res['test_acc_own_regime']:.4f}  "
                    f"all-pos {res['test_acc_all_positions']:.4f}  ({time.time() - t0:.0f} s)"
                )
            if EXPORT_SEED not in per_seed:
                continue
            r0 = per_seed[EXPORT_SEED]
            summary["models"][mid] = {
                "seeds": sorted(per_seed),
                "seed_spread": {
                    key: spread([r[key] for r in per_seed.values()])
                    for key in ("test_acc_own_regime", "test_acc_all_positions")
                },
                "fp16_diff_pp": {
                    key: round(100 * (r0["fp16"][key] - r0[key]), 3)
                    for key in ("test_acc_own_regime", "test_acc_all_positions")
                },
                "pattern_fp32": r0["pattern"],
                "pattern_fp16": r0["fp16"]["pattern"],
            }

    diffs = [
        abs(v) for m in summary["models"].values() for v in m["fp16_diff_pp"].values()
    ]
    summary["fp16_max_abs_diff_pp"] = max(diffs) if diffs else None
    summary["export_dtype"] = (
        "float16" if diffs and max(diffs) < FP16_MAX_DIFF_PP else "float32"
    )

    gap_path = CKPT_DIR / f"{run_id('cnn_gap', 'tl', EXPORT_SEED)}.pt"
    if gap_path.exists():
        model, _ = load_checkpoint(gap_path, device)
        digits = test_x[torch.tensor(subset[:EQUIV_N_DIGITS], device=device)]
        summary["equivariance_cnn_gap_tl"] = {
            "n_digits": EQUIV_N_DIGITS,
            "base_offset": [0, 0],
            "shift_direction": "right (dx += t)",
            "activation": "after ReLU",
            "rows": equivariance_table(model, digits),
        }
    summary["eval_time_s"] = round(time.time() - t0, 1)
    write_json(EVAL_DIR / "summary.json", summary)
    print_summary(summary)


def print_summary(summary: dict) -> None:
    print(
        f"\nmaps on {summary['map_set']} ({summary['map_size']} images); export dtype {summary['export_dtype']} "
        f"(max |fp16 - fp32| = {summary['fp16_max_abs_diff_pp']} pp)"
    )
    print(
        f"{'model':12s} {'own':>14s} {'all-pos':>14s} {'origin':>7s} {'near':>6s} {'far':>6s} {'mult4':>6s} {'other':>6s}"
    )
    for mid, m in summary["models"].items():
        own, allp, p = (
            m["seed_spread"]["test_acc_own_regime"],
            m["seed_spread"]["test_acc_all_positions"],
            m["pattern_fp32"],
        )
        print(
            f"{mid:12s} {own['mean']:.4f}±{own['std']:.4f} {allp['mean']:.4f}±{allp['std']:.4f} "
            f"{p['at_origin']:7.3f} {p['near_top_left']:6.3f} {p['far_bottom_right']:6.3f} "
            f"{p['dx_dy_multiple_of_4']:6.3f} {p['other_offsets']:6.3f}"
        )
    for row in summary.get("equivariance_cnn_gap_tl", {}).get("rows", []):
        print(
            f"  {row['layer']} t={row['t_px']} s={row['feature_shift']} exact={row['exact_shift']!s:5s} err={row['rel_err_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
