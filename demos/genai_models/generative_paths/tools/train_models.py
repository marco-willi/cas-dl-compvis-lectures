"""Train the GAN and the VAE of the generative-paths demo and export them for the page.

Both models are trained on 1,000 samples of the shared 1-D target (0.4 N(2, 0.8^2) + 0.6 N(7, 1^2))
with fixed seeds, so every run gives the same result. The page needs no network: it evaluates the
trained models from lookup tables (generator output on a grid of z, discriminator output on a grid
of x, decoder and encoder parameters on grids), interpolated linearly.

Writes, next to `index.html`:

    data/models.js        globalThis.TRAINED_MODELS = {...}  (loaded with a <script> tag, works from file://)
    tests/fixtures.json   reference values of the exact (ideal) models, computed here independently
                          of toy.js with the same formulas; tests/toy_test.ts compares both.

Usage (from the repository root, CPU, about a minute):

    uv run python demos/genai_models/generative_paths/tools/train_models.py
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEMO_DIR = Path(__file__).resolve().parents[1]
TARGET = {"weights": [0.4, 0.6], "means": [2.0, 7.0], "stds": [0.8, 1.0]}
SEED = 2026
N_TRAIN = 1000
# Training works in standardized coordinates (mean 5, std of the target 2.62); exported values
# are in data coordinates.
SHIFT = 5.0
SCALE = math.sqrt(6.856)
Z_GRID = np.linspace(-4, 4, 161)
X_GRID = np.linspace(-4.5, 11.5, 161)
N_EVAL = 20000  # samples per W1 / KS evaluation

GAN_RUNS = {
    # A standard setup: non-saturating loss, Adam with beta1 = 0.5.
    "standard": {"lr": 1e-3, "steps": 6000, "batch": 256, "hidden": 64},
    # Deliberately unstable: a learning rate ten times larger and small batches.
    "unstable": {"lr": 1e-2, "steps": 6000, "batch": 32, "hidden": 64},
}
GAN_SNAPSHOTS = [0, 25, 50, 75, 100, 150, 200, 300, 400, 800, 1500, 3000, 6000]
VAE_RUN = {
    "lr": 2e-3,
    "steps": 6000,
    "batch": 256,
    "hidden": 64,
    "beta": 1.0,
    "min_std": 0.03,
}


# ---- target and comparisons (same definitions as toy.js) -----------------------------------------


def target_cdf(x):
    x = np.asarray(x, dtype=float)
    erfc = np.vectorize(math.erfc)
    return sum(
        w * 0.5 * erfc(-(x - m) / (s * math.sqrt(2)))
        for w, m, s in zip(TARGET["weights"], TARGET["means"], TARGET["stds"])
    )


def sample_target(n, rng):
    k = rng.random(n) < TARGET["weights"][0]
    means = np.where(k, TARGET["means"][0], TARGET["means"][1])
    stds = np.where(k, TARGET["stds"][0], TARGET["stds"][1])
    return means + stds * rng.standard_normal(n)


def wasserstein1(samples, lo=-15.0, hi=25.0, step=0.002):
    """Integral of |F_n(x) - F(x)| dx on a grid, as in toy.js."""
    grid = np.arange(lo + step / 2, hi, step)
    empirical = np.searchsorted(np.sort(samples), grid, side="right") / len(samples)
    return float(np.sum(np.abs(empirical - target_cdf(grid))) * step)


def ks(samples):
    s = np.sort(samples)
    f = target_cdf(s)
    n = len(s)
    i = np.arange(n)
    return float(max(np.max(np.abs(f - i / n)), np.max(np.abs(f - (i + 1) / n))))


def mlp(n_in, n_out, hidden):
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden, hidden),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden, n_out),
    )


def rounded(a, digits=4):
    return [round(float(v), digits) for v in np.asarray(a).ravel()]


# ---- GAN -------------------------------------------------------------------------------------------


def train_gan(data, cfg, seed):
    torch.manual_seed(seed)
    gen = mlp(1, 1, cfg["hidden"])
    disc = mlp(1, 1, cfg["hidden"])
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    real_all = torch.tensor((data - SHIFT) / SCALE, dtype=torch.float32).unsqueeze(1)
    z_eval = torch.randn(N_EVAL, 1, generator=torch.Generator().manual_seed(seed + 1))
    snapshots = []
    losses_d, losses_g = [], []
    for step in range(cfg["steps"] + 1):
        if step in GAN_SNAPSHOTS:
            with torch.no_grad():
                g_grid = (
                    gen(torch.tensor(Z_GRID, dtype=torch.float32).unsqueeze(1))
                    .squeeze(1)
                    .numpy()
                    * SCALE
                    + SHIFT
                )
                x_std = torch.tensor(
                    (X_GRID - SHIFT) / SCALE, dtype=torch.float32
                ).unsqueeze(1)
                d_grid = torch.sigmoid(disc(x_std)).squeeze(1).numpy()
                samples = gen(z_eval).squeeze(1).numpy() * SCALE + SHIFT
            snapshots.append(
                {
                    "step": step,
                    "g": rounded(g_grid),
                    "d": rounded(d_grid, 3),
                    "w1": round(wasserstein1(samples), 4),
                    "ks": round(ks(samples), 4),
                    "lossD": round(float(np.mean(losses_d)), 4) if losses_d else None,
                    "lossG": round(float(np.mean(losses_g)), 4) if losses_g else None,
                }
            )
            losses_d, losses_g = [], []
        if step == cfg["steps"]:
            break
        real = real_all[torch.randint(0, len(real_all), (cfg["batch"],))]
        fake = gen(torch.randn(cfg["batch"], 1))
        loss_d = bce(disc(real), torch.ones_like(real)) + bce(
            disc(fake.detach()), torch.zeros_like(real)
        )
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        loss_g = bce(disc(fake), torch.ones_like(real))  # non-saturating generator loss
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        losses_d.append(loss_d.item())
        losses_g.append(loss_g.item())
    return snapshots


# ---- VAE -------------------------------------------------------------------------------------------


def train_vae(data, cfg, seed):
    torch.manual_seed(seed)
    enc = mlp(1, 2, cfg["hidden"])  # x -> (mu, log sigma) of q(z | x)
    dec = mlp(1, 2, cfg["hidden"])  # z -> (mu, sigma) of p(x | z)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=cfg["lr"])
    x_all = torch.tensor((data - SHIFT) / SCALE, dtype=torch.float32).unsqueeze(1)

    def decode(z):
        out = dec(z)
        return out[:, :1], nn.functional.softplus(out[:, 1:]) + cfg["min_std"] / SCALE

    history = []
    for step in range(cfg["steps"]):
        x = x_all[torch.randint(0, len(x_all), (cfg["batch"],))]
        e = enc(x)
        mu_q, log_s_q = e[:, :1], e[:, 1:]
        z = mu_q + torch.exp(log_s_q) * torch.randn_like(mu_q)  # reparameterization
        mu_p, s_p = decode(z)
        nll = (
            0.5 * ((x - mu_p) / s_p) ** 2 + torch.log(s_p) + 0.5 * math.log(2 * math.pi)
        )
        kl = 0.5 * (mu_q**2 + torch.exp(2 * log_s_q) - 1) - log_s_q
        loss = (nll + cfg["beta"] * kl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == cfg["steps"] - 1:
            history.append(
                {
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "kl": round(kl.mean().item(), 4),
                }
            )

    with torch.no_grad():
        mu_d, s_d = decode(torch.tensor(Z_GRID, dtype=torch.float32).unsqueeze(1))
        e = enc(
            torch.tensor((X_GRID - SHIFT) / SCALE, dtype=torch.float32).unsqueeze(1)
        )
        gen = torch.Generator().manual_seed(seed + 1)
        z = torch.randn(N_EVAL, 1, generator=gen)
        m, s = decode(z)
        with_noise = (m + s * torch.randn(N_EVAL, 1, generator=gen)).squeeze(
            1
        ).numpy() * SCALE + SHIFT
        mean_only = m.squeeze(1).numpy() * SCALE + SHIFT
    return {
        "decMean": rounded(mu_d.squeeze(1).numpy() * SCALE + SHIFT),
        "decStd": rounded(s_d.squeeze(1).numpy() * SCALE),
        "encMean": rounded(e[:, 0].numpy()),
        "encStd": rounded(np.exp(e[:, 1].numpy())),
        "w1": round(wasserstein1(with_noise), 4),
        "ks": round(ks(with_noise), 4),
        "w1MeanOnly": round(wasserstein1(mean_only), 4),
        "history": history,
    }


# ---- reference values of the ideal models -----------------------------------------------------------


def mix_pdf(x, mix):
    return sum(
        w * math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        for w, m, s in zip(*mix)
    )


def mix_cdf(x, mix):
    return sum(
        w * 0.5 * math.erfc(-(x - m) / (s * math.sqrt(2))) for w, m, s in zip(*mix)
    )


def mix_quantile(p, mix):
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mix_cdf(mid, mix) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def target_mix():
    return (TARGET["weights"], TARGET["means"], TARGET["stds"])


def fm_marginal(t):
    w, m, s = target_mix()
    return (
        w,
        [t * mm for mm in m],
        [math.sqrt((1 - t) ** 2 + t * t * ss * ss) for ss in s],
    )


def fm_velocity(x, t):
    w, m, s = target_mix()
    _, mt, st = fm_marginal(t)
    comp = [ww * math.exp(-0.5 * ((x - a) / b) ** 2) / b for ww, a, b in zip(w, mt, st)]
    total = sum(comp)
    return sum(
        c / total * (mm + (t * ss * ss - (1 - t)) / b**2 * (x - a))
        for c, mm, ss, a, b in zip(comp, m, s, mt, st)
    )


def alpha_bar(t):
    return math.exp(-(0.1 * t + 0.5 * (20 - 0.1) * t * t))


def diff_marginal(t):
    w, m, s = target_mix()
    ab = alpha_bar(t)
    return (
        w,
        [math.sqrt(ab) * mm for mm in m],
        [math.sqrt(ab * ss * ss + 1 - ab) for ss in s],
    )


def diff_score_and_denoise(x, t):
    w, m, s = target_mix()
    a = math.sqrt(alpha_bar(t))
    _, mt, st = diff_marginal(t)
    comp = [ww * math.exp(-0.5 * ((x - c) / b) ** 2) / b for ww, c, b in zip(w, mt, st)]
    total = sum(comp)
    r = [c / total for c in comp]
    score = sum(rr * -(x - c) / b**2 for rr, c, b in zip(r, mt, st))
    denoise = sum(
        rr * (mm + a * ss * ss / b**2 * (x - c))
        for rr, mm, ss, c, b in zip(r, m, s, mt, st)
    )
    return score, denoise


def nf_stage(lam):
    w, m, s = target_mix()
    ramp = lambda stage: min(max(3 * lam - stage, 0.0), 1.0)  # noqa: E731
    mean = sum(ww * mm for ww, mm in zip(w, m))
    return (
        w,
        [ramp(0) * (mm - mean) + ramp(1) * mean for mm in m],
        [1 + ramp(2) * (ss - 1) for ss in s],
    )


def nf_forward(h, k, K):
    return mix_quantile(mix_cdf(h, nf_stage((k - 1) / K)), nf_stage(k / K))


def ar_tables(tau):
    lo, coarse, fine = -1, 12, 10
    joint = []
    for c in range(coarse):
        row = []
        for f in range(fine):
            a = 0.0 if (c == 0 and f == 0) else mix_cdf(lo + c + f / fine, target_mix())
            b = (
                1.0
                if (c == coarse - 1 and f == fine - 1)
                else mix_cdf(lo + c + (f + 1) / fine, target_mix())
            )
            row.append(b - a)
        joint.append(row)
    pc = [sum(r) for r in joint]

    def temper(p):
        q = [v ** (1 / tau) if v > 0 else 0.0 for v in p]
        return [v / sum(q) for v in q]

    return temper(pc), [temper([v / pc[c] for v in row]) for c, row in enumerate(joint)]


def fixtures():
    xs = [-2.0, 0.5, 2.0, 4.5, 7.0, 9.5]
    ts = [0.0, 0.1, 0.5, 0.9, 1.0]
    out = {
        "alphaBar": [[t, alpha_bar(t)] for t in ts],
        "fmVelocity": [[x, t, fm_velocity(x, t)] for x in xs for t in ts],
        "diffScore": [
            [x, t, diff_score_and_denoise(x, t)[0]]
            for x in xs
            for t in [0.05, 0.3, 0.7, 1.0]
        ],
        "diffDenoise": [
            [x, t, diff_score_and_denoise(x, t)[1]]
            for x in xs
            for t in [0.05, 0.3, 0.7, 1.0]
        ],
        "nfForward": [
            [h, k, 3, nf_forward(h, k, 3)]
            for h in [-2.0, -0.5, 0.0, 1.0, 2.5]
            for k in [1, 2, 3]
        ],
        "targetCdf": [[x, mix_cdf(x, target_mix())] for x in xs],
    }
    for tau in [0.5, 1.0, 2.0]:
        pc, pf = ar_tables(tau)
        out[f"arTables_{tau}"] = {"pc": pc, "pf3": pf[3], "pf8": pf[8]}
    return out


def main():
    rng = np.random.default_rng(SEED)
    data = sample_target(N_TRAIN, rng)
    print(
        f"training data: {N_TRAIN} samples, W1 to the target {wasserstein1(data):.3f}"
    )

    gans = {}
    for name, cfg in GAN_RUNS.items():
        snaps = train_gan(data, cfg, SEED)
        gans[name] = {"config": cfg, "snapshots": snaps}
        for s in snaps:
            print(
                f"GAN {name:9s} step {s['step']:5d}  W1 {s['w1']:.3f}  KS {s['ks']:.3f}  lossD {s['lossD']}  lossG {s['lossG']}"
            )

    vae = train_vae(data, VAE_RUN, SEED)
    vae["config"] = VAE_RUN
    print(
        f"VAE  W1 {vae['w1']:.3f}  KS {vae['ks']:.3f}  W1 (decoder mean only) {vae['w1MeanOnly']:.3f}"
    )

    models = {
        "about": "Generated by tools/train_models.py. GAN and VAE trained on 1,000 samples of the target.",
        "seed": SEED,
        "nTrain": N_TRAIN,
        "trainingData": rounded(data[:300], 3),
        "zGrid": {"lo": float(Z_GRID[0]), "hi": float(Z_GRID[-1]), "n": len(Z_GRID)},
        "xGrid": {"lo": float(X_GRID[0]), "hi": float(X_GRID[-1]), "n": len(X_GRID)},
        "gan": gans,
        "vae": vae,
    }
    js = (
        "// Generated by tools/train_models.py. Do not edit.\nglobalThis.TRAINED_MODELS = "
        + json.dumps(models, separators=(",", ":"))
        + ";\n"
    )
    (DEMO_DIR / "data").mkdir(exist_ok=True)
    (DEMO_DIR / "data" / "models.js").write_text(js)
    (DEMO_DIR / "tests" / "fixtures.json").write_text(
        json.dumps(fixtures(), indent=1) + "\n"
    )
    print(f"wrote data/models.js ({len(js) / 1024:.0f} KB) and tests/fixtures.json")


if __name__ == "__main__":
    main()
