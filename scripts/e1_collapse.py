"""E1: the normalized RNTK for any branch scales is close to K_T, T = sum eta_l.

For each schedule of branch scales, the sup over augmented angles of
|rntk_discrete - kernel_T(T)| is recorded with T and eta_max = max_l eta_l.
"""
import argparse

import numpy as np

from effdepth import kernels as K
from effdepth.runio import Run

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
args = ap.parse_args()

run = Run(args.run_dir, {"experiment": "e1_collapse"})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()

angles = np.unique(np.concatenate([np.geomspace(1e-6, 0.1, 60), np.linspace(0.1, np.pi / 2, 120)]))
cache = {}


def ref(T):
    key = round(T, 12)
    if key not in cache:
        cache[key] = K.kernel_T(angles, T)
    return cache[key]


records = []


def record(kind, params, alphas):
    a = np.asarray(alphas, float)
    eta = a ** 2 / (1 + a ** 2)
    T = float(eta.sum())
    gap = np.abs(K.rntk_discrete(angles, a) - ref(T))
    records.append({"schedule": kind, **params, "L": int(a.size), "T": T,
                    "eta_max": float(eta.max()), "gap": float(gap.max()),
                    "argmax_angle": float(angles[np.argmax(gap)])})
    print(kind, params, f"L={a.size} T={T:.4g} eta_max={eta.max():.3g} gap={gap.max():.3e}", flush=True)


# constant scale: L = T / eta
for alpha in [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
    eta = alpha ** 2 / (1 + alpha ** 2)
    for T in [0.25, 1.0, 4.0, 16.0, 64.0, 256.0]:
        L = int(round(T / eta))
        if L >= 1:
            record("constant", {"alpha": alpha, "T_target": T}, np.full(L, alpha))

# alpha = L^-gamma (Huang et al. 2020; Belfer et al. 2024; Yu et al. 2026)
for gamma in [0.25, 0.5, 0.75, 1.0]:
    for L in [16, 64, 256, 1024, 4096]:
        record("power", {"gamma": gamma}, np.full(L, L ** -gamma))

# alpha_l = l^-1/2 (Hayou et al., scaled ResNet): T = H_{L+1} - 1 ~ log L
for L in [16, 128, 1024, 8192]:
    record("inv_sqrt_l", {}, np.arange(1, L + 1) ** -0.5)

# square-summable decreasing scales (Stable ResNet's experimental sequence)
for L in [16, 128, 1024, 8192]:
    l = np.arange(1, L + 1)
    record("stable", {}, 1 / (np.sqrt(l) * np.log(l + 1)))

# random layer-wise scales with a given eta_max
rng = np.random.default_rng(0)
for eta_max in [0.4, 0.1, 0.025]:
    for T in [1.0, 16.0]:
        eta = rng.uniform(0, eta_max, size=int(2 * T / eta_max) + 1)
        eta *= T / eta.sum()
        eta = np.minimum(eta, 0.999)
        record("random", {"eta_cap": eta_max}, np.sqrt(eta / (1 - eta)))

ratios = [r["gap"] / r["eta_max"] for r in records]
run.finish({"records": records, "angles": angles,
            "max_gap_over_eta": float(max(ratios)),
            "max_gap_over_eta_eta_le_0.25": float(max(r["gap"] / r["eta_max"] for r in records
                                                       if r["eta_max"] <= 0.25))})
