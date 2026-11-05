"""E2: large effective depth.

(a) T (K_T - 1/4) -> h(theta), with the O(1/T) correction;
(b) near-diagonal profile K_T(v/T) -> F(3 pi / v);
(c) degeneration of the discrete kernel under alpha = L^-gamma, to depth 1e7 for
    constant alpha (one pass: for constant alpha the depth-L kernel is read off
    the same recursion at every L), and the mean over random pairs on S^2 as in
    Fig. 1 of Yu et al. (2026).
"""
import argparse

import numpy as np

from effdepth import kernels as K
from effdepth.data import sphere
from effdepth.runio import Run

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--max-log10-depth", type=int, default=7)
args = ap.parse_args()

run = Run(args.run_dir, vars(args) | {"experiment": "e2_large_depth"})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()
out = {}

# (a) first-order term
th = np.array([0.05, 0.1, 0.3, 0.6, 1.0, np.pi / 2])
Ts = np.geomspace(10, 1e5, 9)
h = K.first_order_term(th)
out["first_order"] = {"angles": th, "h": h, "T": Ts,
                      "T_times_K_minus_quarter": [T * (K.kernel_T(th, T) - 0.25) for T in Ts]}
print("first-order done", flush=True)

# (b) profile
v = np.geomspace(0.03, 300, 40)
prof = {}
for T in [10.0, 100.0, 1000.0, 10000.0]:
    ok = v / T < np.pi / 2
    prof[str(T)] = {"v": v[ok], "K": K.kernel_T(v[ok] / T, T)}
out["profile"] = {"by_T": prof, "v": v, "F": K.profile(3 * np.pi / v)}
print("profile done", flush=True)

# random pairs on S^2, augmented angle (Fig. 1 of the source)
rng = np.random.default_rng(0)
x, y = sphere(100, 3, rng), sphere(100, 3, rng)
pair_angles = K.augmented_angle(np.sum(x * y, axis=1))
fixed = np.array([0.3, 1.0, np.pi / 2])
probe = np.concatenate([fixed, pair_angles])


def stable_pass(alpha, depths):
    """Constant alpha: kernel at each depth in `depths` from one forward recursion."""
    a2 = alpha ** 2
    eta = a2 / (1 + a2)
    th = probe.copy()
    e = 2 * np.sin(th / 2) ** 2
    log_q = np.zeros_like(th)
    acc = np.zeros_like(th)
    res = {}
    want = set(int(d) for d in depths)
    for i in range(1, max(want) + 1):
        if i > 1:
            log_q += np.log1p(-eta * th / np.pi)
        acc += eta * K.phi0(th) * np.exp(-log_q)
        e = e - eta * K.gap(th)
        th = K.angle_from_e(e)
        if i in want:
            # depth i: factors up to block i only
            res[i] = np.exp(log_q) * acc / (2 * eta * i)
    return res


degeneration = {}
depths = [int(10 ** k) for k in range(2, args.max_log10_depth + 1)]
depths += [50000, 150000, 200000]
for gamma in [0.0, 0.125, 0.25, 0.375]:
    rows = []
    if gamma == 0.0:
        vals = stable_pass(1.0, depths)
        for L in sorted(vals):
            rows.append({"L": L, "T": L / 2, "values": vals[L]})
    else:
        for L in sorted(set(depths)):
            if L > 10 ** 6:
                continue
            a = L ** -gamma
            T = L * a * a / (1 + a * a)
            rows.append({"L": L, "T": T, "values": K.rntk_discrete(probe, np.full(L, a))})
    for r in rows:
        r["minus_quarter_fixed"] = r["values"][:3] - 0.25
        r["pair_mean"] = float(r["values"][3:].mean())
        r["values"] = None
    degeneration[str(gamma)] = rows
    print("gamma", gamma, "done", flush=True)
out["degeneration"] = {"fixed_angles": fixed, "h_fixed": K.first_order_term(fixed),
                       "h_pair_mean": float(K.first_order_term(pair_angles).mean()),
                       "by_gamma": degeneration}
run.finish(out)
