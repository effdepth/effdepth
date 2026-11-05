"""E5: the synthetic experiment of Yu et al. (2026, Sec. 4.2.1, Fig. 2), in its own setting.

Y = <X, (1,1,1)> + 0.1 eps, X uniform on S^2, 160 training and 40 test points;
kernel gradient descent with learning rate 1e-4 on the exact discrete RNTK at
depth L with alpha = L^-gamma. The update is f <- f - lr K (f - y) (no 1/n),
the convention under which the source's curves reach their plateaus within its
epoch budget. Besides the source's budgets (2500 epochs at L = 5000; 4500 epochs
across L), the run records long training, interpolation, the oracle and the
validation-selected stopping times, and the excess risk on 4000 fresh points.
"""
import effdepth  # noqa: F401
import argparse

import numpy as np
from scipy.interpolate import CubicSpline

from effdepth import data, kernels as K, krr
from effdepth.runio import Run

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--lr", type=float, default=1e-4)
args = ap.parse_args()

run = Run(args.run_dir, vars(args) | {"experiment": "e5_source_protocol"})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()

gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
depths = [2, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
grid = np.geomspace(1e-7, np.pi / 2, 6000)
splines = {}
for g in gammas:
    for L in depths:
        a = L ** -g
        splines[(g, L)] = (CubicSpline(np.log(grid), K.rntk_discrete(grid, np.full(L, a))),
                           K.effective_depth(np.full(L, a)))
    print("tables gamma", g, flush=True)


def gram(xa, xb, spline):
    d2 = np.clip(2 - 2 * xa @ xb.T, 0, 4)
    th = K.augmented_angle_from_dist2(d2)
    out = spline(np.log(np.maximum(th, grid[0])))
    out[th == 0] = 1.0
    return out


beta = np.ones(3)
epochs_a = np.arange(0, 2501, 100)
times_long = krr.time_grid(100, 1e-2, 1e12)
res = {f"{g}:{L}": {"T": splines[(g, L)][1], "fixed_4500": [], "curve_L5000": [], "src_test_4500": [],
                    "interp": [], "oracle": [], "val_selected": [], "long_1e6_epochs": []}
       for g in gammas for L in depths}
for seed in range(args.seeds):
    rng = np.random.default_rng(seed)
    x = data.sphere(200, 3, rng)
    y = x @ beta + 0.1 * rng.standard_normal(200)
    x_tr, y_tr, x_srcte, y_srcte = x[:160], y[:160], x[160:], y[160:]
    x_val = data.sphere(160, 3, rng)
    y_val = x_val @ beta + 0.1 * rng.standard_normal(160)
    x_te = data.sphere(4000, 3, rng)
    f_te = x_te @ beta
    n = 160
    for g in gammas:
        for L in depths:
            spl = splines[(g, L)][0]
            flow = krr.SpectralFlow(gram(x_tr, x_tr, spl))
            b_te = gram(x_te, x_tr, spl) @ flow.q
            b_src = gram(x_srcte, x_tr, spl) @ flow.q
            b_val = gram(x_val, x_tr, spl) @ flow.q
            r = res[f"{g}:{L}"]
            gd = lambda e: flow.gd_coefficients(y_tr, lr=args.lr * n, steps=int(e))
            c = gd(4500)
            r["fixed_4500"].append(float(np.mean((b_te @ c - f_te) ** 2)))
            r["src_test_4500"].append(float(np.mean((b_src @ c - y_srcte) ** 2)))
            if L == 5000:
                r["curve_L5000"].append([float(np.mean((b_te @ gd(e) - f_te) ** 2)) for e in epochs_a])
            c = gd(1e6)
            r["long_1e6_epochs"].append(float(np.mean((b_te @ c - f_te) ** 2)))
            risk = np.array([np.mean((b_te @ flow.coefficients(y_tr, t) - f_te) ** 2) for t in times_long])
            vmse = np.array([np.mean((b_val @ flow.coefficients(y_tr, t) - y_val) ** 2) for t in times_long])
            r["interp"].append(float(risk[-1]))
            r["oracle"].append(float(risk.min()))
            r["val_selected"].append(float(risk[np.argmin(vmse)]))
    print("seed", seed, "done", flush=True)

run.finish({"results": res, "epochs_curve": epochs_a, "gammas": gammas, "depths": depths,
            "note": "time t of the flow = lr * n * epochs; excess risk on 4000 fresh points"})
