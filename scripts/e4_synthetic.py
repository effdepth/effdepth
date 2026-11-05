"""E4a: kernel gradient flow with K_T on the sphere, across effective depths T.

One unit of work is (d, n, seed): the same inputs serve every T, target and noise
level. For each, the excess risk (test MSE against the target) is recorded along
the gradient-flow time grid, with interpolation (t = inf), the oracle stopping
time and the time selected on a held-out validation set.
"""
import effdepth  # noqa: F401  (sets BLAS threads before numpy loads)
import argparse
from pathlib import Path

import numpy as np

from effdepth import data, kernels as K, krr
from effdepth.runio import Run

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--Ts", type=float, nargs="+", required=True)
ap.add_argument("--tables", default="out/tables")
ap.add_argument("--targets", nargs="+", default=["linear", "harm1", "harm2"])
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.5])
ap.add_argument("--n-test", type=int, default=4000)
ap.add_argument("--block", type=int, default=2048)
args = ap.parse_args()

run = Run(args.run_dir, vars(args) | {"experiment": "e4_synthetic"})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()

rng = np.random.default_rng(1000 * args.d + args.seed)
n_val = max(args.n // 4, 100)
x_tr = data.sphere(args.n, args.d, rng)
x_val = data.sphere(n_val, args.d, rng)
x_te = data.sphere(args.n_test, args.d, rng)

targets = {}
for name in args.targets:
    if name == "linear":
        targets[name] = data.LinearTarget(args.d)
    else:  # harm<s>: fixed function, independent of the data seed
        s = float(name[4:])
        targets[name] = data.HarmonicTarget(args.d, s, np.random.default_rng(777 + args.d))
f = {name: (t(x_tr), t(x_val), t(x_te)) for name, t in targets.items()}
noise_tr = rng.standard_normal(args.n)
noise_val = rng.standard_normal(n_val)


def gram(xa, xb, T):
    """K_T between rows of xa and xb, in blocks, through the tabulated kernel."""
    if T > 0:
        with np.load(Path(args.tables) / f"K_T{T:g}.npz") as tab:
            grid, vals = tab["grid"], tab["values"]
        lg, lo = np.log(grid), grid[0]
    out = np.empty((len(xa), len(xb)))
    for i in range(0, len(xa), args.block):
        d2 = np.clip(2 - 2 * xa[i:i + args.block] @ xb.T, 0, 4)
        th = K.augmented_angle_from_dist2(d2)
        if T == 0:
            out[i:i + args.block] = K.kernel_one_layer(th)
        else:
            blk = np.interp(np.log(np.maximum(th, lo)), lg, vals)
            blk[th == 0] = 1.0
            out[i:i + args.block] = blk
    return out


times = krr.time_grid(140, 1e-2, 1e15)
keep = np.arange(0, len(times), 4)            # stored curve: every 4th time and inf
keep = np.unique(np.concatenate([keep, [len(times) - 1]]))
results = {name: {str(s): {} for s in args.sigmas} for name in targets}
diag = {}
for T in args.Ts:
    k_tr = gram(x_tr, x_tr, T)
    flow = krr.SpectralFlow(k_tr)
    del k_tr
    b_te = gram(x_te, x_tr, T) @ flow.q
    b_val = gram(x_val, x_tr, T) @ flow.q
    diag[str(T)] = {"min_eig": flow.min_eig, "max_eig": float(flow.lam[-1]),
                    "eig_quantiles": np.quantile(flow.lam, [0.001, 0.01, 0.1, 0.5]).tolist()}
    for name, (ftr, fval, fte) in f.items():
        for s in args.sigmas:
            y = ftr + s * noise_tr
            yval = fval + s * noise_val
            risk, vmse = [], []
            for t in times:
                c = flow.coefficients(y, t)
                risk.append(float(np.mean((b_te @ c - fte) ** 2)))
                vmse.append(float(np.mean((b_val @ c - yval) ** 2)))
            risk, vmse = np.array(risk), np.array(vmse)
            io, iv = int(np.argmin(risk)), int(np.argmin(vmse))
            results[name][str(s)][str(T)] = {
                "interp": risk[-1], "oracle": risk[io], "t_oracle": times[io],
                "val_selected": risk[iv], "t_val": times[iv],
                "curve": risk[keep]}
    print(f"T={T:g}: min eig {flow.min_eig:.3e}; linear/0.1 interp "
          f"{results.get('linear', {}).get('0.1', {}).get(str(T), {}).get('interp', float('nan')):.4f}",
          flush=True)
    del flow, b_te, b_val

run.finish({"results": results, "times": times[keep], "diagnostics": diag,
            "target_variance": {name: float(np.mean(v[2] ** 2)) for name, v in f.items()},
            "target_energy": {name: t.energy(400) for name, t in targets.items()}})
