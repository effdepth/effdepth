"""E4c: kernel regression with K_T on MNIST and CIFAR-10 (fully connected kernel).

Inputs are centred by the training mean and projected onto the sphere; targets
are centred one-hot vectors; the predicted class is the argmax. For each kernel
(a table name, e.g. K_T4 or K_L50_a0.02), accuracy and MSE are recorded along the
gradient-flow time grid, for interpolation, the oracle time, and the time
selected on a validation set drawn from the unused training images.
"""
import effdepth  # noqa: F401  (sets BLAS threads before numpy loads)
import argparse
from pathlib import Path

import numpy as np

from effdepth import data, kernels as K, krr
from effdepth.runio import Run

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--n-val", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kernels", nargs="+", required=True, help="table names in --tables")
ap.add_argument("--tables", default="out/tables")
ap.add_argument("--block", type=int, default=2048)
args = ap.parse_args()

run = Run(args.run_dir, vars(args) | {"experiment": "e4_real"})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()

x_all, y_all = data.image_dataset(args.dataset, train=True)
x_test, y_test = data.image_dataset(args.dataset, train=False)
rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(x_all))
n_val = min(args.n_val, len(x_all) - args.n)
tr, va = perm[:args.n], perm[args.n:args.n + n_val]
f_tr, f_va, f_te = data.sphere_features(x_all[tr], x_all[va], x_test)
del x_all, x_test
classes = int(y_all.max()) + 1
onehot = lambda y: np.eye(classes)[y] - 1.0 / classes
y_tr, y_va, y_te = y_all[tr], y_all[va], y_test


def gram(xa, xb, name):
    if name == "K_T0":
        tab = None
    else:
        with np.load(Path(args.tables) / f"{name}.npz") as tab:
            grid, vals = tab["grid"], tab["values"]
        lg, lo = np.log(grid), grid[0]
    out = np.empty((len(xa), len(xb)))
    for i in range(0, len(xa), args.block):
        d2 = np.clip(2 - 2 * xa[i:i + args.block] @ xb.T, 0, 4)
        th = K.augmented_angle_from_dist2(d2)
        if tab is None:
            out[i:i + args.block] = K.kernel_one_layer(th)
        else:
            blk = np.interp(np.log(np.maximum(th, lo)), lg, vals)
            blk[th == 0] = 1.0
            out[i:i + args.block] = blk
    return out


times = krr.time_grid(140, 1e-2, 1e15)
keep = np.unique(np.concatenate([np.arange(0, len(times), 4), [len(times) - 1]]))
results = {}
for name in args.kernels:
    flow = krr.SpectralFlow(gram(f_tr, f_tr, name))
    b_te = gram(f_te, f_tr, name) @ flow.q
    b_va = gram(f_va, f_tr, name) @ flow.q
    acc, mse, vacc = [], [], []
    Y = onehot(y_tr)
    for t in times:
        c = flow.coefficients(Y, t)
        p_te, p_va = b_te @ c, b_va @ c
        acc.append(float(np.mean(p_te.argmax(1) == y_te)))
        mse.append(float(np.mean(np.sum((p_te - onehot(y_te)) ** 2, axis=1))))
        vacc.append(float(np.mean(p_va.argmax(1) == y_va)))
    acc, mse, vacc = map(np.array, (acc, mse, vacc))
    io, iv = int(np.argmax(acc)), int(np.argmax(vacc))
    meta = {} if name == "K_T0" else {k: float(v) for k, v in np.load(
        Path(args.tables) / f"{name}.npz").items() if k in ("T", "L", "alpha")}
    results[name] = {**meta, "interp_acc": acc[-1], "interp_mse": mse[-1],
                     "oracle_acc": acc[io], "t_oracle": times[io],
                     "val_acc": acc[iv], "t_val": times[iv], "val_mse": mse[iv],
                     "min_eig": flow.min_eig, "acc_curve": acc[keep], "mse_curve": mse[keep]}
    print(f"{name}: interp {acc[-1]:.4f}  oracle {acc[io]:.4f}  val-selected {acc[iv]:.4f}", flush=True)
    del flow, b_te, b_va

run.finish({"results": results, "times": times[keep]})
