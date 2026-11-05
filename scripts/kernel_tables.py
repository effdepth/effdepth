"""Tabulate kernels on a log grid of augmented angles, one .npz each, for the regression jobs.

--Ts        effective-depth kernels K_T            -> K_T<T>.npz
--discrete  exact discrete kernels at (L, alpha)   -> K_L<L>_a<alpha>.npz
"""
import argparse
from pathlib import Path

import numpy as np

from effdepth import kernels as K

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out/tables")
ap.add_argument("--Ts", type=float, nargs="*", default=[])
ap.add_argument("--discrete", nargs="*", default=[], help="pairs L:alpha")
ap.add_argument("--n", type=int, default=1 << 18)
ap.add_argument("--theta-min", type=float, default=1e-9)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
grid = np.geomspace(args.theta_min, np.pi / 2, args.n)


def save(path, vals, **meta):
    np.savez(path, grid=grid, values=vals, **meta)
    print(f"{path}: done (K at pi/2 = {vals[-1]:.6f})", flush=True)


for T in args.Ts:
    path = out / f"K_T{T:g}.npz"
    if path.exists():
        print(f"{path}: exists", flush=True)
        continue
    vals = K.kernel_one_layer(grid) if T == 0 else np.concatenate(
        [K.kernel_T(chunk, T) for chunk in np.array_split(grid, 8)])
    save(path, vals, T=T)

for spec in args.discrete:
    L, alpha = spec.split(":")
    L, alpha = int(L), float(alpha)
    path = out / f"K_L{L}_a{alpha:g}.npz"
    if path.exists():
        print(f"{path}: exists", flush=True)
        continue
    save(path, K.rntk_discrete(grid, np.full(L, alpha)), L=L, alpha=alpha,
         T=K.effective_depth(np.full(L, alpha)))
