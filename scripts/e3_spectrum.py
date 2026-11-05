"""E3: Mercer eigenvalues mu_k(T) of K_T on S^{d-1}, and those of the first-order term h."""
import argparse

import numpy as np

from effdepth import kernels as K
from effdepth import spectrum as S
from effdepth.runio import Run

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--kmax", type=int, default=2000)
ap.add_argument("--Ts", type=float, nargs="+", default=[0, 0.25, 1, 4, 16, 64, 256, 1024])
args = ap.parse_args()

run = Run(args.run_dir, vars(args) | {"experiment": "e3_spectrum"})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()

k = np.arange(args.kmax + 1)
mult = S.multiplicity(args.d, k)
mus = {}
for T in args.Ts:
    mu = S.eigenvalues(S.rntk_of_sphere_angle(T), args.d, args.kmax)
    mus[str(T)] = mu
    print(f"T={T}: trace up to kmax = {np.sum(mult * mu):.6f}, mu_1 = {mu[1]:.4e}", flush=True)


def h_of_sphere_angle(theta):
    aug = 2 * np.arcsin(np.sin(np.asarray(theta, float) / 2) / np.sqrt(2))
    return K.first_order_term(aug)


mu_h = S.eigenvalues(h_of_sphere_angle, args.d, args.kmax)
run.finish({"k": k, "multiplicity": mult, "mu": mus, "mu_h": mu_h,
            "trace": {T: float(np.sum(mult * mu)) for T, mu in mus.items()}})
