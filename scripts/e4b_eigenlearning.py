"""E4b: eigenlearning predictions of the excess risk of K_T regression on the sphere.

Uses the spectra of E3 (all T of the E4a grid) and the exact energy per degree
of the E4a targets; the spectrum beyond the computed degrees enters as a ridge
(its trace, 1 minus the computed part). Interpolation is ridge 0; the optimally
regularized risk (minimum over ridges) stands in for optimal early stopping.
"""
import argparse

import numpy as np

from effdepth import data, eigenlearning as EL
from effdepth.runio import Run, load

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--spectra", nargs="+", default=["out/runs/e3-spectrum-d3-grid", "out/runs/e3-spectrum-d5-grid"])
ap.add_argument("--targets", nargs="+", default=["linear", "harm1", "harm2"])
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.5])
args = ap.parse_args()

run = Run(args.run_dir, vars(args) | {"experiment": "e4b_eigenlearning"})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()

n_sim = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
n_dense = np.unique(np.round(np.geomspace(100, 1e6, 41)).astype(int))
ridges = np.concatenate([[0.0], np.geomspace(1e-4, 1e5, 91)])
out = {}
for path in args.spectra:
    spec = load(f"{path}/metrics.json")
    d = int(load(f"{path}/config.json")["d"])
    mult = np.array(spec["multiplicity"])
    kmax = len(mult) - 1
    energies = {}
    for name in args.targets:
        if name == "linear":
            energies[name] = data.LinearTarget(d).energy(kmax)
        else:
            tgt = data.HarmonicTarget(d, float(name[4:]), np.random.default_rng(777 + d))
            energies[name] = tgt.energy(kmax)
    res = {}
    ns = np.array(sorted(set(n_sim) | set(n_dense.tolist())), float)
    nn, rr = np.meshgrid(ns, ridges, indexing="ij")
    for T, mu in spec["mu"].items():
        mu = np.array(mu)
        tail = max(1.0 - float(np.sum(mult * mu)), 0.0)
        per = {}
        kappa = EL.solve_kappa_batch(mu, mult, nn, rr + tail)
        for name, en in energies.items():
            for s in args.sigmas:
                risk = EL.predict_batch(mu, mult, en, nn, s * s, rr, kappa=kappa).reshape(nn.shape)
                i = np.argmin(risk, axis=1)
                per[f"{name}:{s}"] = [{"n": int(ns[j]), "interp": float(risk[j, 0]),
                                       "optimal": float(risk[j, i[j]]), "ridge_opt": float(ridges[i[j]])}
                                      for j in range(len(ns))]
        res[T] = {"tail_mass": tail, "by_target": per}
        print(f"d={d} T={T}: tail mass {tail:.4f}", flush=True)
    out[str(d)] = res
run.finish({"predictions": out, "n_sim": n_sim})
