"""Learning-curve predictions for kernel ridge(less) regression from the kernel spectrum.

The omniscient risk estimate of Jacot et al. (2020), Bordelon et al. (2020),
Canatar et al. (2021) and Simon et al. (2023): with population eigenvalues
lambda_i (here mu_k, repeated N(d, k) times), n samples and a ridge delta on the
Gram matrix (K + delta I), the effective regularization kappa solves

    sum_i lambda_i / (lambda_i + kappa) + delta / kappa = n,

the learnabilities are L_i = lambda_i / (lambda_i + kappa), and the excess risk
(mean squared error against the target) is

    E0 * (sum_i (1 - L_i)^2 v_i^2 + sigma^2) - sigma^2,   E0 = n / (n - sum_i L_i^2),

with v_i^2 the target's energy along eigenfunction i. Eigenvalues are grouped by
degree: `eig[k]`, `mult[k]`, and `energy[k]` is the target energy of degree k.
"""
from __future__ import annotations

import numpy as np


def solve_kappa(eig, mult, n, delta=0.0):
    """Effective regularization kappa (> 0) for n samples and Gram ridge delta."""
    eig = np.asarray(eig, float)
    mult = np.asarray(mult, float)

    def excess(log_k):
        k = np.exp(log_k)
        return np.sum(mult * eig / (eig + k)) + delta / k - n

    lo, hi = np.log(1e-300), np.log(max(eig.max(), delta, 1e-300) * 1e6 + 1.0)
    if excess(lo) < 0:          # fewer effective dimensions than samples and no ridge
        raise ValueError("the spectrum has fewer than n directions and delta = 0")
    for _ in range(200):
        mid = (lo + hi) / 2
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(np.exp((lo + hi) / 2))


def predict(eig, mult, energy, n, noise_var, delta=0.0, tail_mass=0.0):
    """Predicted excess risk of KRR with Gram ridge delta (0: interpolation).

    `tail_mass` is the trace of the spectrum beyond the computed degrees; those
    eigenvalues are far below kappa and act as an additional ridge.
    """
    eig = np.asarray(eig, float)
    mult = np.asarray(mult, float)
    energy = np.asarray(energy, float)
    kappa = solve_kappa(eig, mult, n, delta + tail_mass)
    learn = eig / (eig + kappa)
    e0 = n / (n - np.sum(mult * learn ** 2))
    bias = np.sum(energy * (1 - learn) ** 2)
    return float(e0 * (bias + noise_var) - noise_var)


def predict_optimal_ridge(eig, mult, energy, n, noise_var, tail_mass=0.0, grid=None):
    """Smallest predicted excess risk over a grid of ridges, and the ridge reaching it."""
    if grid is None:
        grid = np.concatenate([[0.0], np.geomspace(1e-6, 1e4, 121)])
    risks = [predict(eig, mult, energy, n, noise_var, delta=g, tail_mass=tail_mass) for g in grid]
    i = int(np.argmin(risks))
    return float(risks[i]), float(grid[i])


def solve_kappa_batch(eig, mult, n, delta, iters: int = 64):
    """Vectorized `solve_kappa` over arrays n and delta of the same shape."""
    eig = np.asarray(eig, float)[None, :]
    mult = np.asarray(mult, float)[None, :]
    n = np.asarray(n, float).ravel()
    delta = np.asarray(delta, float).ravel()
    lo = np.full(n.shape, np.log(1e-300))
    hi = np.full(n.shape, np.log(max(eig.max(), delta.max(), 1e-300) * 1e6 + 1.0))
    for _ in range(iters):
        mid = (lo + hi) / 2
        k = np.exp(mid)
        f = np.sum(mult * eig / (eig + k[:, None]), axis=1) + delta / k - n
        lo = np.where(f > 0, mid, lo)
        hi = np.where(f > 0, hi, mid)
    return np.exp((lo + hi) / 2)


def predict_batch(eig, mult, energy, n, noise_var, delta, tail_mass=0.0, kappa=None):
    """Vectorized `predict` over arrays n and delta (same shape); `kappa` may be precomputed."""
    eig = np.asarray(eig, float)
    mult = np.asarray(mult, float)
    energy = np.asarray(energy, float)
    n = np.asarray(n, float).ravel()
    delta = np.asarray(delta, float).ravel()
    if kappa is None:
        kappa = solve_kappa_batch(eig, mult, n, delta + tail_mass)
    kappa = np.asarray(kappa, float).ravel()[:, None]
    learn = eig[None, :] / (eig[None, :] + kappa)
    e0 = n / (n - np.sum(mult[None, :] * learn ** 2, axis=1))
    bias = np.sum(energy[None, :] * (1 - learn) ** 2, axis=1)
    return e0 * (bias + noise_var) - noise_var
