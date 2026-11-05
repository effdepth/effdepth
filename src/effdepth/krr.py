"""Kernel gradient flow and interpolation through one eigendecomposition.

Gradient flow on the empirical squared loss (1/2n) sum (f(x_i) - y_i)^2 in the
RKHS, started from f = 0, gives at time t the predictor

    f_t(x) = k(x, X) K^{-1} (I - exp(-K t / n)) y,

which is the kernel regressor that a wide network trained by gradient flow
follows (Yu et al. 2026, eq. 2_NTK_GF). t = infinity is the minimum-norm
interpolant. With K = Q diag(lam) Q^T, every t costs one matrix product.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


class SpectralFlow:
    def __init__(self, gram: np.ndarray):
        self.n = gram.shape[0]
        # divide and conquer (evd) needs 1 + 6n + 2n^2 doubles of workspace, which
        # overflows LAPACK's 32-bit integers beyond n = 32766; evr needs O(n)
        driver = "evd" if 1 + 6 * self.n + 2 * self.n ** 2 < 2 ** 31 else "evr"
        lam, q = eigh(gram, driver=driver, check_finite=False)
        if not np.all(np.isfinite(lam)):
            raise FloatingPointError("non-finite eigenvalues")
        self.lam = np.maximum(lam, 0.0)
        self.q = q
        self.min_eig = float(lam[0])

    def coefficients(self, y: np.ndarray, t: float) -> np.ndarray:
        """Coefficients c_t in the eigenbasis: f_t(x) = k(x, X) Q c_t."""
        a = self.q.T @ y
        lam = self.lam
        if np.isinf(t):
            filt = np.where(lam > 0, 1.0 / np.where(lam > 0, lam, 1.0), 0.0)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                filt = np.where(lam > 0, -np.expm1(-lam * t / self.n) / lam, t / self.n)
        return filt.reshape((-1,) + (1,) * (a.ndim - 1)) * a

    def paths(self, gram_test: np.ndarray, y: np.ndarray, times) -> list[np.ndarray]:
        """Predictions on the test points at each time in `times` (inf allowed)."""
        b = gram_test @ self.q
        return [b @ self.coefficients(y, t) for t in times]

    def gd_coefficients(self, y: np.ndarray, lr: float, steps: int) -> np.ndarray:
        """Gradient descent with step lr on (1/2n)|K c - y|^2 in function space, `steps` times.

        f_{s+1}(X) = f_s(X) - (lr/n) K (f_s(X) - y): the eigencomponents decay as
        (1 - lr lam/n)^steps, the discrete-time version of the flow.
        """
        a = self.q.T @ y
        lam = self.lam
        r = 1 - lr * lam / self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            filt = np.where(lam > 0, (1 - r ** steps) / lam, lr * steps / self.n)
        return filt.reshape((-1,) + (1,) * (a.ndim - 1)) * a


def time_grid(n_points: int = 121, lo: float = 1e-3, hi: float = 1e12) -> np.ndarray:
    """Log-spaced stopping times, with infinity (interpolation) last."""
    return np.concatenate([np.geomspace(lo, hi, n_points), [np.inf]])
