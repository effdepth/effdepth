"""Data: uniform points on the sphere, regression targets, and image datasets."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .spectrum import legendre_sequence, multiplicity


def sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n points drawn uniformly on S^{d-1}, as rows."""
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class LinearTarget:
    """f(x) = <x, beta>: the synthetic target of Yu et al. (2026), beta = (1, ..., 1)."""

    def __init__(self, d: int):
        self.beta = np.ones(d)
        self.d = d

    def __call__(self, x):
        return x @ self.beta

    def energy(self, kmax: int) -> np.ndarray:
        """Energy per degree: everything in degree 1, E[<x,beta>^2] = |beta|^2/d."""
        e = np.zeros(kmax + 1)
        e[1] = self.beta @ self.beta / self.d
        return e


class HarmonicTarget:
    """Random target with energy E_k proportional to k^(-2s-1) in degree k = 1..kmax.

    The degree-k part is sum_m xi_m P_k(x . w_m) over `terms` random directions,
    rescaled so that its squared L2 norm is exactly E_k (addition theorem:
    E[P_k(x.a) P_k(x.b)] = P_k(a.b)/N(d,k)). The total energy is 1, and the
    target lies in the Sobolev space H^r for every r < s.
    """

    def __init__(self, d: int, s: float, rng: np.random.Generator, kmax: int = 300, terms: int = 8):
        self.d, self.s, self.kmax = d, s, kmax
        k = np.arange(1, kmax + 1)
        e = k ** (-2.0 * s - 1.0)
        self.e = e / e.sum()
        self.w = sphere(terms, d, rng)
        self.xi = rng.standard_normal((kmax, terms))
        gram_w = self.w @ self.w.T
        n_k = multiplicity(d, k)
        self.scale = np.empty(kmax)
        for kk, pk in enumerate(legendre_sequence(kmax, gram_w, d)):
            if kk == 0:
                continue
            xi = self.xi[kk - 1]
            norm2 = xi @ pk @ xi / n_k[kk - 1]
            self.scale[kk - 1] = np.sqrt(self.e[kk - 1] / norm2)

    def __call__(self, x, chunk: int = 4096):
        out = np.empty(len(x))
        for i in range(0, len(x), chunk):
            t = x[i:i + chunk] @ self.w.T                # (chunk, terms)
            acc = np.zeros(len(t))
            for kk, pk in enumerate(legendre_sequence(self.kmax, t, self.d)):
                if kk == 0:
                    continue
                acc += self.scale[kk - 1] * (pk @ self.xi[kk - 1])
            out[i:i + chunk] = acc
        return out

    def energy(self, kmax: int) -> np.ndarray:
        e = np.zeros(kmax + 1)
        m = min(kmax, self.kmax)
        e[1:m + 1] = self.e[:m]
        return e


# --- image datasets -----------------------------------------------------------

def data_root() -> Path:
    """Where datasets are stored: $EFFDEPTH_DATA, else ~/data/effdepth (outside the synced tree)."""
    return Path(os.environ.get("EFFDEPTH_DATA", Path.home() / "data" / "effdepth"))


def image_dataset(name: str, train: bool, download: bool = False):
    """(X, y) as float32 images in [0, 1] (N, C, H, W) and int64 labels, via torchvision."""
    import torchvision
    cls = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10,
           "cifar100": torchvision.datasets.CIFAR100}[name]
    ds = cls(root=str(data_root()), train=train, download=download)
    x = np.asarray(ds.data)
    if x.ndim == 3:                                      # MNIST: (N, 28, 28)
        x = x[:, None]
    else:                                                # CIFAR: (N, 32, 32, 3)
        x = x.transpose(0, 3, 1, 2)
    y = np.asarray(ds.targets, dtype=np.int64)
    return x.astype(np.float32) / 255.0, y


def sphere_features(x_train: np.ndarray, *others: np.ndarray):
    """Flatten, centre by the training mean, and project every row onto the unit sphere."""
    flat = [a.reshape(len(a), -1).astype(np.float64) for a in (x_train,) + others]
    mean = flat[0].mean(axis=0)
    out = []
    for a in flat:
        a = a - mean
        out.append(a / np.linalg.norm(a, axis=1, keepdims=True))
    return out
