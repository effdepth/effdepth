"""Mercer eigenvalues of dot-product kernels on the sphere S^{d-1} (Funk-Hecke).

For a kernel k(x.x') on S^{d-1} with the uniform probability measure,
k(x.x') = sum_k mu_k sum_j Y_kj(x) Y_kj(x'), with Y_kj orthonormal, and
mu_k = E[k(t) P_k(t)], where t = x.x' for x' uniform and P_k is the Legendre
polynomial of degree k in dimension d (P_k(1) = 1). Each mu_k has multiplicity
N(d, k), and sum_k N(d, k) mu_k = k(1).
"""
from __future__ import annotations

from math import lgamma, log, exp, pi

import numpy as np

from . import kernels


def multiplicity(d: int, k):
    """Number N(d, k) of linearly independent spherical harmonics of degree k on S^{d-1}."""
    from scipy.special import gammaln
    k = np.asarray(k, dtype=float)
    if d == 2:
        return np.where(k == 0, 1.0, 2.0)
    # N = (2k + d - 2) Gamma(k + d - 2) / (Gamma(d - 1) Gamma(k + 1)) for k >= 1
    kk = np.maximum(k, 1.0)
    n = (2 * kk + d - 2) * np.exp(gammaln(kk + d - 2) - gammaln(d - 1) - gammaln(kk + 1))
    return np.where(k == 0, 1.0, n)


def legendre_sequence(kmax: int, t, d: int):
    """Yield P_0(t), ..., P_kmax(t), the Legendre polynomials of dimension d."""
    t = np.asarray(t, float)
    p0 = np.ones_like(t)
    yield p0
    if kmax == 0:
        return
    p1 = t.copy()
    yield p1
    for k in range(1, kmax):
        p2 = ((2 * k + d - 2) * t * p1 - k * p0) / (k + d - 2)
        yield p2
        p0, p1 = p1, p2


def _weight_constant(d: int) -> float:
    """1 / int_0^pi sin^{d-2}(theta) d theta."""
    return exp(lgamma(d / 2) - 0.5 * log(pi) - lgamma((d - 1) / 2))


def quadrature(d: int, kmax: int, nodes_per_panel: int = 24, near_zero: float = 1e-8):
    """Nodes theta in [0, pi] and weights for E[g(theta)] under t = cos(theta) ~ tau_d.

    Uniform panels short enough to resolve P_kmax (about 12 nodes per period),
    plus geometrically graded panels towards theta = 0, where the kernels of this
    project have a cusp of width ~1/T.
    """
    gx, gw = np.polynomial.legendre.leggauss(nodes_per_panel)
    width = min(np.pi / 64, 2 * np.pi / max(kmax, 1))
    m = int(np.ceil(np.pi / width))
    edges = list(np.linspace(0.0, np.pi, m + 1))
    first = edges[1]
    graded = list(np.geomspace(near_zero, first, 40))
    edges = [0.0] + graded + edges[2:]
    edges = np.array(edges)
    a, b = edges[:-1], edges[1:]
    x = ((a + b)[:, None] / 2 + (b - a)[:, None] / 2 * gx).ravel()
    w = (((b - a)[:, None] / 2) * gw).ravel()
    w = w * np.sin(x) ** (d - 2) * _weight_constant(d)
    return x, w


def eigenvalues(kernel_of_t, d: int, kmax: int, **quad):
    """mu_0, ..., mu_kmax of the dot-product kernel given as a function of theta = arccos(x.x')."""
    x, w = quadrature(d, kmax, **quad)
    wk = w * kernel_of_t(x)
    t = np.cos(x)
    mu = np.empty(kmax + 1)
    for k, pk in enumerate(legendre_sequence(kmax, t, d)):
        mu[k] = np.dot(wk, pk)
    return mu


def rntk_of_sphere_angle(T: float):
    """K_T as a function of the angle between x and x' on the sphere (not the augmented angle)."""
    def f(theta):
        aug = 2 * np.arcsin(np.sin(np.asarray(theta, float) / 2) / np.sqrt(2))
        return kernels.kernel_T(aug, T)
    return f
