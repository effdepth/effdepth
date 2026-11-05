"""Neural tangent kernel of a wide fully connected ResNet and its effective depth.

The network is the one of Huang et al. (2020), Belfer et al. (2024) and Yu et al.
(2026): x^(0) = (A x + b)/sqrt(m),
x^(l) = x^(l-1) + alpha_l V^(l) relu(sqrt(2/m) W^(l) x^(l-1)) / sqrt(m),
f = v^T x^(L), with W, V trained. Its infinite-width NTK, divided by its value on
the diagonal, depends on two inputs only through the angle theta between their
augmented versions (x, 1)/|(x, 1)|. For x, x' on the unit sphere,
cos(theta) = (1 + x.x')/2, so theta lies in [0, pi/2].

Writing eta_l = alpha_l^2 / (1 + alpha_l^2), the correlation recursion of the NTK
is an explicit Euler step of size eta_l for the ODE  u' = kappa_1(u) - u, and the
normalized NTK is close to a kernel K_T that depends on the branch scales only
through the effective depth T = sum_l eta_l (`kernel_T`).

All functions act elementwise on arrays of angles.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

PI = np.pi


def kappa0(theta):
    """kappa_0(cos theta) = (pi - theta)/pi."""
    return 1.0 - np.asarray(theta, float) / PI


def kappa1(theta):
    """kappa_1(cos theta) = ((pi - theta) cos theta + sin theta)/pi."""
    t = np.asarray(theta, float)
    return ((PI - t) * np.cos(t) + np.sin(t)) / PI


def gap(theta):
    """kappa_1(u) - u at u = cos theta, i.e. (sin t - t cos t)/pi, accurate at small t."""
    t = np.asarray(theta, float)
    t2 = t * t
    series = t2 * t * (1 / 3 - t2 * (1 / 30 - t2 * (1 / 840 - t2 * (1 / 45360 - t2 / 3991680)))) / PI
    direct = (np.sin(t) - t * np.cos(t)) / PI
    return np.where(t < 0.1, series, direct)


def speed(theta):
    """H(theta) = gap/sin theta, so that the ODE in the angle reads theta' = -H(theta)."""
    t = np.asarray(theta, float)
    t2 = t * t
    series = t2 * (1 / 3 + t2 * (1 / 45 + t2 * (2 / 945 + t2 / 4725))) / PI
    with np.errstate(invalid="ignore", divide="ignore"):
        direct = gap(t) / np.sin(t)
    return np.where(t < 0.05, series, direct)


def phi0(theta):
    """u kappa_0(u) + kappa_1(u) at u = cos theta; equals 2 at theta = 0."""
    t = np.asarray(theta, float)
    return (2 * np.cos(t) * (PI - t) + np.sin(t)) / PI


def angle_from_e(e):
    """theta from e = 1 - cos theta, without cancellation."""
    return 2 * np.arcsin(np.sqrt(np.clip(e, 0.0, 2.0) / 2))


def augmented_angle(cos_xx):
    """Angle between (x, 1)/sqrt(2) and (x', 1)/sqrt(2) for unit x, x' with x.x' = cos_xx."""
    c = np.clip(np.asarray(cos_xx, float), -1.0, 1.0)
    # 1 - cos(theta~) = (1 - c)/2 and sin(theta~/2) = sqrt((1 - c)/4)
    return 2 * np.arcsin(np.sqrt((1 - c) / 4))


def augmented_angle_from_dist2(dist2):
    """Same as `augmented_angle`, from the squared distance |x - x'|^2 = 2 - 2 x.x'."""
    d2 = np.clip(np.asarray(dist2, float), 0.0, 4.0)
    return 2 * np.arcsin(np.sqrt(d2 / 8))


def effective_depth(alphas):
    """T = sum_l alpha_l^2 / (1 + alpha_l^2)."""
    a2 = np.asarray(alphas, float) ** 2
    return float(np.sum(a2 / (1 + a2)))


def alpha_for_T(T, L):
    """Constant branch scale that gives effective depth T at depth L (needs T < L)."""
    if not 0 < T < L:
        raise ValueError("need 0 < T < L")
    return float(np.sqrt(T / (L - T)))


def rntk_discrete(theta, alphas):
    """Normalized infinite-width NTK (value 1 on the diagonal) for branch scales `alphas`.

    Exact recursion of Yu et al. (2026, appendix C, eq. NormRNTK), generalized to
    layer-dependent scales, iterated in e = 1 - u so that it stays accurate when u
    is within 1e-12 of 1 (constant alpha at depth 1e7).
    """
    th = np.array(theta, dtype=float, copy=True)
    a2 = np.asarray(alphas, float) ** 2
    eta = a2 / (1 + a2)
    T = float(eta.sum())
    e = 2 * np.sin(th / 2) ** 2
    log_q = np.zeros_like(th)
    acc = np.zeros_like(th)
    for i, et in enumerate(eta):
        if i:
            # backward factor of block i+1: 1 - eta (1 - kappa_0(u)) = 1 - eta theta/pi
            log_q += np.log1p(-et * th / PI)
        acc += et * phi0(th) * np.exp(-log_q)
        e = e - et * gap(th)
        th = angle_from_e(e)
    return np.exp(log_q) * acc / (2 * T)


def kernel_one_layer(theta):
    """K_0: the one-hidden-layer kernel, the limit T -> 0."""
    return phi0(theta) / 2


def kernel_T(theta, T, rtol=1e-11):
    """Effective-depth kernel K_T at the given angles.

    K_T = (1/2T) int_0^T exp(-int_s^T theta(r)/pi dr) phi0(theta(s)) ds, where
    theta(s) solves theta' = -H(theta) from the given angle. Integrated per angle
    in the forward form K_T = exp(-B(T)) A(T)/(2T), B' = theta/pi, A' = e^B phi0.
    """
    th0 = np.atleast_1d(np.asarray(theta, float)).ravel()
    if T == 0:
        return kernel_one_layer(th0).reshape(np.shape(theta))
    n = th0.size

    def rhs(_, y):
        th, b = y[:n], y[n:2 * n]
        return np.concatenate([-speed(th), th / PI, np.exp(b) * phi0(th)])

    y0 = np.concatenate([th0, np.zeros(n), np.zeros(n)])
    sol = solve_ivp(rhs, (0.0, float(T)), y0, method="DOP853", rtol=rtol, atol=1e-14,
                    t_eval=[float(T)])
    if not sol.success:
        raise RuntimeError(sol.message)
    _, b, a = np.split(sol.y[:, -1], 3)
    return (np.exp(-b) * a / (2 * T)).reshape(np.shape(theta))


# --- master trajectory: phase, first-order term and near-diagonal profile -----

def _remainder(phi):
    """1/H(phi) - 3 pi/phi^2, smooth on [0, pi/2] (value -pi/5 at 0)."""
    p = np.asarray(phi, float)
    p2 = p * p
    # 1/H = (3 pi/p^2)(1 + p^2/15 + 2p^4/315 + ...)^(-1) expanded
    series = PI * (-1 / 5 - p2 * (1 / 175 + p2 * (2 / 7875 + p2 * 37 / 3031875)))
    with np.errstate(invalid="ignore", divide="ignore"):
        direct = 1.0 / speed(p) - 3 * PI / p2
    return np.where(p < 0.05, series, direct)


_PANELS = np.linspace(0.0, PI / 2, 257)
_GX, _GW = np.polynomial.legendre.leggauss(24)


def _panel_integrals():
    a, b = _PANELS[:-1], _PANELS[1:]
    x = (a + b)[:, None] / 2 + (b - a)[:, None] / 2 * _GX
    return ((b - a) / 2) * (_remainder(x) @ _GW)


_TAIL = np.concatenate([np.cumsum(_panel_integrals()[::-1])[::-1], [0.0]])  # int_{edge_j}^{pi/2}


def _remainder_integral(theta):
    """int_theta^{pi/2} R(phi) dphi, by Gauss-Legendre on fixed panels (R is smooth)."""
    th = np.clip(np.atleast_1d(np.asarray(theta, float)), 0.0, PI / 2)
    j = np.clip(np.searchsorted(_PANELS, th, side="right") - 1, 0, len(_PANELS) - 2)
    a, b = th, _PANELS[j + 1]
    x = (a + b)[:, None] / 2 + (b - a)[:, None] / 2 * _GX
    partial = ((b - a) / 2) * (_remainder(x) @ _GW)
    return partial + _TAIL[j + 1]


C_INF = float(6.0 - _remainder_integral(0.0)[0])
"""lim_{theta->0} (3 pi/theta - tau(theta)): the asymptotic phase of the master trajectory."""


def phase(theta):
    """tau(theta): time the ODE takes from pi/2 to theta (infinite at theta = 0)."""
    th = np.asarray(theta, float)
    with np.errstate(divide="ignore"):
        return 3 * PI * (1 / th - 2 / PI) + _remainder_integral(th).reshape(th.shape)


def first_order_term(theta):
    """h(theta) such that K_T(theta) = 1/4 + h(theta)/T + O(T^-2) as T -> infinity."""
    return (phase(theta) + C_INF - 2) / 4


def profile(z):
    """F(z) = ((1+z)^4 - z^4)/(4(1+z)^3): K_T(theta) ~ F(3 pi/(T theta)) near the diagonal."""
    z = np.asarray(z, float)
    return (((4 * z + 6) * z + 4) * z + 1) / (4 * (1 + z) ** 3)


def kernel_T_large_phase(theta, T):
    """K_T from the asymptotic master trajectory theta(s) = 3 pi/(s + tau + C_INF).

    Accurate when the phase tau(theta) is large (theta small), for any T > 0:
    K_T = F(z) - (1 - (z/(1+z))^3)/(2T), z = (tau(theta) + C_INF)/T.
    """
    z = (phase(theta) + C_INF) / T
    return profile(z) - ((3 * z + 3) * z + 1) / (1 + z) ** 3 / (2 * T)


class KernelTable:
    """K_T on a log-spaced grid of angles, for fast evaluation on Gram matrices.

    Cubic interpolation in log(theta) over [theta_min, pi/2]; angles below
    theta_min (in practice only the diagonal) take the value at theta_min, which
    differs from 1 by O(T theta_min).
    """

    def __init__(self, T, n=1 << 15, theta_min=1e-9):
        from scipy.interpolate import CubicSpline
        self.T = float(T)
        self.grid = np.geomspace(theta_min, PI / 2, n)
        self.values = kernel_T(self.grid, self.T)
        self._spline = CubicSpline(np.log(self.grid), self.values)
        self.theta_min = theta_min

    def __call__(self, theta):
        th = np.asarray(theta, float)
        out = np.ones_like(th)
        m = th > 0
        if self.T == 0:
            out[m] = kernel_one_layer(th[m])
            return out
        tm = np.maximum(th[m], self.theta_min)
        out[m] = self._spline(np.log(tm))
        return out
