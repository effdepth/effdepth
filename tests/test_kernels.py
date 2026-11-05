import numpy as np
import pytest

from effdepth import kernels as K

ANGLES = np.array([1e-7, 1e-3, 0.05, 0.3, 0.8, 1.2, np.pi / 2])


def naive_rntk(rho, L, alpha):
    """The recursion of Yu et al. (2026, eq. NormRNTK) as written, constant alpha."""
    k0 = lambda u: (np.pi - np.arccos(u)) / np.pi
    k1 = lambda u: (u * (np.pi - np.arccos(u)) + np.sqrt(np.clip(1 - u * u, 0, None))) / np.pi
    a2 = alpha ** 2
    u = [np.asarray(rho, float)]
    for _ in range(L):
        u.append(np.clip((u[-1] + a2 * k1(u[-1])) / (1 + a2), -1, 1))
    total = 0.0
    for l in range(1, L + 1):
        p = np.ones_like(u[0])
        for i in range(l, L):
            p = p * (1 + a2 * k0(u[i])) / (1 + a2)
        total = total + p * (k1(u[l - 1]) + u[l - 1] * k0(u[l - 1]))
    return total / (2 * L)


@pytest.mark.parametrize("alpha,L", [(1.0, 1), (1.0, 6), (0.3, 20), (2.0, 4)])
def test_discrete_matches_source_formula(alpha, L):
    th = ANGLES[2:]
    np.testing.assert_allclose(K.rntk_discrete(th, np.full(L, alpha)),
                               naive_rntk(np.cos(th), L, alpha), atol=1e-10)


def test_diagonal_is_one_and_values_in_range():
    th = np.array([0.0])
    assert K.rntk_discrete(th, np.full(10, 0.7))[0] == pytest.approx(1.0)
    for T in (0.0, 0.3, 5.0, 300.0):
        v = K.kernel_T(ANGLES, T)
        assert np.all(v <= 1 + 1e-12) and np.all(v > 0)
        assert K.kernel_T(np.array([1e-12]), T)[0] == pytest.approx(1.0, abs=1e-8)


def test_one_layer_limit():
    k = K.kernel_T(ANGLES, 1e-6)
    np.testing.assert_allclose(k, K.kernel_one_layer(ANGLES), atol=1e-5)
    # depth 1 with small alpha is the one-layer kernel exactly
    np.testing.assert_allclose(K.rntk_discrete(ANGLES, [1e-4]), K.kernel_one_layer(ANGLES), atol=1e-7)


@pytest.mark.parametrize("T", [0.5, 4.0])
def test_effective_depth_collapse(T):
    """The discrete kernel approaches K_T at rate O(eta) at fixed T."""
    ref = K.kernel_T(ANGLES, T)
    gaps = []
    for eta in (0.2, 0.05, 0.0125):
        L = int(round(T / eta))
        alpha = np.sqrt(eta / (1 - eta))
        gaps.append(np.abs(K.rntk_discrete(ANGLES, np.full(L, alpha)) - ref).max())
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 0.1 * 0.0125  # gap / eta below 0.1


def test_layer_dependent_scales_collapse():
    rng = np.random.default_rng(0)
    eta = rng.uniform(0.0, 0.01, size=400)
    alphas = np.sqrt(eta / (1 - eta))
    T = K.effective_depth(alphas)
    assert T == pytest.approx(eta.sum())
    gap = np.abs(K.rntk_discrete(ANGLES, alphas) - K.kernel_T(ANGLES, T)).max()
    assert gap < 0.1 * eta.max()


def test_first_order_term_and_large_phase_form():
    th = np.array([0.05, 0.3, 1.0, np.pi / 2])
    T = 2e3
    np.testing.assert_allclose(T * (K.kernel_T(th, T) - 0.25), K.first_order_term(th), atol=0.06)
    small = np.array([1e-6, 1e-4])
    np.testing.assert_allclose(K.kernel_T(small, 50.0), K.kernel_T_large_phase(small, 50.0), atol=1e-8)


def test_phase_asymptotics_and_constant():
    assert K.C_INF == pytest.approx(7.0118218, abs=1e-6)
    assert K.phase(np.array([np.pi / 2]))[0] == pytest.approx(0.0, abs=1e-12)
    th = 1e-4    # 3 pi/theta - tau(theta) = C_INF + int_0^theta R, and R(0) = -pi/5
    assert 3 * np.pi / th - K.phase(np.array([th]))[0] == pytest.approx(K.C_INF - np.pi * th / 5, abs=1e-9)


def test_profile_limits():
    assert K.profile(np.array([0.0]))[0] == pytest.approx(0.25)
    assert K.profile(np.array([1e8]))[0] == pytest.approx(1.0, abs=1e-7)


def test_alpha_for_T_roundtrip():
    a = K.alpha_for_T(2.0, 16)
    assert K.effective_depth(np.full(16, a)) == pytest.approx(2.0)


def test_kernel_table_interpolation():
    tab = K.KernelTable(8.0, n=4096)
    th = np.array([0.0, 1e-6, 0.01, 0.4, 1.5])
    exact = K.kernel_T(np.maximum(th, 1e-12), 8.0)
    np.testing.assert_allclose(tab(th), exact, atol=1e-7)
