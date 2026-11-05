import numpy as np
import pytest

from effdepth import data, eigenlearning, krr, spectrum


def test_multiplicities():
    np.testing.assert_allclose(spectrum.multiplicity(3, np.arange(5)), [1, 3, 5, 7, 9])
    np.testing.assert_allclose(spectrum.multiplicity(4, np.arange(4)), [1, 4, 9, 16])


def test_funk_hecke_on_known_kernels():
    # k(t) = t^2 on S^2: mu_0 = E[t^2] = 1/3, mu_2 = 2/15, others 0
    mu = spectrum.eigenvalues(lambda th: np.cos(th) ** 2, d=3, kmax=4)
    np.testing.assert_allclose(mu, [1 / 3, 0, 2 / 15, 0, 0], atol=1e-12)
    # trace: sum_k N(d,k) mu_k = k(1)
    f = lambda th: np.exp(np.cos(th))
    for d in (3, 5):
        mu = spectrum.eigenvalues(f, d=d, kmax=40)
        assert np.sum(spectrum.multiplicity(d, np.arange(41)) * mu) == pytest.approx(np.e, rel=1e-10)


def test_rntk_spectrum_sums_to_one():
    mu = spectrum.eigenvalues(spectrum.rntk_of_sphere_angle(4.0), d=3, kmax=400)
    total = np.sum(spectrum.multiplicity(3, np.arange(401)) * mu)
    assert 0.98 < total <= 1.0 + 1e-9          # the k^-2 tail beyond 400 holds the rest
    assert np.all(mu[:50] > 0)


def test_harmonic_target_energy():
    rng = np.random.default_rng(1)
    tgt = data.HarmonicTarget(3, s=1.0, rng=rng, kmax=30, terms=6)
    x = data.sphere(200000, 3, rng)
    f = tgt(x)
    assert np.mean(f ** 2) == pytest.approx(1.0, rel=0.02)
    assert abs(np.mean(f)) < 0.01


def test_flow_limits_and_gd():
    rng = np.random.default_rng(2)
    x = data.sphere(60, 3, rng)
    gram = np.exp(x @ x.T)
    y = rng.standard_normal(60)
    flow = krr.SpectralFlow(gram)
    pred_inf = gram @ flow.q @ flow.coefficients(y, np.inf)
    np.testing.assert_allclose(pred_inf, y, atol=1e-6)               # interpolation
    assert np.allclose(flow.coefficients(y, 0.0), 0)
    # GD with a tiny step for many steps approaches the flow at time lr*steps
    c_gd = flow.gd_coefficients(y, lr=1e-3, steps=20000)
    c_gf = flow.coefficients(y, 20.0)
    np.testing.assert_allclose(c_gd, c_gf, rtol=1e-2, atol=1e-8)


def test_eigenlearning_sanity():
    k = np.arange(0, 200)
    mult = spectrum.multiplicity(3, k)
    eig = (k + 1.0) ** -3.0
    eig = eig / np.sum(mult * eig)
    energy = np.zeros(200); energy[1] = 1.0
    r1 = eigenlearning.predict(eig, mult, energy, n=100, noise_var=0.0, delta=1e-3)
    r2 = eigenlearning.predict(eig, mult, energy, n=1000, noise_var=0.0, delta=1e-3)
    assert 0 <= r2 < r1 < 1
    best, _ = eigenlearning.predict_optimal_ridge(eig, mult, energy, n=1000, noise_var=0.25)
    interp = eigenlearning.predict(eig, mult, energy, n=1000, noise_var=0.25, delta=0.0, tail_mass=1e-6)
    assert best <= interp
