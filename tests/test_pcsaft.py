"""Published curve checks for the Novak pure-water PC-SAFT fits."""

import numpy as np
import pytest
from scipy.optimize import brentq

import pcsaft_eos
from pcsaft_eos.core import _residual
from watereos import compute
from watereos.model_registry import MODEL_REGISTRY
from watereos.two_state_general import M_H2O, R

TV = 'novak2024_pcsaft_ts_tv'
T_ONLY = 'novak2024_pcsaft_ts_t'
CAF = 'novak2025_pcsaft_ts_caf'
BASE = 'pcsaft_water'


@pytest.mark.parametrize('key, temperature, expected', [
    (TV, 300.0, 0.26),
    (CAF, 300.0, 0.105),
    (CAF, 250.0, 0.46),
])
def test_fig12_low_density_fraction(key, temperature, expected):
    assert compute(temperature, 0.1, key).x.item() == pytest.approx(expected, abs=0.015)


@pytest.mark.parametrize('key, expected', [
    (TV, 5.8), (T_ONLY, 11.3), (CAF, 3.8), (BASE, 3.3),
])
def test_fig10_ambient_heat_capacity(key, expected):
    out = compute(273.16, 0.1, key)
    assert out.Cp.item() / 1000 == pytest.approx(expected, abs=0.2)


@pytest.mark.parametrize('key', [TV, T_ONLY, CAF, BASE])
def test_fig8_density_at_400_mpa(key):
    out = compute(600.0, 400.0, key)
    assert out.rho.item() / (1000 * M_H2O) == pytest.approx(52.5, abs=0.55)


def test_2024_fig3_sound_speed():
    out = compute(350.0, 0.1, TV)
    assert out.vel.item() == pytest.approx(1555, abs=15)


@pytest.mark.parametrize('key', [TV, T_ONLY, CAF])
def test_density_maximum_at_ambient_pressure(key):
    def expansivity(temperature):
        return compute(temperature, 0.1, key).alpha.item()

    # Both papers' fits target d(rho)/dT > 0 at 276.15 K.
    assert expansivity(276.15) < 0
    assert brentq(expansivity, 276.15, 285.0) == pytest.approx(277.0, abs=1.0)


@pytest.mark.parametrize('key', [TV, T_ONLY, CAF, BASE])
def test_derivatives_and_registry(key):
    T = np.array([290., 450., 600.])
    p = np.array([0.1, 100., 400.])
    out = pcsaft_eos.compute_batch(T, p, key)
    assert set(out) == set(MODEL_REGISTRY[key].properties)
    assert np.isfinite(list(out.values())).all()
    np.testing.assert_allclose(pcsaft_eos.pressure(T, out['rho'], key), p, atol=1e-6)
    h = 0.02
    hot = pcsaft_eos.compute_batch(T + h, p, key)
    cold = pcsaft_eos.compute_batch(T - h, p, key)
    high = pcsaft_eos.compute_batch(T, p + h, key)
    low = pcsaft_eos.compute_batch(T, p - h, key)
    np.testing.assert_allclose(out['S'], -(hot['G'] - cold['G']) / (2*h), atol=0.02)
    np.testing.assert_allclose(out['V'], (high['G'] - low['G']) / (2*h*1e6), rtol=5e-6)
    np.testing.assert_allclose(out['Cp'], T * (hot['S'] - cold['S']) / (2*h), rtol=1e-5)
    np.testing.assert_allclose(out['alpha'], (hot['V'] - cold['V']) / (2*h*out['V']), rtol=1e-5)
    np.testing.assert_allclose(out['Kt'], -2*h*out['V'] / (high['V'] - low['V']), rtol=1e-5)

    # Independent real-density difference of Aʳ verifies complex-step P.
    n = out['rho'] / M_H2O
    fit = pcsaft_eos.FITS[key]
    dr = (_residual(T, n + h, fit) - _residual(T, n - h, fit)) / (2*h)
    np.testing.assert_allclose(p, n * R * T * (1 + n * dr) / 1e6, atol=2e-5)


def test_no_dense_root_returns_nan():
    out = pcsaft_eos.compute_batch([700, np.nan], [0.1, 100], CAF)
    assert np.isnan(out['rho']).all()
    with pytest.raises(ValueError, match='branch'):
        compute(300, 100, CAF, branch='ldl')
