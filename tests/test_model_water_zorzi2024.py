"""Checks for the ML-BOP Gibbs fit in Zorzi et al. (2024)."""

import numpy as np
import pytest
from scipy.optimize import brentq

from model_water_eos import zorzi2024 as model
from watereos.two_state_general import TWO_STATE_KEYS


def test_batch_contract_and_probe():
    key = 'zorzi2024_mlbop'
    assert model.MODELS == {key: model.compute_batch}
    assert model.BRANCH_MODELS == (key,)
    out = model.compute_batch([273.15], [0.1])
    properties = set(TWO_STATE_KEYS) | {'x'}
    properties.update(k + suffix for k in TWO_STATE_KEYS for suffix in ('_A', '_B'))
    assert out.keys() == properties
    assert all(v.shape == (1,) for v in out.values())
    for k in ('rho', 'V', 'alpha', 'S', 'G', 'H', 'Cp'):
        assert np.isfinite(out[k][0])


def test_critical_fraction():
    # Fixed coordinates from ref. 20, with psi = 1/2 from p. 3.
    cp = model.critical_point()
    assert (cp['T_K'], cp['p_MPa']) == (181.5, 175.0)
    out = model.compute_batch([181.5], [175.0])
    assert out['x'][0] == pytest.approx(0.5, abs=1e-5)


def test_fig_2_and_4_high_pressure():
    # Fits at 100 K and 7000 bar, read from Figs. 2a, 2b and 4a.
    out = model.compute_batch([100.0], [700.0])
    assert out['rho'][0] == pytest.approx(1256.0, abs=3.0)
    assert out['U'][0] * model.M / 1e3 == pytest.approx(-38.05, abs=0.05)
    assert 1e3 / out['Kt'][0] == pytest.approx(0.125, abs=0.015)


def test_fig_3_order_parameter():
    out = model.compute_batch([250.0, 250.0], [0.1, 700.0])
    np.testing.assert_allclose(out['x'], [0.50, 0.075], rtol=0.0, atol=0.01)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    'Printed parameters give 1022.2 kg/m3 versus about 997 in Fig. 2a.'
))
def test_fig_2_ambient_density():
    out = model.compute_batch([270.0], [0.1])
    assert out['rho'][0] == pytest.approx(997.0, abs=3.0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    'Printed parameters imply 1040.4 kg/m3 versus about 1020 in Fig. 1.'
))
def test_fig_1_critical_density():
    assert model.critical_point()['rho'] == pytest.approx(1020.0, abs=8.0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    'Printed parameters give 986.6 and 1149.7 kg/m3 versus 974 and 1132 in Fig. 1.'
))
def test_fig_1_coexistence_densities():
    T = np.array([150.0])

    def delta_g(P):
        return model.terms(T, np.array([P * 1e6]))[1][0][0]

    P = brentq(delta_g, 295.0, 320.0)
    ldl = model.compute_batch(T, [P], 'ldl')
    hdl = model.compute_batch(T, [P], 'hdl')
    np.testing.assert_allclose(
        [ldl['rho'][0], hdl['rho'][0]], [974.0, 1132.0], rtol=0.0, atol=5.0,
    )


@pytest.mark.parametrize('branch', ['stable', 'hdl', 'ldl'])
def test_gibbs_derivatives(branch):
    # Both minima exist at each point. These points span both sides
    # of coexistence, so the named branches include metastability.
    T = np.array([150.0, 160.0, 170.0])
    P = np.array([307.0, 263.3, 221.0])
    dt, dp = 0.001, 0.001
    out = model.compute_batch(T, P, branch)
    warm = model.compute_batch(T + dt, P, branch)
    cold = model.compute_batch(T - dt, P, branch)
    high = model.compute_batch(T, P + dp, branch)
    low = model.compute_batch(T, P - dp, branch)

    expected = {
        'S': -(warm['G'] - cold['G']) / (2 * dt),
        'V': (high['G'] - low['G']) / (2 * dp * 1e6),
        'Cp': T * (warm['S'] - cold['S']) / (2 * dt),
        'Kt': -out['V'] / ((high['V'] - low['V']) / (2 * dp)),
        'alpha': (warm['V'] - cold['V']) / (2 * dt * out['V']),
    }
    for k, value in expected.items():
        assert np.all(np.isfinite(out[k]))
        np.testing.assert_allclose(out[k], value, rtol=2e-6, atol=0.0, err_msg=k)
    if branch != 'stable':
        np.testing.assert_array_equal(
            out['metastable'], [branch == 'hdl', branch == 'ldl', branch == 'ldl'],
        )
