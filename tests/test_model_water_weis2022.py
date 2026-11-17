"""Published WAIL checks and consistency of the Table III implementation."""

import numpy as np
import pytest

from model_water_eos import weis2022 as wail

compute = wail.MODELS['weis2022_wail']


def test_critical_point():
    assert wail.Tc == 208.5
    assert wail.Pc / 1e6 == pytest.approx(35.63)
    result = compute([208.5], [35.63])
    assert result['x'][0] == pytest.approx(0.5, abs=5e-6)
    assert result['rho'][0] == pytest.approx(965.0, abs=5.0)  # Fig. 1A


def _coexistence_pressure(T):
    u = (T - wail.Tc) / wail.Tc
    return wail.Pc / 1e6 * (1.0 - u / (wail.A + wail.B * u))


def test_coexisting_densities_fig1a():
    P = _coexistence_pressure(200.0)
    hdl = compute([200.0], [P], branch='hdl')
    ldl = compute([200.0], [P], branch='ldl')
    np.testing.assert_allclose(hdl['G'], ldl['G'], rtol=0, atol=1e-6)
    assert hdl['rho'][0] == pytest.approx(1048.0, abs=4.0)
    assert ldl['rho'][0] == pytest.approx(908.0, abs=4.0)


def test_low_pressure_isobar_fig1():
    assert compute([250.0], [10.0])['rho'][0] == pytest.approx(1032.0, abs=2.0)
    np.testing.assert_allclose(compute([200.0, 260.0], [10.0])['x'],
                               [0.81, 0.23], rtol=0, atol=0.01)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    'Table III gives 998.21 kg/m3 at 250 K, 80 MPa; Fig. 1A shows about 1079'))
def test_high_pressure_isobar_fig1a():
    assert compute([250.0], [80.0])['rho'][0] == pytest.approx(1079.0, abs=2.0)


def _check_derivatives(T, P, branch):
    T, P = np.asarray(T), np.asarray(P)
    dt, dp = 0.001, 0.001  # K, MPa
    out = compute(T, P, branch=branch)
    hot = compute(T + dt, P, branch=branch)
    cold = compute(T - dt, P, branch=branch)
    high = compute(T, P + dp, branch=branch)
    low = compute(T, P - dp, branch=branch)
    expected = {
        'S': -(hot['G'] - cold['G']) / (2 * dt),
        'V': (high['G'] - low['G']) / (2 * dp * 1e6),
        'Cp': T * (hot['S'] - cold['S']) / (2 * dt),
        'Kt': -out['V'] * (2 * dp) / (high['V'] - low['V']),
        'alpha': (hot['V'] - cold['V']) / (2 * dt * out['V']),
    }
    for name, value in expected.items():
        assert np.isfinite(out[name]).all()
        np.testing.assert_allclose(out[name], value, rtol=2e-6, atol=1e-10,
                                   err_msg=name)


@pytest.mark.parametrize('branch, T, P', [
    ('stable', [216.0, 230.0, 248.0], [12.0, 18.0, 22.0]),
    ('hdl', [225.0, 235.0, 248.0], [12.0, 18.0, 22.0]),
    ('ldl', [215.1, 216.0, 218.0], [11.0, 10.5, 10.5]),
])
def test_thermodynamic_consistency(branch, T, P):
    _check_derivatives(T, P, branch)


@pytest.mark.parametrize('branch, metastable', [
    ('hdl', [True, False]),
    ('ldl', [False, True]),
])
def test_metastable_branches(branch, metastable):
    T = 200.5
    P = _coexistence_pressure(T) + np.array([-0.5, 0.5])
    np.testing.assert_array_equal(compute(T, P, branch=branch)['metastable'],
                                  metastable)
    _check_derivatives(T, P, branch)


def test_registry_probe():
    out = compute([273.15], [0.1])
    for key in ('rho', 'V', 'alpha', 'S', 'G', 'H', 'Cp'):
        assert np.isfinite(out[key][0])
