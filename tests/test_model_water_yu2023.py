"""Checks for the TIP4P/2005 hierarchical two-state fit of Yu et al. (2023)."""

import numpy as np
import pytest

from model_water_eos import yu2023 as model
from watereos.batch_getprop import getProp


def test_batch_contract_and_probe():
    key = 'yu2023_tip4p2005'
    assert model.MODELS == {key: model.compute_batch}
    assert model.BRANCH_MODELS == (key,)
    assert model.NAN_KEYS == {key: ()}
    out = model.compute_batch([273.15], [0.1])
    volumetric = ('rho', 'V', 'Kt', 'alpha')
    properties = {'x', 'D', 'tau_r', 'f'}
    properties.update(k + suffix for k in volumetric for suffix in ('', '_A', '_B'))
    assert out.keys() == properties
    assert all(v.shape == (1,) for v in out.values())
    for k in (*volumetric, 'x', 'D', 'tau_r', 'f'):
        assert np.isfinite(out[k][0])

    named = model.compute_batch([273.15], [0.1], 'hdl')
    assert named.keys() == properties | {'metastable'}
    PT = np.array([[0.1], [273.15]], dtype=object)
    grid = getProp(PT, model.MODELS[key], nan_keys=model.NAN_KEYS[key])
    assert vars(grid).keys() == properties | {'PTM'}


def test_table_1_and_fig_5_critical_point():
    cp = model.critical_point()
    assert (cp['T_K'], cp['p_MPa']) == (175.9, 177.2)
    # Density is implied by Table 1 and the derivative of the Gibbs fit.
    assert cp['rho'] == pytest.approx(1023.44, abs=0.01)
    out = model.compute_batch([175.9], [177.2])
    for k in ('x', 'f'):
        assert out[k][0] == pytest.approx(0.5, abs=1e-5)


def test_fig_1_and_2_ambient_thermodynamics():
    out = model.compute_batch([235.0, 280.0], 0.1)
    # Fig. 1 caption and fits in Figs. 2B and 2C, respectively.
    assert out['x'][0] == pytest.approx(0.5, abs=0.02)
    assert out['rho'][1] == pytest.approx(997.0, abs=3.0)
    assert 1e5 / out['Kt'][0] == pytest.approx(65.0, abs=5.0)


def test_fig_3_dynamics():
    out = model.compute_batch(200.0, [0.1, 180.0])
    # Solid curves at 1000/T = 5, using the units of the printed axes.
    log_inv_D = -np.log10(out['D'] * 1e6)
    log_tau = np.log10(out['tau_r'] * 1e12)
    np.testing.assert_allclose(log_inv_D, [6.7, 4.6], rtol=0.0, atol=0.1)
    np.testing.assert_allclose(log_tau, [4.9, 2.3], rtol=0.0, atol=0.1)
    # Red curve in Fig. 3C near s = 0.86.
    assert out['f'][0] == pytest.approx(0.62, abs=0.02)


def test_fig_4_diffusion_scale_and_slowing_peak():
    # Fig. 4A near the Price et al. point at 298.15 K. The fit curve at
    # this coordinate reads about 2.65 in common-log units.
    out = model.compute_batch([298.15], [0.1])
    assert -np.log10(out['D'][0] * 1e6) == pytest.approx(2.65, abs=0.05)

    inv_T = np.linspace(1000.0 / 350.0, 1000.0 / 170.0, 3001)
    out = model.compute_batch(1000.0 / inv_T, 0.1)
    rate = np.gradient(-np.log10(out['D']), inv_T)
    i = np.argmax(rate)
    # The text gives about 205 K. The Fig. 4B fit curve peaks near 207 K.
    assert 1000.0 / inv_T[i] == pytest.approx(205.0, abs=3.0)
    assert rate[i] == pytest.approx(4.7, abs=0.2)
    assert out['f'][i] == pytest.approx(0.5, abs=0.025)


@pytest.mark.parametrize('branch', ['stable', 'hdl', 'ldl'])
def test_gibbs_derivatives(branch):
    # Use the internal Gibbs convention on both sides of the LLT,
    # with both local minima present.
    T = np.array([175.0, 175.0, 174.8, 174.8])
    P = np.array([179.15, 179.23, 179.6, 179.68])
    dt, dp = 0.0002, 0.0005
    calc = model._compute_thermodynamics
    out = calc(T, P, branch)
    warm = calc(T + dt, P, branch)
    cold = calc(T - dt, P, branch)
    high = calc(T, P + dp, branch)
    low = calc(T, P - dp, branch)

    expected = {
        'S': -(warm['G'] - cold['G']) / (2 * dt),
        'V': (high['G'] - low['G']) / (2 * dp * 1e6),
        'Cp': T * (warm['S'] - cold['S']) / (2 * dt),
        'Kt': -out['V'] / ((high['V'] - low['V']) / (2 * dp)),
        'alpha': (warm['V'] - cold['V']) / (2 * dt * out['V']),
    }
    for k, value in expected.items():
        assert np.all(np.isfinite(out[k]))
        np.testing.assert_allclose(out[k], value, rtol=1e-5, atol=0.0, err_msg=k)


def test_named_branch_dynamics():
    # HDL is metastable at point 0, stable at point 1 and absent at point 2.
    T = [175.0, 175.0, 200.0]
    P = [179.15, 179.22, 0.1]
    stable = model.compute_batch(T, P)
    hdl = model.compute_batch(T, P, 'hdl')
    ldl = model.compute_batch(T, P, 'ldl')
    np.testing.assert_array_equal(hdl['metastable'], [True, False, False])
    np.testing.assert_array_equal(ldl['metastable'], [False, True, False])
    assert np.isfinite(hdl['rho'][0])
    assert np.isnan(hdl['rho'][2])
    for k in ('D', 'tau_r', 'f'):
        np.testing.assert_allclose(
            hdl[k], [np.nan, stable[k][1], np.nan], rtol=0.0, atol=0.0,
        )
        np.testing.assert_allclose(
            ldl[k], [stable[k][0], np.nan, stable[k][2]], rtol=0.0, atol=0.0,
        )
