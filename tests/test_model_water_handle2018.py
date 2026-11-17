import numpy as np
from numpy.polynomial import Polynomial
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import root

from model_water_eos import handle2018 as model
from watereos.pv_eos import deriv_coeffs, polyval_rows, volume_roots


def _pressure_coefficients(T):
    f, _, _ = model._helmholtz_coefficients(np.atleast_1d(T))
    return -1000 * deriv_coeffs(f)


def test_paper_liquid_liquid_critical_point():
    def stationary(x):
        v, T = x
        p = Polynomial(_pressure_coefficients(T)[0])
        return p.deriv()(v), p.deriv(2)(v)

    sol = root(stationary, [model._M_G / 0.997, 175])
    assert sol.success
    assert_allclose(stationary(sol.x), 0, rtol=0, atol=1e-7)
    v, T = sol.x
    P = polyval_rows(_pressure_coefficients(T), np.array([v]))[0]
    # Sec. V C and Fig. 12 give uncertainty intervals for all three values.
    assert T == pytest.approx(175, abs=2)
    assert P == pytest.approx(175, abs=2)
    assert model._M_G / v == pytest.approx(0.9970, abs=0.001)


def test_table_i_liquid_free_energies():
    T = np.array([200.0, 250.0, 270.0])
    v = np.full_like(T, model._M_G / 1.10)
    P = polyval_rows(_pressure_coefficients(T), v)
    state = model.compute_batch(T, P)
    assert_allclose(state['rho'], 1100, rtol=1e-9)
    # Table I is from thermodynamic integration. The analytic PEL is a fit.
    assert_allclose(state['A'] * model.M_H2O / 1000,
                    [-54.15, -55.95, -56.93], rtol=0, atol=0.05)


def test_minimum_gibbs_selected_in_two_root_region():
    T = np.array([173.0, 173.0])
    P = np.array([185.2, 186.7])
    roots, g_rel = volume_roots(_pressure_coefficients(T), P, model._V_LO, model._V_HI)
    assert np.isfinite(roots[:, :2]).all()
    assert g_rel[0, 1] < g_rel[0, 0]
    assert g_rel[1, 0] < g_rel[1, 1]
    state = model.compute_batch(T, P)
    assert_allclose(state['V'] * model.M_H2O * 1e6, [roots[0, 1], roots[1, 0]])


def test_thermodynamic_derivatives():
    T = np.array([215.0, 240.0, 260.0])
    P = np.array([180.0, 350.0, 700.0])
    h = 0.01  # K or MPa
    state = model.compute_batch(T, P)
    t_plus = model.compute_batch(T + h, P)
    t_minus = model.compute_batch(T - h, P)
    p_plus = model.compute_batch(T, P + h)
    p_minus = model.compute_batch(T, P - h)

    S = -(t_plus['G'] - t_minus['G']) / (2 * h)
    V = (p_plus['G'] - p_minus['G']) / (2 * h * 1e6)
    Cp = T * (t_plus['S'] - t_minus['S']) / (2 * h)
    Kt = -state['V'] / ((p_plus['V'] - p_minus['V']) / (2 * h))
    alpha = (t_plus['V'] - t_minus['V']) / (2 * h * state['V'])
    Kp = (p_plus['Kt'] - p_minus['Kt']) / (2 * h)
    assert_allclose(
        [state[k] for k in ('S', 'V', 'Cp', 'Kt', 'alpha', 'Kp')],
        [S, V, Cp, Kt, alpha, Kp],
        rtol=2e-6, atol=1e-10,
    )


def test_probe_and_unavailable_root():
    state = model.compute_batch([273.15, 250.0, np.nan], [0.1, 1e5, 0.1])
    assert set(state) == {
        'rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv', 'Kt', 'Ks',
        'Kp', 'alpha', 'vel',
    }
    for values in state.values():
        assert np.isfinite(values[0])
        assert np.isnan(values[1:]).all()
    for branch in ('hdl', 'ldl'):
        with pytest.raises(ValueError, match='stable'):
            model.compute_batch([250.0], [100.0], branch=branch)
