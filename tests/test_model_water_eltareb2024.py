import numpy as np
from numpy.polynomial import Polynomial, polynomial as poly
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import least_squares

from model_water_eos import eltareb2024 as model
from watereos.pv_eos import deriv_coeffs, polyval_rows


def _pressure_coefficients(T):
    f, _, _ = model._helmholtz_coefficients(np.atleast_1d(T))
    return -model._PRESSURE_SCALE * deriv_coeffs(f)


def test_figure_7b_kauzmann_temperature():
    # The red curve has a minimum near 137 K at v about 0.88 cm^3/g.
    v = 0.88
    sigma2 = poly.polyval(v, model._SIGMA2)
    alpha = poly.polyval(v, model._ALPHA)
    b = poly.polyval(v, model._B)
    T_K = 1 / (model._R * (np.sqrt(2 * alpha / sigma2) - b))
    assert T_K == pytest.approx(137, abs=1)


def test_figure_8a_classical_isochore_at_200_k():
    # The cyan PEL curve at 1.20 g/cm^3 is about 435 MPa.
    u = np.array([1 / 1.20 - 1])
    P = polyval_rows(_pressure_coefficients(200.0), u)
    assert P[0] == pytest.approx(435, abs=30)
    assert_allclose(model.compute_batch([200.0], P)['rho'], 1200, rtol=1e-9)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='The direct Table S2 polynomials do not recover the Table S3 LLCP.')
def test_table_s3_liquid_liquid_critical_point():
    def stationary(x):
        u, T = x
        p = Polynomial(_pressure_coefficients(T)[0])
        return p.deriv()(u), p.deriv(2)(u)

    # An inflection must have both derivatives zero somewhere in the
    # reporting interval implied by the printed T and density precision.
    sol = least_squares(stationary, [1 / 1.03 - 1, 197],
                        bounds=([1 / 1.035 - 1, 196.5], [1 / 1.025 - 1, 197.5]))
    assert_allclose(stationary(sol.x), 0, rtol=0, atol=1e-6)
    u, T = sol.x
    P = Polynomial(_pressure_coefficients(T)[0])(u)
    assert P == pytest.approx(162, abs=0.5)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='Table S2 misses the independently integrated Table S1 free energy.')
def test_table_s1_reference_free_energy():
    f, _, _ = model._helmholtz_coefficients(np.array([260.0]))
    # u = v - 1, so at v = 1 only the constant Helmholtz coefficient remains.
    assert f[0, 0] == pytest.approx(-37.219, abs=0.0005)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='Table S2 misses the approximately ambient Table S1 pressure.')
def test_table_s1_reference_pressure():
    f, _, _ = model._helmholtz_coefficients(np.array([260.0]))
    assert -model._PRESSURE_SCALE * f[0, 1] == pytest.approx(0.1, abs=1)


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
