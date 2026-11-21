"""Regression values of Bridge37 from its reference implementation.

The expected values come from amorphous/fit/stage3/s3_model.py (JAX) with the
coefficients of amorphous/fit/results/option_b_phase2/runs/PROD.json, which
are those of watereos/data/bridge37.json. Specific SI units; the reference
works in molar units and in compressibility, converted here.

The reference evaluates the spinodal functions as a clamped B-spline; waterEoS
converts them to power series in t. The two agree to 1e-12 or better, G and S
relative to their typical size.
"""
from math import comb

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval

from watereos import backend, compute, getProp
from watereos.bridge29.equation import BERNSTEIN_RANGE_K, bernstein_to_power
from watereos.bridge37.core import EQUATION, P_RANGE, T_RANGE
from watereos.bridge37.phase import find_LLCP, vapor_spinodal
from watereos.model_registry import MODEL_REGISTRY

try:
    from watereos_rs import compute_batch_bridge37  # noqa: F401
    EXPECTED_BACKEND = 'rust'
except ImportError:
    EXPECTED_BACKEND = 'numpy'

M = 0.018015268  # kg/mol
T = np.array([120., 180., 230., 250., 273.15, 300., 350., 400.])
P = np.array([300., -50., 200., 0.1, 0.1, 100., 400., -100.])
REFERENCE = {
    'rho': [1199.2673762288825, 892.5379634385789, 1097.0870473777532, 991.3033032283217,
            999.8364757885245, 1037.183469592183, 1100.0653548247199, 872.2617770644654],
    'vel': [1783.2187307495087, 1954.1343836907138, 1567.9586646266685, 1244.950502440446,
            1401.5639493115616, 1668.0139813101455, 2149.58466149706, 1208.8523587164375],
    'Cp': [1242.5329358316105, 1852.5581520879664, 3623.3592393233084, 4557.636139239371,
           4227.670624570753, 3971.0982193711498, 3826.124349622298, 4822.51231179241],
    # 1/kT, with kT in 1/MPa
    'Kt': 1 / np.array([0.00027329691815697153, 0.000293405233826002, 0.00037618934862351946,
                        0.0006881947580486499, 0.0005094524851598557, 0.0003558725650885722,
                        0.0002156489969059953, 0.000987333535315441]),
    'alpha': [0.00037078747705873873, 4.598169151854725e-06, 0.0003064042150083057,
              -0.0008213816406588725, -6.849171962975273e-05, 0.00035807797287076436,
              0.0004769736583265239, 0.0014604056678203842],
    'x': [0.004697214687959639, 0.9642051246100324, 0.05333924840178306, 0.19476521355602383,
          0.10644898215062021, 0.035470107535228335, 0.0050459649246124005, 0.01343291954346365],
    # J/mol and J/(mol K) in the reference
    'G': np.array([1672.9008903862882, -2587.22520045495, 3180.9755529269955, -75.67754625503859,
                   1.777423330485607, 1673.3034906996375, 6164.378252904146, -3930.8988761962523]) / M,
    'S': np.array([-41.883577505363704, -31.97727994196828, -12.234417792486099, -6.90159301663652,
                   0.02637200119998445, 6.540371182125533, 15.06029472005696, 31.12481441368943]) / M,
}
# (rtol, atol): G and S pass through zero near the triple point.
TOL = {'rho': (1e-13, 0.), 'vel': (1e-13, 0.), 'Cp': (1e-13, 0.), 'Kt': (1e-13, 0.), 'x': (1e-12, 0.),
       'alpha': (1e-12, 1e-16), 'G': (1e-12, 1e-8), 'S': (1e-12, 1e-10)}


def test_points_match_reference():
    pt = np.empty(len(T), dtype=object)
    pt[:] = list(zip(P, T))
    out = getProp(pt, 'bridge37')
    assert backend('bridge37') == EXPECTED_BACKEND
    for key, expected in REFERENCE.items():
        rtol, atol = TOL[key]
        np.testing.assert_allclose(getattr(out, key), expected, rtol=rtol, atol=atol, err_msg=key)
    assert not hasattr(out, 'f') and not hasattr(out, 'eta')
    grid = compute([250., 300.], [0.1, 100.], 'Bridge37')
    np.testing.assert_allclose(grid.rho[[0, 1], [0, 1]], out.rho[[3, 5]], rtol=1e-14)


def test_critical_point_and_coexistence():
    assert find_LLCP() == pytest.approx(dict(T_K=214.26071468061156, p_MPa=70.87475656417614,
                                             x=.5), rel=1e-12)
    # Liquid-liquid coexistence at 200 K in the reference.
    p = 122.44694171847038
    h, l = [compute(200., p, 'bridge37', branch=b) for b in ('hdl', 'ldl')]
    assert h.x.item() == pytest.approx(0.2063687202527063, rel=1e-12)
    assert l.x.item() == pytest.approx(0.7936312797472744, rel=1e-12)
    assert h.rho.item() == pytest.approx(1047.9433020548202, rel=1e-13)
    assert l.rho.item() == pytest.approx(950.1595148066889, rel=1e-13)
    assert h.G.item() == pytest.approx(l.G.item(), abs=1e-6)
    assert h.G.item() == pytest.approx(1393.7575539711736 / M, rel=1e-12)


def test_bernstein_conversion():
    """The power series in t equal the Bernstein quartics in u on 100-420 K."""
    lo, hi = BERNSTEIN_RANGE_K
    T_eff = np.linspace(lo, hi, 33)
    t, u = (T_eff - 200.) / 100., (T_eff - lo) / (hi - lo)
    Q = EQUATION.Q
    for key, series in (('psk', EQUATION.PS), ('vsk', EQUATION.VS), ('phk', EQUATION.PHI),
                        ('csk', EQUATION.CUBIC)):
        direct = sum(Q[f'{key}{k}'] * comb(4, k) * u**k * (1 - u)**(4 - k) for k in range(5))
        np.testing.assert_allclose(series(t), direct, rtol=0, atol=1e-14 * np.abs(direct).max(),
                                   err_msg=key)
    # A constant, and the identity u = sum_k (k/4) B_k,4(u), come out as such.
    np.testing.assert_allclose(bernstein_to_power([2.] * 5), [2., 0., 0., 0., 0.], atol=1e-15)
    np.testing.assert_allclose(polyval(t, bernstein_to_power([0., .25, .5, .75, 1.])), u, atol=1e-15)


def test_scope_and_invalid_points():
    assert (T_RANGE, P_RANGE) == ((100.0, 400.0), (-140.0, 400.0))
    info = MODEL_REGISTRY['bridge37']
    assert (info.T_min, info.T_max, info.P_min, info.P_max) == (100, 400, -140, 400)
    assert info.is_two_state and info.has_phase_diagram and not info.has_transport
    # The high-density state's liquid-vapor spinodal: -168.2 MPa at 400 K.
    assert vapor_spinodal(400.) == pytest.approx(-168.2478211, abs=1e-6)
    with pytest.warns(UserWarning, match='outside suggested range'):
        outside = compute(400., [-160., -170.], 'bridge37')
    assert outside.rho.shape == (2, 1)  # pressure along the first axis
    assert np.isfinite(outside.rho[0, 0]) and np.isnan(outside.rho[1, 0])
    with pytest.warns(UserWarning):
        invalid = compute(250., -600., 'bridge37')
    assert np.isnan(invalid.rho.item()) and np.isnan(invalid.x.item())
