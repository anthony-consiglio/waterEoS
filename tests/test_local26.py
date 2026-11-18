"""Regression values of Local26 from its reference implementation.

The expected values come from amorphous/fit/stage3/s3_model.py (JAX) with the
coefficients of amorphous/fit/results/local_reduction/selected/Local26_T300.json,
which are those of watereos/data/local26.json. Specific SI units; the reference
works in molar units and in compressibility, converted here.

Below 297.5 K the two implementations agree to about 3e-10 in density and 3e-6
in the second-derivative properties. At 297.5-300 K the reference stops its
Newton iteration for the high-density state's expansion before convergence
(residual up to 2e-7 at 300 K, and growing above), so its compressibility and
expansivity there differ from the derivatives of its own volume by up to 0.7%
and 17%. waterEoS converges the root, so the points here stay below 295 K.

Local26's high-density state ends at 302.38 K, where the curvature phi of its
spinodal expansion, and with it Speedy's B^2, passes through zero; waterEoS
returns NaN above it. The reference clamps B^2 at 1e-12 and continues.
"""
import numpy as np
import pytest

from watereos import backend, compute, getProp
from watereos.local26.core import P_RANGE, T_RANGE
from watereos.local26.phase import find_LLCP
from watereos.model_registry import MODEL_REGISTRY

try:
    from watereos_rs import compute_batch_local26  # noqa: F401
    EXPECTED_BACKEND = 'rust'
except ImportError:
    EXPECTED_BACKEND = 'numpy'

M = 0.018015268  # kg/mol
T = np.array([180., 250., 273.15, 290., 200., 230.])
P = np.array([-50., 0.1, 0.1, 100., 200., 250.])
REFERENCE = {
    'rho': [907.227220811103, 991.224308482262, 999.838054233442,
            1040.673350587677, 1114.084601056069, 1117.063555819856],
    'vel': [2134.930453842274, 1250.082092752322, 1402.193387405799,
            1640.30077959814, 1417.702776957944, 1694.766831080381],
    'Cp': [1658.989163229707, 4573.28552261824, 4221.537029874288,
           3954.170034716878, 3978.179766580467, 3854.276247127972],
    # 1/kT, with kT in 1/MPa
    'Kt': 1 / np.array([0.0002424816839025255, 0.0006833215012839953, 0.0005090057836460738,
                        0.00036394575748090294, 0.0004687000518901693, 0.00032303087896682324]),
    'alpha': [7.361941888457e-05, -8.272405968657e-04, -6.968480057748e-05,
              3.107576065372e-04, 6.999328594455e-04, 4.610558785255e-04],
    'x': [0.965390999809, 0.167604862915, 0.080863204756, 0.025376599908,
          0.043522406419, 0.020551028037],
    # J/mol and J/(mol K) in the reference
    'G': np.array([-2.559102474633e+03, -7.577962262856e+01, 1.777356191757e+00,
                   1.726789900668e+03, 2.665763441700e+03, 3.992632418761e+03]) / M,
    'S': np.array([-3.096722184901e+01, -6.909411108855e+00, 1.968398683158e-02,
                   4.109052382400e+00, -2.213512233972e+01, -1.274328326245e+01]) / M,
}
RTOL = {'rho': 1e-11, 'vel': 1e-10, 'Cp': 1e-9, 'Kt': 1e-9, 'alpha': 1e-9, 'x': 1e-10,
        'G': 1e-11, 'S': 1e-10}


def test_points_match_reference():
    pt = np.empty(len(T), dtype=object)
    pt[:] = list(zip(P, T))
    out = getProp(pt, 'local26')
    assert backend('local26') == EXPECTED_BACKEND
    for key, expected in REFERENCE.items():
        np.testing.assert_allclose(getattr(out, key), expected, rtol=RTOL[key], err_msg=key)
    assert not hasattr(out, 'f') and not hasattr(out, 'eta')
    grid = compute([250., 290.], [0.1, 100.], 'Local26')
    np.testing.assert_allclose(grid.rho[[0, 1], [0, 1]], out.rho[[1, 3]], rtol=1e-14)


def test_critical_point_and_coexistence():
    assert find_LLCP() == pytest.approx(dict(T_K=218.72755498146427, p_MPa=56.66438157175168,
                                             x=.5), rel=1e-11)
    # Liquid-liquid coexistence at 200 K in the reference.
    p = 124.95273379597828
    h, l = [compute(200., p, 'local26', branch=b) for b in ('hdl', 'ldl')]
    assert h.x.item() == pytest.approx(0.144799778182, rel=1e-9)
    assert l.x.item() == pytest.approx(0.855200221818, rel=1e-9)
    assert h.rho.item() == pytest.approx(1059.987716842779, rel=1e-11)
    assert l.rho.item() == pytest.approx(949.698564851956, rel=1e-11)
    assert h.G.item() == pytest.approx(l.G.item(), abs=1e-6)
    assert h.G.item() == pytest.approx(1425.662745887225 / M, rel=1e-11)


def test_calibration_window_and_invalid_points():
    assert (T_RANGE, P_RANGE) == ((175.0, 300.0), (-50.0, 250.0))
    info = MODEL_REGISTRY['local26']
    assert (info.T_min, info.T_max, info.P_min, info.P_max) == (175, 300, -50, 250)
    assert info.is_two_state and info.has_phase_diagram and not info.has_transport
    with pytest.warns(UserWarning, match='outside suggested range'):
        outside = compute([175., 302.3, 302.4], [-140., 0.1], 'local26')
    assert np.isfinite(outside.rho[:, :2]).all() and np.isnan(outside.rho[:, 2]).all()
    with pytest.warns(UserWarning):
        invalid = compute(250., -600., 'local26')
    assert np.isnan(invalid.rho.item()) and np.isnan(invalid.x.item())
