"""Regression against Mishima & Sumita (2023), case 3, 7 × 7."""

import numpy as np
import pytest
from scipy.optimize import root

import mishima_eos as m
import watereos


@pytest.mark.parametrize('kind, expected', [
    ('critical', (204, 114, 0.982)),  # Table 1, K, MPa, cm³/g
    ('saddle', (194, 122, 0.951)),    # Table 2, same units
])
def test_published_special_points(kind, expected):
    def derivatives(tv):
        T, v = tv
        return [d[0] for d in m._pressure_derivatives(np.array([T]), np.array([v]))]

    def residual(tv):
        _, p_v, p_vv, p_t = derivatives(tv)
        return p_v, p_vv if kind == 'critical' else p_t

    sol = root(residual, (expected[0], expected[2]))
    assert sol.success
    T, v = sol.x
    p, *_ = derivatives(sol.x)
    assert T == pytest.approx(expected[0], abs=0.6)
    assert p == pytest.approx(expected[1], abs=0.5)
    assert v == pytest.approx(expected[2], abs=0.0005)


def test_atmospheric_pressure_low_density_root():
    roots = m.volume_roots([135, 150], 0.101325)
    assert np.isfinite(roots[0]).sum() == 1
    assert roots[0, 0] == pytest.approx(0.873, abs=0.001)
    assert np.isfinite(roots[1]).sum() == 2
    assert roots[1, 1] == pytest.approx(1.071, abs=0.001)

    batch = m.compute_batch([135, 150], 0.101325)
    np.testing.assert_allclose(batch['V'] * 1000, [roots[0, 0], roots[1, 1]])


def test_both_stable_roots_resolved_close_to_critical_point():
    roots = m.volume_roots(203.702975313, 113.707900324)[0]
    assert np.isfinite(roots).sum() == 2
    assert roots[0] < 0.98249 < roots[1]
    np.testing.assert_allclose(
        m._pressure_derivatives(np.array([203.702975313] * 2), roots[:2])[0],
        113.707900324, atol=1e-9, rtol=0)


def test_dispatch_and_volume_derivatives():
    T = np.array([220., 273.15, 298.15, 400.])
    p = np.array([50., 0.1, 100., 300.])
    out = watereos.getProp(np.column_stack((p, T)), 'mishima2023equation')
    assert set(vars(out)) == {'PTM', 'rho', 'V', 'Kt', 'alpha'}
    h = 0.01

    def volume(t, pressure):
        return m.compute_batch(t, pressure)['V']

    v_t = (volume(T + h, p) - volume(T - h, p)) / (2 * h)
    v_p = (volume(T, p + h) - volume(T, p - h)) / (2 * h)
    np.testing.assert_allclose(out.alpha, v_t / out.V, rtol=1e-5)
    np.testing.assert_allclose(out.Kt, -out.V / v_p, rtol=1e-5)
    np.testing.assert_allclose(out.rho * out.V, 1.0)
