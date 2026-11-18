"""Regression values from the frozen Bridge29 manuscript model (specific SI units)."""
import numpy as np
import pytest

from watereos import backend, compute, getProp
from watereos.bridge29.phase import find_LLCP

try:
    from watereos_rs import compute_batch_bridge29  # noqa: F401
    EXPECTED_BACKEND = 'rust'
except ImportError:
    EXPECTED_BACKEND = 'numpy'


def test_grid_and_scatter_match_manuscript():
    grid = compute([150., 250.], [0.1, 100.], 'Bridge29')
    assert backend('bridge29') == EXPECTED_BACKEND
    np.testing.assert_allclose(grid.rho[0], [912.52116856, 991.34620673], rtol=1e-10)
    np.testing.assert_allclose(grid.vel[0], [1959.32356812, 1243.67866267], rtol=1e-10)
    np.testing.assert_allclose(grid.Cp[0], [1364.74218956, 4554.67043861], rtol=1e-10)
    pt = np.empty(2, dtype=object)
    pt[:] = [(100., 150.), (0.1, 250.)]
    scatter = getProp(pt, 'bridge29')
    for key in ('rho', 'rho_A', 'rho_B', 'x', 'f', 'D', 'eta', 'tau_r'):
        np.testing.assert_allclose(getattr(scatter, key), getattr(grid, key)[[1, 0], [0, 1]])
    values = [0.1986433483, 0.0732808602, 3.91679493e-10, 0.00520026432, 1.10479296e-11]
    for key, expected in zip(('x', 'f', 'D', 'eta', 'tau_r'), values):
        assert getattr(scatter, key)[1] == pytest.approx(expected, rel=2e-8, abs=0)


def test_coexisting_compositions_and_stability_limit():
    # Low-temperature coexistence point from the manuscript's phase-boundary table.
    h, l = [compute(100., 227.6706077220823, 'bridge29', branch=b) for b in ('hdl', 'ldl')]
    assert h.x.item() == pytest.approx(0.00394976235)
    assert l.x.item() == pytest.approx(1-h.x.item())
    assert h.G.item() == pytest.approx(l.G.item(), abs=1e-6)
    assert h.rho.item() > l.rho.item()
    assert find_LLCP() == pytest.approx(dict(T_K=222.3396934523, p_MPa=47.6454410840, x=.5))
    with pytest.warns(UserWarning):
        invalid = compute(250., -600., 'bridge29')
    assert np.isnan(invalid.rho.item()) and np.isnan(invalid.D.item())
