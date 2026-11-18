"""
The Rust backend of Bridge29 (src/bridge29.rs) against the NumPy core
(watereos/bridge29/core.py).

Both evaluate the same operations; NumPy takes second derivatives of the
high-density state by a complex step and Rust by dual numbers, which give
the same derivative. They differ by rounding, which the model amplifies
only next to a singular point: the liquid-liquid critical point and
spinodals, where the curvature of the mixing Gibbs energy vanishes and the
second-derivative properties diverge, and the vapor spinodal of the
high-density state. The first-derivative properties agree to about 1e-12
everywhere, and so do the second-derivative ones away from those points;
measured on a 0.5 K by 1 MPa grid, the second-derivative ones differ by up
to 1e-12 where the normalized mixing curvature below is 0.2 or more, and
by up to 6e-10 as it goes to zero at a spinodal.

Skipped if the installed watereos_rs has no Bridge29.
"""
import numpy as np
import pytest

rs = pytest.importorskip('watereos_rs')
if not hasattr(rs, 'compute_batch_bridge29'):
    pytest.skip('installed watereos_rs has no compute_batch_bridge29',
                allow_module_level=True)

from watereos import backend  # noqa: E402
from watereos.bridge29 import compute_batch, core  # noqa: E402
from watereos.bridge29.phase import vapor_spinodal  # noqa: E402

SECOND = ('Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel')


def _grid():
    """100-400 K and -140 to 400 MPa, including points beyond the vapor
    spinodal of the high-density state (NaN in both backends)."""
    T, P = np.meshgrid(np.arange(100.0, 400.01, 2.5),
                       np.arange(-140.0, 400.01, 5.0))
    return T.ravel(), P.ravel()


def _well_conditioned(T, P, x):
    """Points away from the singular points named in the module docstring.

    The normalized curvature 1 - 2 W x (1 - x) / (R_mix T) of the mixing
    Gibbs energy is 1 for a pure state and 0 at a liquid-liquid spinodal.
    """
    W = core.Q['w0'] + core.Q['w1'] * P
    with np.errstate(invalid='ignore'):
        curvature = 1 - 2 * W * x * (1 - x) / (core.RM * T)
        return (curvature > 0.2) & (P > vapor_spinodal(T) + 5.0)


@pytest.mark.parametrize('branch', ['stable', 'hdl', 'ldl'])
def test_rust_matches_numpy(branch):
    T, P = _grid()
    r = rs.compute_batch_bridge29(T, P, branch=branch)
    n = core.compute_batch(T, P, branch)
    assert list(r) == list(n)
    if branch == 'stable':
        assert 'metastable' not in r
    else:
        np.testing.assert_array_equal(np.asarray(r['metastable']), n['metastable'])
        assert n['metastable'].any()
    x = n['x']
    assert np.isnan(x).any() and np.isfinite(x).sum() > 4000
    well = _well_conditioned(T, P, x)
    assert well.sum() > 0.85 * np.isfinite(x).sum()
    for key in n:
        if key == 'metastable':
            continue
        a, b = np.asarray(r[key]), n[key]
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=key)
        finite = np.isfinite(b)
        # Relative to the value, or to the property's median size where the
        # value passes through zero (G, S, H, U, A have arbitrary zeros).
        scale = np.maximum(np.abs(b), np.median(np.abs(b[finite])))
        err = np.where(finite, np.abs(a - b) / scale, 0.0)
        second = key.split('_')[0] in SECOND
        # At most 5e-13 on macOS arm64; the bound leaves room for the
        # vectorized exp and log NumPy uses on other platforms.
        assert err[well].max() < 5e-12, (key, err[well].max())
        assert err[~well].max() < (1e-8 if second else 1e-11), (key, err[~well].max())


def test_invalid_points_are_nan():
    T = np.array([0.0, -5.0, np.nan, 250.0, 250.0])
    P = np.array([0.1, 0.1, 0.1, np.nan, -600.0])
    for branch in ('stable', 'hdl'):
        r = rs.compute_batch_bridge29(T, P, branch=branch)
        n = core.compute_batch(T, P, branch)
        for key in n:
            if key == 'metastable':
                assert not np.asarray(r[key]).any() and not n[key].any()
            else:
                assert np.isnan(np.asarray(r[key])).all(), key
                assert np.isnan(n[key]).all(), key


def test_unknown_branch_is_rejected():
    with pytest.raises(ValueError, match='Unknown branch'):
        rs.compute_batch_bridge29(np.array([250.0]), np.array([0.1]), branch='liquid')


def test_package_dispatch():
    """watereos.bridge29.compute_batch uses Rust, broadcasts like the NumPy
    core, and leaves a forced composition to NumPy."""
    assert backend('bridge29') == 'rust'
    T = np.array([[180.0], [230.0]])
    P = np.array([0.1, 100.0, 200.0])
    out = compute_batch(T, P, 'hdl')
    ref = core.compute_batch(T, P, 'hdl')
    assert list(out) == list(ref)
    for key in ref:
        assert out[key].shape == (2, 3), key
        np.testing.assert_allclose(out[key], ref[key], rtol=1e-12, equal_nan=True,
                                   err_msg=key)
    scalar = compute_batch(250.0, 50.0)
    assert scalar['rho'].shape == ()
    np.testing.assert_allclose(scalar['rho'], core.compute_batch(250.0, 50.0)['rho'],
                               rtol=1e-12)
    forced = compute_batch(T, P, x=np.full((2, 3), 0.3))
    np.testing.assert_array_equal(forced['x'], 0.3)
    np.testing.assert_array_equal(forced['G'],
                                  core.compute_batch(T, P, x=np.full((2, 3), 0.3))['G'])
