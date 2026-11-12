"""
Tests for liquid-branch selection in the two-state models.

``branch='hdl'`` / ``'ldl'`` return the high-density / low-density liquid
wherever it exists, including its metastable range between the
liquid-liquid transition and its spinodal.  The default (``'stable'``)
must be unchanged.

The transition and spinodal temperatures used as references come from the
package's own phase-diagram routines, which were written independently of
the branch solver.
"""
import importlib
import warnings

import numpy as np
import pytest

from watereos import compute, getProp
from watereos.two_state_branch import select_branch, solve_branches

# model key -> (package, pressure offset above the LLCP [MPa])
TWO_STATE = {
    'holten2014': ('holten_eos', 80.0),
    'caupin2019': ('caupin_eos', 40.0),
    'duska2020': ('duska_eos', 60.0),
}
MIX_KEYS = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv', 'Kt', 'Ks',
            'alpha', 'vel', 'x']


def _compute(T, P, model, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute(T, P, model, **kw)


def _tol(key):
    """Relative tolerance for comparing one property between two evaluations.

    The default branch may run on a different backend than a named branch
    (Rust versus NumPy), and second-derivative properties agree between
    backends less tightly than the first-derivative ones.
    """
    second = key.split('_')[0] in ('Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel')
    return 1e-6 if second else 1e-8


def _lines(model):
    """Pressure, binodal and both spinodal temperatures at one pressure."""
    pkg, dp = TWO_STATE[model]
    pd = importlib.import_module(f'{pkg}.phase_diagram')
    p = pd.find_LLCP()['p_MPa'] + dp
    sp = pd.compute_spinodal_curve(p_range_MPa=np.array([p]))
    bn = pd.compute_binodal_curve(p_range_MPa=np.array([p]), spinodal=sp)
    return dict(p=p, T_bin=float(bn['T_binodal'][0]),
                T_ldl_spin=float(sp['T_upper'][0]),
                T_hdl_spin=float(sp['T_lower'][0]))


# ---------------------------------------------------------------------------
# The default path is unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model', list(TWO_STATE) + ['caupin2019_kim'])
def test_stable_is_the_default(model):
    T = np.arange(300.0, 150.0, -7.0)
    P = np.array([0.1, 60.0, 120.0, 200.0])
    a = _compute(T, P, model)
    b = _compute(T, P, model, branch='stable')
    for k in MIX_KEYS + ['rho_A', 'G_B']:
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    assert not hasattr(a, 'metastable')
    assert not hasattr(b, 'metastable')


def test_branch_name_is_case_insensitive_and_validated():
    a = _compute(200.0, 150.0, 'caupin2019', branch='HDL')
    b = _compute(200.0, 150.0, 'caupin2019', branch='hdl')
    np.testing.assert_array_equal(a.x, b.x)
    with pytest.raises(ValueError, match='Unknown branch'):
        _compute(200.0, 150.0, 'caupin2019', branch='liquid')


@pytest.mark.parametrize('model', ['IAPWS95', 'water1', 'grenke2025',
                                   'shi_tanaka2020'])
def test_models_without_branches_reject_the_option(model):
    assert _compute(280.0, 0.1, model, branch='stable') is not None
    with pytest.raises(ValueError, match='not available'):
        _compute(280.0, 0.1, model, branch='hdl')


# ---------------------------------------------------------------------------
# One liquid state: exactly one name exists and it is the stable state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model', list(TWO_STATE))
def test_single_state_region(model):
    T = np.array([300.0, 280.0, 260.0, 245.0])
    P = np.array([0.1, 100.0, 200.0])
    s = _compute(T, P, model)
    h = _compute(T, P, model, branch='hdl')
    l = _compute(T, P, model, branch='ldl')
    assert np.all(s.x < 0.5)
    for k in MIX_KEYS:
        np.testing.assert_allclose(getattr(h, k), getattr(s, k), rtol=_tol(k))
        assert np.all(np.isnan(getattr(l, k)))
    assert not h.metastable.any()
    assert not l.metastable.any()
    # pure-state properties do not depend on the branch
    np.testing.assert_allclose(l.rho_A, s.rho_A, rtol=1e-9)
    np.testing.assert_allclose(l.G_B, s.G_B, rtol=1e-9)


def test_low_density_side_beyond_the_critical_point():
    """Below the Widom line at p < p_c the single state is the LDL."""
    T, P = 190.0, 20.0                      # Caupin LLCP: 218.1 K, 71.9 MPa
    s = _compute(T, P, 'caupin2019')
    h = _compute(T, P, 'caupin2019', branch='hdl')
    l = _compute(T, P, 'caupin2019', branch='ldl')
    assert s.x[0, 0] > 0.5
    np.testing.assert_allclose(l.x, s.x, rtol=1e-9)
    np.testing.assert_allclose(l.G, s.G, rtol=1e-12)
    assert np.isnan(h.x[0, 0]) and np.isnan(h.rho[0, 0])
    assert not l.metastable.any()


# ---------------------------------------------------------------------------
# Two liquid states: transition, metastable ranges, spinodals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model', list(TWO_STATE))
def test_branches_cross_at_the_binodal(model):
    ln = _lines(model)
    dT = 0.05
    above = dict(T=ln['T_bin'] + dT, P=ln['p'])
    below = dict(T=ln['T_bin'] - dT, P=ln['p'])
    for pt, hdl_meta in ((above, False), (below, True)):
        h = _compute(pt['T'], pt['P'], model, branch='hdl')
        l = _compute(pt['T'], pt['P'], model, branch='ldl')
        s = _compute(pt['T'], pt['P'], model)
        assert h.x[0, 0] < 0.5 < l.x[0, 0]
        assert bool(h.metastable[0, 0]) is hdl_meta
        assert bool(l.metastable[0, 0]) is (not hdl_meta)
        # the default result is the branch with the lower Gibbs energy
        stable = l if hdl_meta else h
        assert stable.G[0, 0] <= min(h.G[0, 0], l.G[0, 0])
        for k in MIX_KEYS:
            np.testing.assert_allclose(getattr(s, k), getattr(stable, k),
                                       rtol=10 * _tol(k))
    # G_hdl - G_ldl changes sign across the package's binodal temperature
    Ts = np.array([above['T'], below['T']])
    dG = (_compute(Ts, ln['p'], model, branch='hdl').G
          - _compute(Ts, ln['p'], model, branch='ldl').G)[0]
    assert dG[0] < 0.0 < dG[1]


@pytest.mark.parametrize('model', list(TWO_STATE))
def test_hdl_is_smooth_through_the_transition(model):
    ln = _lines(model)
    lo = max(ln['T_hdl_spin'] + 1.0, ln['T_bin'] - 6.0)
    T = np.linspace(ln['T_bin'] + 6.0, lo, 241)
    s = _compute(T, ln['p'], model)
    h = _compute(T, ln['p'], model, branch='hdl')
    assert np.abs(np.diff(s.x[0])).max() > 0.2        # default jumps
    for k in ('x', 'rho', 'G', 'S'):
        v = getattr(h, k)[0]
        assert np.all(np.isfinite(v))
        d = np.abs(np.diff(v))
        # no step much larger than its neighbors
        assert d.max() < 5.0 * np.median(d) + 1e-12, k
    assert np.abs(np.diff(h.x[0])).max() < 0.02
    # stable above the binodal, metastable below (the reference binodal
    # temperature is itself only converged to about 1e-3 K)
    clear = np.abs(T - ln['T_bin']) > 0.01
    np.testing.assert_array_equal(h.metastable[0][clear],
                                  (T < ln['T_bin'])[clear])


@pytest.mark.parametrize('model', list(TWO_STATE))
def test_branches_end_at_their_spinodals(model):
    ln = _lines(model)
    p = ln['p']
    # HDL: exists just above its spinodal, not below
    h = _compute(np.array([ln['T_hdl_spin'] + 0.05, ln['T_hdl_spin'] - 0.05]),
                 p, model, branch='hdl')
    assert np.isfinite(h.x[0, 0]) and h.metastable[0, 0]
    for k in MIX_KEYS:
        assert np.isnan(getattr(h, k)[0, 1]), k
    assert np.isfinite(h.rho_A[0, 1]) and np.isfinite(h.rho_B[0, 1])
    assert not h.metastable[0, 1]
    # LDL: exists just below its spinodal, not above
    l = _compute(np.array([ln['T_ldl_spin'] + 0.05, ln['T_ldl_spin'] - 0.05]),
                 p, model, branch='ldl')
    assert np.isnan(l.x[0, 0])
    assert np.isfinite(l.x[0, 1]) and l.metastable[0, 1]
    # below the HDL spinodal the default result is the LDL
    T = ln['T_hdl_spin'] - 0.05
    np.testing.assert_allclose(_compute(T, p, model, branch='ldl').x,
                               _compute(T, p, model).x, rtol=1e-9)


@pytest.mark.parametrize('model', list(TWO_STATE))
@pytest.mark.parametrize('branch', ['hdl', 'ldl'])
def test_thermodynamic_consistency_on_metastable_branch(model, branch):
    """V = dG/dp and S = -dG/dT hold along a metastable branch.

    The entropy identity is checked on the mixing part, G - G_A and
    S - S_A, which is everything the branch choice affects.  (In the Duska
    model the state-A Gibbs energy comes from a quadrature, so S_A and
    -dG_A/dT differ by up to a few percent on every branch, the default
    included; subtracting state A keeps that out of this test.)
    """
    ln = _lines(model)
    if branch == 'hdl':     # metastable between the binodal and the HDL spinodal
        T = ln['T_bin'] - 0.4 * (ln['T_bin'] - max(ln['T_hdl_spin'], ln['T_bin'] - 10.0))
    else:                   # metastable between the LDL spinodal and the binodal
        T = ln['T_bin'] + 0.4 * (ln['T_ldl_spin'] - ln['T_bin'])
    p = ln['p']
    o = _compute(T, p, model, branch=branch)
    assert o.metastable[0, 0]
    dT, dp = 1e-3, 1e-3
    oT = _compute(np.array([T - dT, T + dT]), p, model, branch=branch)
    gp = _compute(T, np.array([p - dp, p + dp]), model, branch=branch).G[:, 0]
    g_mix = (oT.G - oT.G_A)[0]
    S_mix_fd = -(g_mix[1] - g_mix[0]) / (2 * dT)
    V_fd = (gp[1] - gp[0]) / (2 * dp * 1e6)
    np.testing.assert_allclose(o.S[0, 0] - o.S_A[0, 0], S_mix_fd,
                               rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(o.V[0, 0], V_fd, rtol=2e-6)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def test_solver_roots_satisfy_equilibrium_condition():
    rng = np.random.default_rng(0)
    L = rng.uniform(-1.5, 1.5, 4000)
    om = rng.uniform(0.5, 4.0, 4000)
    x_lo, x_hi = solve_branches(L, om)
    for x in (x_lo, x_hi):
        ok = np.isfinite(x)
        F = L[ok] + np.log(x[ok] / (1 - x[ok])) + om[ok] * (1 - 2 * x[ok])
        assert np.abs(F).max() < 1e-11
        # a minimum of g, not the maximum between the two minima
        assert np.all(1.0 / (x[ok] * (1 - x[ok])) - 2 * om[ok] >= -1e-9)
    assert np.all(x_lo[np.isfinite(x_lo)] <= 0.5)
    assert np.all(x_hi[np.isfinite(x_hi)] >= 0.5)
    # at least one liquid exists everywhere; both only where omega > 2
    assert np.all(np.isfinite(x_lo) | np.isfinite(x_hi))
    both = np.isfinite(x_lo) & np.isfinite(x_hi)
    assert both.any() and np.all(om[both] > 2.0)
    # symmetry of the regular solution: x_hdl(L) = 1 - x_ldl(-L)
    y_lo, y_hi = solve_branches(-L, om)
    np.testing.assert_allclose(x_lo, 1.0 - y_hi, rtol=0, atol=1e-12)


def test_solver_handles_spinodal_and_nonfinite_input():
    om = np.array([3.0, 3.0, np.nan, 3.0])
    xm = 0.5 * (1 - np.sqrt(1 - 2 / 3.0))
    L_spin = -(np.log(xm / (1 - xm)) + 3.0 * (1 - 2 * xm))   # F(x_minus) = 0
    L = np.array([L_spin, L_spin - 1e-6, 0.1, np.inf])
    x_lo, x_hi = solve_branches(L, om)
    assert abs(x_lo[0] - xm) < 1e-6          # double root at the spinodal
    assert np.isnan(x_lo[1])                 # just beyond it
    assert np.isnan(x_lo[2]) and np.isnan(x_hi[2])
    assert np.isnan(x_lo[3]) and np.isnan(x_hi[3])
    x, meta = select_branch(L, om, 'stable')
    assert np.isfinite(x[:2]).all() and not meta.any()


# ---------------------------------------------------------------------------
# Interfaces and backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model', list(TWO_STATE) + ['caupin2019_kim'])
def test_grid_and_scatter_agree(model):
    P = np.array([120.0, 180.0])
    T = np.array([230.0, 205.0, 195.0])
    grid = getProp(np.array([P, T], dtype=object), model, branch='hdl')
    assert grid.x.shape == (2, 3) and grid.metastable.shape == (2, 3)
    assert grid.metastable.dtype == bool
    PT = np.empty(6, dtype=object)
    for i, (p, t) in enumerate((p, t) for p in P for t in T):
        PT[i] = (p, t)
    scat = getProp(PT, model, branch='hdl')
    assert scat.x.shape == (6,) and scat.metastable.shape == (6,)
    np.testing.assert_allclose(scat.x, grid.x.ravel(), rtol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(scat.metastable, grid.metastable.ravel())


def _backends(model):
    rs = pytest.importorskip('watereos_rs')
    name = {'holten2014': 'compute_batch_holten',
            'caupin2019': 'compute_batch_caupin',
            'duska2020': 'compute_batch_duska'}[model]
    core = importlib.import_module(f'{TWO_STATE[model][0]}.core')
    return getattr(rs, name), core.compute_batch


@pytest.mark.parametrize('model', list(TWO_STATE))
@pytest.mark.parametrize('branch', ['hdl', 'ldl'])
def test_rust_matches_numpy(model, branch):
    rust, numpy_batch = _backends(model)
    T, P = np.meshgrid(np.arange(290.0, 150.0, -2.5),
                       np.array([-50.0, 0.1, 40.0, 75.0, 110.0, 160.0, 200.0]))
    T, P = T.ravel(), P.ravel()
    try:
        r = rust(T, P, branch=branch)
    except TypeError:
        pytest.skip('installed watereos_rs predates branch selection')
    with np.errstate(all='ignore'):
        n = numpy_batch(T, P, branch=branch)
    assert sorted(r.keys()) == sorted(n.keys())
    np.testing.assert_array_equal(np.asarray(r['metastable']), n['metastable'])
    np.testing.assert_array_equal(np.isnan(r['x']), np.isnan(n['x']))
    assert np.isnan(n['x']).any() and np.isfinite(n['x']).any()
    assert n['metastable'].any()
    # Second-derivative properties are large near a spinodal and, in the
    # Duska model, partly come from finite differences, so they agree
    # between the backends less tightly than the first-derivative ones.
    for k in n:
        if k == 'metastable':
            continue
        second = k.split('_')[0] in ('Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel')
        np.testing.assert_allclose(np.asarray(r[k]), n[k],
                                   rtol=1e-6 if second else 1e-9,
                                   atol=1e-12, equal_nan=True, err_msg=k)


@pytest.mark.parametrize('model', list(TWO_STATE))
def test_rust_default_has_no_metastable_key(model):
    rust, numpy_batch = _backends(model)
    T = np.array([250.0, 190.0]); P = np.array([0.1, 150.0])
    assert 'metastable' not in rust(T, P)
    assert 'metastable' not in numpy_batch(T, P)
    try:
        assert 'metastable' not in rust(T, P, branch='stable')
    except TypeError:
        pytest.skip('installed watereos_rs predates branch selection')


def test_numpy_fallback_when_backend_lacks_branch():
    """A backend without the keyword (JAX, older Rust build) falls back."""
    from watereos.two_state_eos import getProp as shared_getProp
    from caupin_eos.core import compute_batch as numpy_batch

    def old_backend(T_K, p_MPa):
        return numpy_batch(T_K, p_MPa)

    PT = np.array([np.array([150.0]), np.array([210.0, 190.0])], dtype=object)
    out = shared_getProp(PT, old_backend, branch='hdl', numpy_batch=numpy_batch)
    ref = getProp(PT, 'caupin2019', branch='hdl')
    np.testing.assert_allclose(out.x, ref.x, rtol=1e-12)
    np.testing.assert_array_equal(out.metastable, ref.metastable)
    with pytest.raises(TypeError):
        shared_getProp(PT, old_backend, branch='hdl')


# ---------------------------------------------------------------------------
# Transport model follows the branch of its backbone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model, backbone', [
    ('singh2017', 'holten2014'),
    ('singh2017_caupin2019', 'caupin2019'),
    ('singh2017_duska2020', 'duska2020'),
])
def test_transport_follows_branch(model, backbone):
    ln = _lines(backbone)
    lo = max(ln['T_hdl_spin'] + 1.0, ln['T_bin'] - 6.0)
    T = np.linspace(ln['T_bin'] + 6.0, lo, 121)
    tr = _compute(T, ln['p'], model, branch='hdl')
    eos = _compute(T, ln['p'], backbone, branch='hdl')
    np.testing.assert_allclose(tr.f, eos.x, rtol=1e-12)
    np.testing.assert_array_equal(tr.metastable, eos.metastable)
    assert tr.metastable.any() and not tr.metastable.all()
    for k in ('eta', 'D', 'tau_r'):
        v = np.log(getattr(tr, k)[0])
        assert np.all(np.isfinite(v))
        d = np.abs(np.diff(v))
        assert d.max() < 5.0 * np.median(d) + 1e-12, k
    # the default follows the stable state and jumps at the transition
    st = _compute(T, ln['p'], model)
    assert not hasattr(st, 'metastable')
    assert np.abs(np.diff(np.log(st.D[0]))).max() > 10 * np.abs(np.diff(np.log(tr.D[0]))).max()
