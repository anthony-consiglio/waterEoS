"""
Thermodynamic consistency of the two-state models: the returned Gibbs
energy must be the potential of the returned entropy and volume,

    S = -(dG/dT)_p        V = (dG/dp)_T

on every liquid branch and in every backend.

Holten (2014) and Caupin (2019) are written as explicit Gibbs energies, so
the identities hold by construction; they are checked here as a guard.
Duska (2020) defines state A through its volume and entropy, and its Gibbs
energy is their closed-form integral (``duska_eos.core.compute_GA``).  An
earlier version integrated the entropy numerically from the vapor-liquid
critical point with one 16-point Gauss-Legendre rule, across the
temperature (597.8 K) where the spinodal pressure of state A changes sign
and the integrand has a square-root singularity.  That left G off by up to
3 J/mol relative to the entropy, with a cusp at 244.05 K where a quadrature
node crossed the singularity.
"""
import importlib

import numpy as np
import pytest

MODELS = ['holten2014', 'caupin2019', 'caupin2019_kim', 'duska2020']
BRANCHES = ['stable', 'hdl', 'ldl']

_PACKAGE = {'holten2014': 'holten_eos', 'caupin2019': 'caupin_eos',
            'caupin2019_kim': 'caupin_eos', 'duska2020': 'duska_eos'}
_RUST_NAME = {'holten2014': 'compute_batch_holten',
              'caupin2019': 'compute_batch_caupin',
              'caupin2019_kim': 'compute_batch_caupin_kim',
              'duska2020': 'compute_batch_duska'}

# Finite-difference steps (five-point stencil) and tolerances.  With these
# steps the truncation and rounding errors of the stencil are below
# 1e-6 J/(kg K) in S and 2e-9 relative in V for all models; the defect
# described above was 0.3 to 670 J/(kg K).
H_T = 0.02          # K
H_P = 0.1           # MPa
S_ATOL = 2e-5       # J/(kg K)
V_RTOL = 2e-8
# A point is used only if the same liquid exists 1 K and 1 MPa to either
# side, which keeps the stencil away from the end of a branch (its
# spinodal) and, for the default branch, off the liquid-liquid transition.
GUARD_T = 1.0       # K
GUARD_P = 1.0       # MPa


def _batch(model, backend):
    """compute_batch(T, P, branch) of one model in one backend."""
    if backend == 'numpy':
        core = importlib.import_module(f'{_PACKAGE[model]}.core')
        if model == 'caupin2019_kim':
            from caupin_eos import params_kim
            return lambda T, P, branch: core.compute_batch(
                T, P, pset=params_kim, branch=branch)
        return lambda T, P, branch: core.compute_batch(T, P, branch=branch)
    if backend == 'rust':
        rs = pytest.importorskip('watereos_rs')
        fn = getattr(rs, _RUST_NAME[model])

        def rust_batch(T, P, branch):
            try:
                return fn(np.ascontiguousarray(T), np.ascontiguousarray(P),
                          branch=branch)
            except TypeError:
                pytest.skip('installed watereos_rs predates branch selection')
        return rust_batch
    if backend == 'jax':
        pytest.importorskip('jax')
        if model == 'caupin2019_kim':
            pytest.skip('the JAX backend has no with-Kim parameter set')
        core_ad = importlib.import_module(f'{_PACKAGE[model]}.core_ad')
        return lambda T, P, branch: core_ad.compute_batch(T, P)
    raise ValueError(backend)


def _grid(backend):
    """Scatter points covering 160-320 K and 0.1-400 MPa.

    The grid includes 244.05 K, where the old Duska Gibbs energy had its
    cusp.  The JAX backend recompiles for every array length, so it gets
    50 points, the length the cross-validation tests already use.
    """
    if backend == 'jax':
        T1 = np.linspace(165.0, 315.0, 9)
        T1 = np.append(T1, 244.05)
        P1 = np.array([0.1, 50.0, 100.0, 200.0, 400.0])
    else:
        T1 = np.concatenate([np.arange(160.0, 320.1, 2.5), [244.05, 273.15]])
        P1 = np.array([0.1, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0,
                       300.0, 400.0])
    T, P = np.meshgrid(T1, P1)
    return T.ravel(), P.ravel()


def _consistency(batch, T, P, branch):
    """S + dG/dT [J/(kg K)], (dG/dp)/V - 1, and the mask of usable points."""
    def run(dT=0.0, dP=0.0):
        with np.errstate(all='ignore'):
            out = batch(T + dT, P + dP, branch)
        return {k: np.asarray(out[k], dtype=float) for k in ('G', 'S', 'V', 'x')}

    o = run()
    side = np.sign(o['x'] - 0.5)
    ok = np.isfinite(o['x']) & np.isfinite(o['G']) & np.isfinite(o['S'])
    for dT, dP in ((GUARD_T, 0.0), (-GUARD_T, 0.0),
                   (0.0, GUARD_P), (0.0, -GUARD_P)):
        x = run(dT, dP)['x']
        with np.errstate(invalid='ignore'):
            ok &= np.isfinite(x) & (np.sign(x - 0.5) == side)

    def G(dT=0.0, dP=0.0):
        return run(dT, dP)['G']

    dG_dT = (-G(2 * H_T) + 8 * G(H_T) - 8 * G(-H_T) + G(-2 * H_T)) / (12 * H_T)
    dG_dp = (-G(0, 2 * H_P) + 8 * G(0, H_P) - 8 * G(0, -H_P)
             + G(0, -2 * H_P)) / (12 * H_P * 1e6)
    with np.errstate(all='ignore'):
        return o['S'] + dG_dT, dG_dp / o['V'] - 1.0, ok


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('branch', BRANCHES)
@pytest.mark.parametrize('backend', ['numpy', 'rust'])
def test_gibbs_energy_is_the_potential_of_S_and_V(model, branch, backend):
    batch = _batch(model, backend)
    T, P = _grid(backend)
    res_S, res_V, ok = _consistency(batch, T, P, branch)
    # the grid must really exercise the branch, on both sides of 244 K for
    # the liquids that exist there
    assert ok.sum() > 150
    assert T[ok].min() < 185.0
    if branch != 'ldl':
        assert T[ok].max() == 320.0 and P[ok].max() == 400.0
        assert np.any(ok & (T == 244.05))
    i = np.argmax(np.abs(res_S[ok]))
    assert np.abs(res_S[ok]).max() < S_ATOL, (
        f'S + dG/dT = {res_S[ok][i]:.3e} J/(kg K) at '
        f'{T[ok][i]:.2f} K, {P[ok][i]:.1f} MPa')
    j = np.argmax(np.abs(res_V[ok]))
    assert np.abs(res_V[ok]).max() < V_RTOL, (
        f'(dG/dp)/V - 1 = {res_V[ok][j]:.3e} at '
        f'{T[ok][j]:.2f} K, {P[ok][j]:.1f} MPa')


@pytest.mark.parametrize('model', ['holten2014', 'caupin2019', 'duska2020'])
def test_gibbs_energy_is_the_potential_of_S_and_V_jax(model):
    batch = _batch(model, 'jax')
    T, P = _grid('jax')
    res_S, res_V, ok = _consistency(batch, T, P, 'stable')
    assert ok.sum() > 40
    assert np.abs(res_S[ok]).max() < S_ATOL
    assert np.abs(res_V[ok]).max() < V_RTOL


@pytest.mark.parametrize('suffix', ['_A', '_B'])
def test_duska_pure_states_are_consistent(suffix):
    """The identities also hold for state A and state B separately."""
    from duska_eos.core import compute_batch
    T, P = _grid('numpy')

    def G(dT=0.0, dP=0.0):
        return compute_batch(T + dT, P + dP)['G' + suffix]

    o = compute_batch(T, P)
    dG_dT = (-G(2 * H_T) + 8 * G(H_T) - 8 * G(-H_T) + G(-2 * H_T)) / (12 * H_T)
    dG_dp = (-G(0, 2 * H_P) + 8 * G(0, H_P) - 8 * G(0, -H_P)
             + G(0, -2 * H_P)) / (12 * H_P * 1e6)
    assert np.all(np.isfinite(dG_dT)) and np.all(np.isfinite(dG_dp))
    assert np.abs(o['S' + suffix] + dG_dT).max() < S_ATOL
    assert np.abs(dG_dp / o['V' + suffix] - 1.0).max() < V_RTOL


def test_duska_scalar_entry_points_match_batch():
    """compute_properties and compute_properties_at_x use the same G."""
    from duska_eos.core import (compute_batch, compute_properties,
                                compute_properties_at_x)
    for T, p in ((273.15, 0.1), (244.05, 0.1), (205.0, 150.0), (300.0, 400.0)):
        b = compute_batch(np.array([T]), np.array([p]))
        s = compute_properties(T, p)
        for k in ('G', 'H', 'U', 'A', 'G_A', 'G_B', 'S', 'V'):
            assert s[k] == pytest.approx(float(b[k][0]), rel=1e-9, abs=1e-6), k
        at_x = compute_properties_at_x(T, p, float(b['x'][0]))
        for k in ('G', 'H', 'U', 'A', 'S', 'V'):
            assert at_x[k] == pytest.approx(float(b[k][0]), rel=1e-9, abs=1e-6), k
        # S = -dG/dT also holds at a forced, non-equilibrium composition
        dT = 1e-3
        x = 0.3
        g = [compute_properties_at_x(T + d, p, x)['G'] for d in (-dT, dT)]
        S_x = compute_properties_at_x(T, p, x)['S']
        assert S_x == pytest.approx(-(g[1] - g[0]) / (2 * dT), abs=1e-4)


def test_duska_reference_state_alignment():
    """G and S at 273.15 K, 0.1 MPa stay on the IAPWS-95 reference values."""
    from watereos import getProp
    PT = np.array([[0.1], [273.15]], dtype=object)
    duska = getProp(PT, 'duska2020')
    ref = getProp(PT, 'IAPWS95')
    assert float(duska.G.flat[0]) == pytest.approx(100.017518, abs=1e-5)
    assert float(duska.G.flat[0]) == pytest.approx(float(ref.G.flat[0]), abs=1e-5)
    assert float(duska.S.flat[0]) == pytest.approx(float(ref.S.flat[0]), abs=1e-6)
    from duska_eos.core import compute_batch
    numpy_G = float(compute_batch(np.array([273.15]), np.array([0.1]))['G'][0])
    assert numpy_G == pytest.approx(100.017518, abs=1e-5)
