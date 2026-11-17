"""
Tests for the Biddle et al. (2017) two-structure equation of state of
TIP4P/2005 model water with a liquid-vapor spinodal
(model_water_eos.biddle2017), with the 2018 erratum applied.

Check values come from J. Chem. Phys. 146, 034502 (2017) (Table I, Figs. 1,
4, 5, 8, 9), its supplementary material (Fig. S1, Fig. S2 and its caption)
and the erratum, J. Chem. Phys. 148, 069901 (2018).  Figure readings carry
tolerances that reflect the reading accuracy.
"""

import numpy as np
import pytest

from model_water_eos import biddle2017 as mod

KEY = 'biddle2017_tip4p2005'

ALL_KEYS = ('rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv', 'Kt', 'Ks',
            'alpha', 'vel')


def calc(T, P, branch='stable'):
    T = np.atleast_1d(np.asarray(T, dtype=float))
    P = np.atleast_1d(np.asarray(P, dtype=float))
    T, P = np.broadcast_arrays(T, P)
    return mod.compute_batch(np.array(T), np.array(P), branch=branch)


# ---------------------------------------------------------------------------
# Registration and printed parameters
# ---------------------------------------------------------------------------

def test_registration():
    assert mod.MODELS == {KEY: mod.compute_batch}
    assert KEY in mod.BRANCH_MODELS


def test_table_I_parameters_with_erratum():
    assert (mod.Tc, mod.Pc, mod.RHO_C) == (182.0, 170e6, 1017.0)
    assert (mod.LAMBDA, mod.A, mod.B, mod.D, mod.F, mod.W0) == (
        1.55607, 0.154014, 0.125093, 0.00854418, 1.14576, 0.03)
    assert (mod.A0, mod.A1) == (-0.0547873, -0.0822462)
    assert mod.S0 == -5.40845
    # Erratum: 5.56087 and -2.5205 replace 0.0305542 and -7.61e-5
    assert (mod.S1, mod.S2) == (5.56087, -2.5205)
    assert mod.S1 / mod.Tc == pytest.approx(0.0305542, rel=1e-5)
    assert mod.S2 / mod.Tc ** 2 == pytest.approx(-7.61e-5, rel=1e-3)
    assert mod.C[(2, 0)] == -6.30589
    assert mod.C[(4, 0)] == -0.256674
    assert mod.C[(1, 3)] == -0.000994166
    assert len(mod.C) == 12           # 11 printed + the derived c01
    # c01 is not printed; it is fixed by rho = rho_c at the critical point
    assert mod.C01 == pytest.approx(1.096205, abs=1e-6)
    assert mod.P_SCALE / 1e6 == pytest.approx(85.425, abs=1e-3)


# ---------------------------------------------------------------------------
# Printed check values
# ---------------------------------------------------------------------------

def test_critical_point_table_I():
    """LLCP 182 K, 170 MPa, 1017 kg/m3 (Table I, fixed input).  The density
    holds by the c01 closure, so only x = 1/2, G_BA = 0 and w = 2 are
    independent checks here."""
    GA, dG, W = mod.terms(np.array([mod.Tc]), np.array([mod.Pc]))
    assert dG[0][0] == pytest.approx(0.0, abs=1e-9)
    assert W[0][0] / (mod.R * mod.Tc) == pytest.approx(2.0, abs=1e-12)
    cp = mod.critical_point()
    assert (cp['T_K'], cp['p_MPa']) == (182.0, 170.0)
    assert cp['rho'] == pytest.approx(1017.0, abs=0.01)
    out = calc(mod.Tc, mod.Pc / 1e6)
    assert out['x'][0] == pytest.approx(0.5, abs=1e-4)
    # Approaching the critical point along the critical isobar
    kt = 1.0 / calc([200.0, 190.0, 185.0, 183.0], 170.0)['Kt']
    assert np.all(np.diff(kt) > 0)


def test_two_liquids_below_critical_point_only():
    jumps = []
    for T in (165.0, 175.0, 180.0):
        P = mod.llt_pressure(T)
        hdl = calc(T, P, 'hdl')
        ldl = calc(T, P, 'ldl')
        assert hdl['G'][0] == pytest.approx(ldl['G'][0], rel=1e-12)
        assert hdl['x'][0] < 0.5 < ldl['x'][0]
        jumps.append(hdl['rho'][0] - ldl['rho'][0])
    assert jumps[0] > jumps[1] > jumps[2] > 0.0
    T = 190.0
    P = np.linspace(mod.llt_pressure(T) - 20, mod.llt_pressure(T) + 20, 401)
    hdl = calc(np.full_like(P, T), P, 'hdl')
    ldl = calc(np.full_like(P, T), P, 'ldl')
    both = np.isfinite(hdl['rho']) & np.isfinite(ldl['rho'])
    assert np.allclose(hdl['rho'][both], ldl['rho'][both], atol=1e-6)


def test_spinodal_230K_supplement():
    """Supplement, Fig. S2 caption: spinodal pressure -351 MPa at 230 K."""
    Ps = float(mod.spinodal_pressure(230.0))
    assert Ps == pytest.approx(-351.0, abs=1.0)      # computed -351.7
    out = calc([230.0, 230.0], [Ps + 0.05, Ps - 0.05])
    assert np.isfinite(out['rho'][0]) and np.isnan(out['rho'][1])
    # kappa_T diverges on approach as (P - P_s)^(-1/2) (Eq. 8)
    Kt = calc(np.full(4, 230.0), Ps + np.array([10.0, 1.0, 0.1, 0.01]))['Kt']
    assert np.all(np.diff(Kt) < 0)
    assert Kt[3] / Kt[2] == pytest.approx(np.sqrt(0.1), rel=0.01)


@pytest.mark.parametrize('T, Ps_read', [
    (320.0, -226.0), (280.0, -265.0), (240.0, -330.0),
])
def test_spinodal_fig9(T, Ps_read):
    """Fig. 9 dashed spinodal (300 dpi figure reading, +-6 MPa)."""
    assert float(mod.spinodal_pressure(T)) == pytest.approx(Ps_read, abs=6.0)


def test_nan_below_spinodal():
    T = np.array([230.0, 300.0, 200.0])
    P = mod.spinodal_pressure(T) - np.array([0.5, 5.0, 50.0])
    for branch in ('stable', 'hdl', 'ldl'):
        out = calc(T, P, branch)
        for key in ALL_KEYS:
            assert np.all(np.isnan(out[key])), (branch, key)
            if branch == 'stable':
                assert np.all(np.isnan(out[key + '_A'])), key
                assert np.all(np.isnan(out[key + '_B'])), key


def test_supplement_fig_S2_100MPa_isobar():
    """Fig. S2, left column (figure reading)."""
    T = np.arange(180.0, 320.01, 0.05)
    out = calc(T, 100.0)
    kt = 1.0 / out['Kt']
    i = np.nanargmax(kt)
    assert T[i] == pytest.approx(203.0, abs=2.0)
    assert kt[i] == pytest.approx(0.00139, rel=0.03)
    j = np.nanargmax(out['Cp'])
    assert T[j] == pytest.approx(200.0, abs=2.0)
    assert out['Cp'][j] == pytest.approx(7900.0, abs=150.0)
    k = np.nanargmax(out['rho'])
    assert T[k] == pytest.approx(258.0, abs=5.0)
    assert out['rho'][k] == pytest.approx(1046.0, abs=3.0)
    assert out['rho'][0] == pytest.approx(950.0, abs=3.0)
    # dashed background: hypothetical pure structure A
    assert out['rho_A'][0] == pytest.approx(1141.0, abs=3.0)
    assert out['rho_A'][-1] == pytest.approx(1085.0, abs=3.0)


def test_supplement_fig_S2_230K_isotherm():
    """Fig. S2, right column (figure reading)."""
    P = np.arange(-345.0, 300.01, 0.5)
    out = calc(np.full_like(P, 230.0), P)
    kt = 1.0 / out['Kt']
    hi = P > -100.0
    i = np.nanargmax(np.where(hi, kt, np.nan))
    assert P[i] == pytest.approx(18.0, abs=15.0)
    assert kt[i] == pytest.approx(0.00070, rel=0.04)
    lo = (P > -300.0) & (P < -100.0)
    m = np.nanargmin(np.where(lo, kt, np.nan))
    assert kt[m] == pytest.approx(0.00033, rel=0.06)
    cp_hi = np.nanargmax(np.where(hi | (P > -150.0), out['Cp'], np.nan))
    assert P[cp_hi] == pytest.approx(-32.0, abs=12.0)
    assert out['Cp'][cp_hi] == pytest.approx(6560.0, abs=100.0)
    cp_lo = np.nanargmin(np.where(lo, out['Cp'], np.nan))
    assert P[cp_lo] == pytest.approx(-230.0, abs=30.0)
    assert out['Cp'][cp_lo] == pytest.approx(5650.0, abs=100.0)
    end = calc([230.0, 230.0], [300.0, 300.0])
    assert end['rho'][0] == pytest.approx(1130.0, abs=4.0)
    assert end['rho_A'][0] == pytest.approx(1185.0, abs=4.0)


@pytest.mark.parametrize('T, P, x_read', [
    (180.0, -150.0, 0.91), (180.0, 0.1, 0.87), (180.0, 150.0, 0.74),
    (320.0, -150.0, 0.26), (320.0, 0.1, 0.175), (320.0, 150.0, 0.11),
    (200.0, -150.0, 0.855), (230.0, 0.0, 0.49), (260.0, -150.0, 0.515),
    (300.0, -150.0, 0.32), (200.0, 300.0, 0.125), (300.0, 300.0, 0.08),
])
def test_supplement_fig_S1_fraction(T, P, x_read):
    """Fig. S1 (figure reading, +-0.015).  The Ĝ_BA = G_BA/(R T) reading of
    Eq. 4 would give 0.16, 0.08, 0.04 at 320 K."""
    assert calc(T, P)['x'][0] == pytest.approx(x_read, abs=0.015)


@pytest.mark.parametrize('P, T_max, cp_max', [
    (-170.0, 257.0, 6480.0),
    (-125.0, 248.0, 6450.0),
    (-80.0, 240.0, 6480.0),
    (-40.0, 230.0, 6550.0),
])
def test_cp_maxima_fig8(P, T_max, cp_max):
    """Fig. 8 TSEOS curves (figure reading, +-3 K, +-80 J/(kg K))."""
    T = np.arange(200.0, 320.01, 0.05)
    cp = calc(T, P)['Cp']
    i = np.nanargmax(cp)
    assert T[i] == pytest.approx(T_max, abs=3.0)
    assert cp[i] == pytest.approx(cp_max, abs=80.0)


@pytest.mark.parametrize('P, T_max, kt_max', [
    (0.1, 230.0, 0.00068),
    (-40.0, 241.0, 0.00060),
    (100.0, 203.0, 0.00139),
])
def test_kappa_T_maxima_fig5(P, T_max, kt_max):
    """Fig. 5 TSEOS curves (figure reading, +-3 K, +-4 %)."""
    T = np.arange(190.0, 300.01, 0.05)
    kt = 1.0 / calc(T, P)['Kt']
    i = np.nanargmax(kt)
    assert T[i] == pytest.approx(T_max, abs=3.0)
    assert kt[i] == pytest.approx(kt_max, rel=0.04)


def test_isochores_figs_1_and_4():
    """Figs. 1 and 4 TSEOS isochores (figure reading, +-5 MPa)."""
    readings = [  # (T K, P MPa, rho kg/m3)
        (300.0, -146.0, 920.0),
        (300.0, 528.0, 1160.0),
        (300.0, 8.0, 1000.0),
        (200.0, -300.0, 875.0),
        (300.0, -137.0, 925.0),
    ]
    T, P, rho = (np.array(c) for c in zip(*readings))
    out = calc(T, P)
    kt = 1.0 / out['Kt']
    # density error implied by a 5 MPa reading error
    assert np.all(np.abs(out['rho'] - rho) <= rho * kt * 5.0 + 0.5)


def test_density_0p1MPa_fig2():
    """Fig. 2, 0.1 MPa isobar (figure reading): maximum about 1000 kg/m3
    near 275 K."""
    T = np.arange(250.0, 300.0, 0.05)
    out = calc(T, 0.1)
    i = np.nanargmax(out['rho'])
    assert T[i] == pytest.approx(275.0, abs=4.0)
    assert out['rho'][i] == pytest.approx(1000.0, abs=2.0)


# ---------------------------------------------------------------------------
# Thermodynamic consistency on every branch
# ---------------------------------------------------------------------------

def _p_llt(T):
    return float(mod.llt_pressure(T))


CONSISTENCY_POINTS = {
    'stable': [(250.0, 50.0), (200.0, 160.0), (290.0, 300.0),
               (230.0, -60.0), (300.0, -200.0), (240.0, -300.0),
               (185.0, 100.0)],
    'hdl': [(260.0, 0.1), (230.0, 100.0), (300.0, -200.0),
            (170.0, _p_llt(170.0) - 4.0)],     # metastable HDL
    'ldl': [(200.0, 0.1), (220.0, -150.0), (190.0, 100.0),
            (170.0, _p_llt(170.0) + 4.0)],     # metastable LDL
}


@pytest.mark.parametrize('branch', ['stable', 'hdl', 'ldl'])
def test_finite_difference_consistency(branch):
    pts = np.array(CONSISTENCY_POINTS[branch])
    T, P = pts[:, 0], pts[:, 1]
    h, k = 0.002, 0.002                   # K, MPa (small: metastable points)
    c = calc(T, P, branch)
    for key in ('G', 'S', 'V', 'Cp', 'Kt', 'alpha'):
        assert np.all(np.isfinite(c[key])), (branch, key)
    tp, tm = calc(T + h, P, branch), calc(T - h, P, branch)
    pp, pm = calc(T, P + k, branch), calc(T, P - k, branch)
    S_fd = -(tp['G'] - tm['G']) / (2 * h)
    V_fd = (pp['G'] - pm['G']) / (2 * k * 1e6)
    Cp_fd = T * (tp['S'] - tm['S']) / (2 * h)
    Kt_fd = -c['V'] * 2 * k / (pp['V'] - pm['V'])
    alpha_fd = (tp['V'] - tm['V']) / (2 * h * c['V'])
    # S and alpha keep an arbitrary zero or change sign: absolute floors
    np.testing.assert_allclose(S_fd, c['S'], rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(V_fd, c['V'], rtol=1e-6)
    np.testing.assert_allclose(Cp_fd, c['Cp'], rtol=1e-6)
    np.testing.assert_allclose(Kt_fd, c['Kt'], rtol=1e-6)
    np.testing.assert_allclose(alpha_fd, c['alpha'], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(c['H'], c['G'] + T * c['S'], rtol=1e-12)
    np.testing.assert_allclose(c['rho'] * c['V'], 1.0, rtol=1e-12)


def test_spinodal_term_derivatives():
    """The analytic derivatives of A(T) (p - p_s(T))^(3/2) against central
    differences, near and far from the spinodal."""
    T = np.array([230.0, 300.0, 200.0, 260.0])
    P = (mod.spinodal_pressure(T) + np.array([2.0, 50.0, 300.0, 600.0])) * 1e6
    h, k = 1e-3, 1e2                      # K, Pa
    f = mod.spinodal_field(T, P)
    ft = (mod.spinodal_field(T + h, P), mod.spinodal_field(T - h, P))
    fp = (mod.spinodal_field(T, P + k), mod.spinodal_field(T, P - k))
    np.testing.assert_allclose((ft[0][0] - ft[1][0]) / (2 * h), f[1],
                               rtol=1e-6)
    np.testing.assert_allclose((fp[0][0] - fp[1][0]) / (2 * k), f[2],
                               rtol=1e-6)
    np.testing.assert_allclose((ft[0][1] - ft[1][1]) / (2 * h), f[3],
                               rtol=1e-6)
    np.testing.assert_allclose((fp[0][1] - fp[1][1]) / (2 * k), f[4],
                               rtol=1e-6)
    np.testing.assert_allclose((ft[0][2] - ft[1][2]) / (2 * h), f[4],
                               rtol=1e-6)
    np.testing.assert_allclose((fp[0][2] - fp[1][2]) / (2 * k), f[5],
                               rtol=1e-6)


def test_branches_agree_with_stable_where_stable():
    T = np.array([260.0, 200.0, 300.0])
    P = np.array([0.1, 0.1, -200.0])
    st = calc(T, P)
    hdl = calc(T, P, 'hdl')
    ldl = calc(T, P, 'ldl')
    assert np.isfinite(hdl['rho'][0]) and np.isnan(ldl['rho'][0])
    assert np.isfinite(ldl['rho'][1]) and np.isnan(hdl['rho'][1])
    np.testing.assert_allclose(hdl['rho'][[0, 2]], st['rho'][[0, 2]])
    np.testing.assert_allclose(ldl['rho'][1], st['rho'][1])


def test_unknown_branch_rejected():
    with pytest.raises(ValueError):
        calc(250.0, 0.1, 'vapor')


# ---------------------------------------------------------------------------
# Registry probe and speed
# ---------------------------------------------------------------------------

def test_registry_probe_finite():
    out = calc(273.15, 0.1)
    for key in ALL_KEYS + ('x',):
        assert np.isfinite(out[key][0]), key
    assert out['rho'][0] == pytest.approx(1000.1, abs=0.1)


def test_vectorized_speed():
    import time
    rng = np.random.default_rng(1)
    T = rng.uniform(180.0, 320.0, 10000)
    P = rng.uniform(-200.0, 400.0, 10000)
    t0 = time.perf_counter()
    out = mod.compute_batch(T, P)
    assert time.perf_counter() - t0 < 1.0
    above = P > mod.spinodal_pressure(T)
    assert np.all(np.isfinite(out['rho'][above]))
