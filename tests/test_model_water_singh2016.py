"""
Tests for the Singh et al. (2016) two-structure equation of state of
TIP4P/2005 model water (model_water_eos.singh2016).

Check values come from J. Chem. Phys. 144, 144504 (2016): Table I, the text
of p. 144504-8 and figure readings (Figs. 1, 3(b), 6), plus the dotted
extrapolations of this equation drawn in Fig. 3 of Biddle et al. (2017).
Figure readings carry tolerances that reflect the reading accuracy.
"""

import numpy as np
import pytest

from model_water_eos import singh2016 as mod

KEY = 'singh2016_tip4p2005'


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


def test_table_I_parameters():
    assert (mod.Tc, mod.Pc, mod.RHO_C) == (182.0, 170e6, 1017.0)
    assert (mod.LAMBDA, mod.A, mod.B, mod.W0) == (1.407, 0.171, -0.100,
                                                   0.0717)
    assert mod.C[(2, 0)] == -5.3481
    assert mod.C[(4, 0)] == -0.13173
    assert mod.C[(3, 3)] == -0.003513
    assert len(mod.C) == 13
    # rho_c R Tc with M = 18.015268 g/mol
    assert mod.P_SCALE / 1e6 == pytest.approx(85.425, abs=1e-3)


# ---------------------------------------------------------------------------
# Printed check values
# ---------------------------------------------------------------------------

def test_critical_point_table_I():
    """LLCP 182 K, 170 MPa, 1017 kg/m3 (Table I, p. 144504-8)."""
    GA, dG, W = mod.terms(np.array([mod.Tc]), np.array([mod.Pc]))
    assert dG[0][0] == pytest.approx(0.0, abs=1e-9)
    assert W[0][0] / (mod.R * mod.Tc) == pytest.approx(2.0, abs=1e-12)
    cp = mod.critical_point()
    assert cp['T_K'] == 182.0 and cp['p_MPa'] == 170.0
    # 1017 / (c01 + lambda a / 2 + w0 / 4) = 1017.08 kg/m3
    assert cp['rho'] == pytest.approx(1017.0, abs=0.5)
    out = calc(mod.Tc, mod.Pc / 1e6)
    assert out['x'][0] == pytest.approx(0.5, abs=1e-4)


def test_two_liquids_below_critical_point_only():
    """Below Tc, HDL and LDL coexist on G_BA = 0 (equal G, different
    densities); the density jump vanishes as T -> Tc."""
    jumps = []
    for T in (165.0, 175.0, 180.0):
        P = mod.llt_pressure(T)
        hdl = calc(T, P, 'hdl')
        ldl = calc(T, P, 'ldl')
        assert hdl['G'][0] == pytest.approx(ldl['G'][0], rel=1e-12)
        assert hdl['x'][0] < 0.5 < ldl['x'][0]
        jumps.append(hdl['rho'][0] - ldl['rho'][0])
    assert jumps[0] > jumps[1] > jumps[2] > 0.0
    assert jumps[1] > 50.0
    # Above Tc: no second liquid across the Widom line
    T = 190.0
    P = np.linspace(mod.llt_pressure(T) - 20, mod.llt_pressure(T) + 20, 401)
    hdl = calc(np.full_like(P, T), P, 'hdl')
    ldl = calc(np.full_like(P, T), P, 'ldl')
    both = np.isfinite(hdl['rho']) & np.isfinite(ldl['rho'])
    assert np.allclose(hdl['rho'][both], ldl['rho'][both], atol=1e-6)


def test_llt_line_fig1():
    """Fig. 1 (figure reading): transition at about 226 MPa at 160 K;
    Widom line crosses 0 MPa near 232 K (computed 233.7 K)."""
    from scipy.optimize import brentq
    assert mod.llt_pressure(160.0) == pytest.approx(226.0, abs=4.0)
    T_w = brentq(lambda T: mod.llt_pressure(T), 200.0, 260.0)
    assert T_w == pytest.approx(232.0, abs=2.5)


def test_isochores_fig1():
    """Fig. 1 TSEOS isochores (figure reading, +-3 MPa)."""
    readings = [  # (T K, P MPa, rho kg/m3)
        (300.0, -82.0, 960.0),
        (300.0, 6.0, 1000.0),
        (300.0, 360.0, 1120.0),
        (260.0, 128.0, 1060.0),
        (240.0, 90.0, 1040.0),
    ]
    T, P, rho = (np.array(c) for c in zip(*readings))
    out = calc(T, P)
    # 3 MPa of reading error is at most about 1.5 kg/m3 here
    assert np.allclose(out['rho'], rho, atol=2.0)


def test_response_maxima_at_ambient_pressure_text():
    """p. 144504-8: the TSEOS predicts within ~5 K the C_P maximum
    (about 220 K) and the kappa_T maximum (about 230 K) at 0.1 MPa."""
    T = np.arange(190.0, 280.0, 0.1)
    out = calc(T, 0.1)
    T_cp = T[np.nanargmax(out['Cp'])]
    T_kt = T[np.nanargmax(1.0 / out['Kt'])]
    assert abs(T_cp - 220.0) <= 5.0       # computed 224.4 K
    assert abs(T_kt - 230.0) <= 5.0       # computed 233.7 K


@pytest.mark.parametrize('P, T_max, kt_max', [
    (120.0, 199.0, 0.00165),
    (100.0, 205.0, 0.00124),
    (70.0, 213.5, 0.00094),
    (40.0, 222.5, 0.00078),
    (0.1, 233.0, 0.00066),
])
def test_kappa_T_maxima_fig3b(P, T_max, kt_max):
    """Fig. 3(b) TSEOS curves (figure reading): +-2 K, +-4 %."""
    T = np.arange(185.0, 280.0, 0.05)
    kt = 1.0 / calc(T, P)['Kt']
    i = np.nanargmax(kt)
    assert T[i] == pytest.approx(T_max, abs=2.0)
    assert kt[i] == pytest.approx(kt_max, rel=0.04)


def test_density_maximum_fig6():
    """Fig. 6 (figure reading): TMD near 277 K at 0.1 MPa."""
    T = np.arange(250.0, 300.0, 0.05)
    out = calc(T, 0.1)
    i = np.nanargmax(out['rho'])
    assert T[i] == pytest.approx(277.0, abs=2.0)
    assert out['rho'][i] == pytest.approx(1001.0, abs=2.0)


def test_negative_pressure_extrapolation_biddle2017_fig3():
    """Biddle et al. (2017), Fig. 3, dotted lines (this equation pushed to
    -300 MPa, figure reading +-5 kg/m3)."""
    out = calc(np.array([300.0, 200.0]), np.array([-300.0, -300.0]))
    assert out['rho'] == pytest.approx([855.0, 890.0], abs=5.0)


# ---------------------------------------------------------------------------
# Thermodynamic consistency on every branch
# ---------------------------------------------------------------------------

def _p_llt(T):
    return float(mod.llt_pressure(T))


CONSISTENCY_POINTS = {
    'stable': [(250.0, 50.0), (200.0, 160.0), (290.0, 300.0),
               (230.0, -60.0), (185.0, 100.0)],
    'hdl': [(260.0, 0.1), (230.0, 100.0), (300.0, 300.0),
            (170.0, _p_llt(170.0) - 4.0)],     # metastable HDL
    'ldl': [(200.0, 0.1), (220.0, -50.0), (190.0, 100.0),
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
    # H = G + T S and the internal consistency of the derived keys
    np.testing.assert_allclose(c['H'], c['G'] + T * c['S'], rtol=1e-12)
    np.testing.assert_allclose(c['rho'] * c['V'], 1.0, rtol=1e-12)


def test_branches_agree_with_stable_where_stable():
    T = np.array([260.0, 200.0, 300.0])
    P = np.array([0.1, 0.1, 300.0])
    st = calc(T, P)
    hdl = calc(T, P, 'hdl')
    ldl = calc(T, P, 'ldl')
    assert np.isfinite(hdl['rho'][0]) and np.isnan(ldl['rho'][0])
    assert np.isfinite(ldl['rho'][1]) and np.isnan(hdl['rho'][1])
    np.testing.assert_allclose(hdl['rho'][[0, 2]], st['rho'][[0, 2]])
    np.testing.assert_allclose(ldl['rho'][1], st['rho'][1])
    assert not hdl['metastable'][0] and not ldl['metastable'][1]


def test_unknown_branch_rejected():
    with pytest.raises(ValueError):
        calc(250.0, 0.1, 'vapor')


# ---------------------------------------------------------------------------
# Registry probe and speed
# ---------------------------------------------------------------------------

def test_registry_probe_finite():
    out = calc(273.15, 0.1)
    for key in ('rho', 'V', 'alpha', 'S', 'G', 'H', 'Cp', 'Cv', 'Kt', 'Ks',
                'vel', 'x'):
        assert np.isfinite(out[key][0]), key
    assert out['rho'][0] == pytest.approx(1000.8, abs=0.1)


def test_vectorized_speed():
    import time
    rng = np.random.default_rng(1)
    T = rng.uniform(180.0, 300.0, 10000)
    P = rng.uniform(-80.0, 400.0, 10000)
    t0 = time.perf_counter()
    out = mod.compute_batch(T, P)
    assert time.perf_counter() - t0 < 1.0
    assert np.all(np.isfinite(out['rho']))
