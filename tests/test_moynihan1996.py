"""
Moynihan (1996) two-species / nonideal-solution model.

Checks against the values the paper prints (MRS Proc. 455, 411):

- critical point Tc = W/2R = 217 K (Eq. 22c) and Pc = 380 atm (Eq. 27);
- coexistence at 135 K at 1850 atm (Eq. 20), and the two spinodals at 135 K
  that the constants were fitted to: 500 atm (HDA -> LDA) and 3200 atm
  (LDA -> HDA) (p. 417);
- the temperature-volume phase diagram (Fig. 7): coexistence volumes at the
  labeled pressures, read off the figure at 300 dpi;
- the excess heat capacity and expansivity at 1 atm with dS0/R = 0.98
  (Figs. 5 and 6), read off the figures.

Plus the internal consistency of alpha and Kt with the returned volume, and
of the excess heat capacity with the closed form of Eq. 17.
"""

import numpy as np
import pytest

import moynihan_eos as m
from moynihan_eos import params as P

ATM_MPA = 0.101325


def test_critical_point_matches_printed():
    c = m.critical_point()
    assert c['T_K'] == pytest.approx(217.0, abs=0.5)          # Eq. 22c
    # Eq. 27 prints 380 atm; the printed (rounded) constants give 376 atm.
    assert c['p_MPa'] / ATM_MPA == pytest.approx(380.0, rel=0.015)
    out = m.compute_properties(c['T_K'], c['p_MPa'])
    # g(x) is flat to fourth order at the critical point, so x converges
    # only to about 1e-5 there.
    assert out['x'] == pytest.approx(0.5, abs=1e-4)
    assert out['V'] == pytest.approx(P.V_C, rel=1e-5)        # Fig. 7


def test_transition_at_135K_matches_printed():
    # Eq. 20: Paa = (500 + 3200)/2 = 1850 atm at 135 K
    assert m.coexistence_pressure(135.0) / ATM_MPA == pytest.approx(1850.0, rel=0.01)
    # Eq. 22b: ln K = -dG0/RT = -0.75 at 135 K
    lnK = -(P.DH0 - 135.0 * P.DS0) / (P.R * 135.0)
    assert lnK == pytest.approx(P.LNK_135, abs=0.01)


def test_spinodals_at_135K_match_mishima_pressures():
    p_hdl, p_ldl = m.spinodal_pressures(135.0)
    assert p_hdl / ATM_MPA == pytest.approx(500.0, rel=0.02)
    assert p_ldl / ATM_MPA == pytest.approx(3200.0, rel=0.02)
    # The liquids end at their spinodals: HDL exists just above p_hdl and
    # not just below it; LDL the other way round.
    assert np.isfinite(m.compute_properties(135.0, p_hdl + 0.5, 'hdl')['V'])
    assert np.isnan(m.compute_properties(135.0, p_hdl - 0.5, 'hdl')['V'])
    assert np.isfinite(m.compute_properties(135.0, p_ldl - 0.5, 'ldl')['V'])
    assert np.isnan(m.compute_properties(135.0, p_ldl + 0.5, 'ldl')['V'])


# Fig. 7 readings (T in K, labeled pressure in atm, v of the LDA-side and
# HDA-side branches in cm3/g; None where the figure has no reading).
FIG7 = [
    (135.0, 1850.0, 1.065, 0.843),
    (171.0, 1200.0, 1.070, None),
    (204.5, 600.0, 1.053, None),
]


@pytest.mark.parametrize('T, p_atm, v_ldl, v_hdl', FIG7)
def test_coexistence_volumes_match_fig7(T, p_atm, v_ldl, v_hdl):
    assert m.coexistence_pressure(T) / ATM_MPA == pytest.approx(p_atm, rel=0.01)
    vh, vl = m.coexistence_volumes([T])
    assert vl[0] * 1e3 == pytest.approx(v_ldl, abs=0.006)
    if v_hdl is not None:
        assert vh[0] * 1e3 == pytest.approx(v_hdl, abs=0.006)


def _excess_vs_277K(T):
    """Cp_ex(T) - Cp_ex(277 K) at 1 atm, J/(K mol)."""
    o = m.compute_batch(T, np.full_like(T, ATM_MPA))
    o277 = m.compute_batch(np.array([277.0]), np.array([ATM_MPA]))
    return (o['Cp_ex'] - o277['Cp_ex'][0]) * P.M_H2O


def test_excess_heat_capacity_matches_fig5():
    T = np.linspace(200.0, 280.0, 1601)
    d = _excess_vs_277K(T)
    i = np.argmax(d)
    # Fig. 5, dS0/R = 0.98: maximum about 37.5 J/(K mol) near 231.5 K
    assert T[i] == pytest.approx(231.5, abs=1.5)
    assert d[i] == pytest.approx(37.5, abs=1.5)
    assert d[0] == pytest.approx(17.7, abs=1.5)       # 200 K


def test_excess_expansivity_matches_fig6():
    # V * alpha excess = dV0 * dX/dT (Eq. 18); with the constant background
    # the returned alpha differs from it by alpha_b * v_bkgr only.
    T = np.linspace(200.0, 280.0, 1601)
    o = m.compute_batch(T, np.full_like(T, ATM_MPA))
    o277 = m.compute_batch(np.array([277.0]), np.array([ATM_MPA]))
    crit = m.critical_point()

    def v_bkgr(TT):
        return P.V_C * np.exp(P.ALPHA_BKGR * (TT - crit['T_K'])
                              - P.KAPPA_BKGR * (ATM_MPA - crit['p_MPa']) * 1e6)

    np.testing.assert_allclose(
        o['V_alpha_ex'], o['alpha'] * o['V'] - P.ALPHA_BKGR * v_bkgr(T))
    np.testing.assert_allclose(
        o['V_kappa_ex'], o['V'] / o['Kt'] - P.KAPPA_BKGR * 1e6 * v_bkgr(T))
    d = (o['V_alpha_ex'] - o277['V_alpha_ex']) * P.M_H2O * 1e6
    j = np.argmin(d)
    # Fig. 6: minimum about -0.061 cm3/(K mol) near 233 K
    assert T[j] == pytest.approx(233.0, abs=1.5)
    assert d[j] == pytest.approx(-0.061, abs=0.004)


@pytest.mark.parametrize('branch', ['stable', 'hdl', 'ldl'])
def test_alpha_and_Kt_are_derivatives_of_V(branch):
    T = np.array([150.0, 200.0, 230.0, 260.0, 180.0])
    p = np.array([150.0, 70.0, 0.1, 50.0, 120.0])
    o = m.compute_batch(T, p, branch=branch)
    hT, hP = 1e-3, 1e-3
    VTp = m.compute_batch(T + hT, p, branch=branch)['V']
    VTm = m.compute_batch(T - hT, p, branch=branch)['V']
    VPp = m.compute_batch(T, p + hP, branch=branch)['V']
    VPm = m.compute_batch(T, p - hP, branch=branch)['V']
    ok = np.isfinite(o['V']) & np.isfinite(VTp) & np.isfinite(VTm) \
        & np.isfinite(VPp) & np.isfinite(VPm)
    assert ok.sum() >= 2
    alpha = (VTp - VTm) / (2 * hT) / o['V']
    Kt = -o['V'] / ((VPp - VPm) / (2 * hP))
    np.testing.assert_allclose(alpha[ok], o['alpha'][ok], rtol=1e-6)
    np.testing.assert_allclose(Kt[ok], o['Kt'][ok], rtol=1e-6)


def test_heat_capacity_is_eq17():
    # Eq. 17 as printed, with X the HDA-species fraction
    T = np.array([200.0, 240.0, 270.0])
    p = np.array([0.1, 0.1, 100.0])
    o = m.compute_batch(T, p)
    X = 1.0 - o['x']
    Pa = p * 1e6
    num = (P.DH0 + P.DV0 * (Pa - P.P0) + P.W * (1 - 2 * X)) \
        * (P.DS0 - P.R * np.log(X / (1 - X)))
    den = P.R * T / (X * (1 - X)) - 2 * P.W
    np.testing.assert_allclose(o['Cp_ex'] * P.M_H2O, num / den, rtol=1e-9)


def test_stable_branch_jumps_at_transition():
    T = 180.0
    p_aa = float(m.coexistence_pressure(T))
    below = m.compute_properties(T, p_aa - 1.0)
    above = m.compute_properties(T, p_aa + 1.0)
    assert below['x'] > 0.5 > above['x']
    assert below['rho'] < above['rho']


def test_getprop_grid_shape_and_dispatch():
    import watereos
    PT = np.array([np.array([0.1, 50.0, 100.0]), np.array([200.0, 250.0])],
                  dtype=object)
    out = m.getProp(PT)
    assert out.rho.shape == (3, 2)
    out2 = watereos.getProp(PT, 'moynihan1996species')
    np.testing.assert_allclose(out2.rho, out.rho)
    for key in ('x', 'rho', 'V', 'alpha', 'Kt', 'Cp_ex'):
        assert hasattr(out2, key)
    for key in ('S', 'G', 'H', 'Cp', 'vel'):
        assert not hasattr(out2, key)
