"""
Liquid-liquid coexistence (binodal) and spinodal curves from the EOS-VaT
two-state model.

The Gibbs free energy of mixing is:
    g(x) = x*DG + Th*[x*ln(x) + (1-x)*ln(1-x)] + omega*x*(1-x)

Equilibrium: dg/dx = 0  =>  F(x) = DG + Th*ln(x/(1-x)) + omega*(1-2x) = 0
Spinodal:    d²g/dx² = 0  =>  Th/(x*(1-x)) = 2*omega
LLCP:        x = 1/2,  DG = 0,  Th = omega/2

Reference: M. Duška, "Water above the spinodal",
J. Chem. Phys. 152, 174501 (2020), Fig. 3.
"""

import numpy as np
from scipy.optimize import brentq
from . import params as P
from .core import compute_DeltaG, compute_omega

from watereos.two_state_phase import (
    get_compute_batch as _gcb,
    compute_spinodal_curve as _spinodal,
    compute_binodal_curve as _binodal,
    compute_phase_diagram as _phase_diag,
    compute_tmd_temperature as _tmd,
    compute_kauzmann_temperature as _kauz,
)

# ═══════════════════════════════════════════════════════════════════════════
# Vectorized adapter for fast_phase_diagram
# ═══════════════════════════════════════════════════════════════════════════

def _get_adapter():
    """Return vectorized adapter for the fast_phase_diagram module."""
    llcp = find_LLCP()

    def omega_vec(T_arr, P_arr):
        Th = T_arr / P.T_VLCP
        ph = P_arr / P.p_VLCP
        return P.w[0] * (1.0 + P.w[1] * ph + P.w[2] * Th + P.w[3] * Th * ph)

    def disc_vec(omega, T_arr, P_arr):
        Th = T_arr / P.T_VLCP
        return 1.0 - 2.0 * Th / omega

    def F_eq_vec(x_arr, T_arr, P_arr):
        Th = T_arr / P.T_VLCP
        ph = P_arr / P.p_VLCP
        DG = (P.a[0] + P.a[1] * ph * Th + P.a[2] * ph + P.a[3] * Th
              + P.a[4] * Th**2 + P.a[5] * ph**2 + P.a[6] * ph**3)
        om = P.w[0] * (1.0 + P.w[1] * ph + P.w[2] * Th + P.w[3] * Th * ph)
        safe_x = np.clip(x_arr, 1e-15, 1.0 - 1e-15)
        return DG + Th * np.log(safe_x / (1.0 - safe_x)) + om * (1.0 - 2.0 * safe_x)

    def g_mix_vec(x_arr, T_arr, P_arr):
        Th = T_arr / P.T_VLCP
        ph = P_arr / P.p_VLCP
        DG = (P.a[0] + P.a[1] * ph * Th + P.a[2] * ph + P.a[3] * Th
              + P.a[4] * Th**2 + P.a[5] * ph**2 + P.a[6] * ph**3)
        om = P.w[0] * (1.0 + P.w[1] * ph + P.w[2] * Th + P.w[3] * Th * ph)
        safe_x = np.clip(x_arr, 1e-15, 1.0 - 1e-15)
        mix_ent = safe_x * np.log(safe_x) + (1.0 - safe_x) * np.log(1.0 - safe_x)
        return safe_x * DG + Th * mix_ent + om * safe_x * (1.0 - safe_x)

    return {
        'T_LLCP': llcp['T_K'],
        'p_LLCP': llcp['p_MPa'],
        'omega_vec': omega_vec,
        'disc_vec': disc_vec,
        'F_eq_vec': F_eq_vec,
        'g_mix_vec': g_mix_vec,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Liquid-Liquid Critical Point
# ═══════════════════════════════════════════════════════════════════════════

def find_LLCP():
    """
    Find the liquid-liquid critical point (LLCP).

    At x = 1/2:
      DG(Th_c, ph_c) = 0
      Th_c = omega(Th_c, ph_c) / 2

    Returns dict with keys 'T_K', 'p_MPa', 'x', 'Th', 'ph'
    """
    def _Th_c(ph):
        num = P.w[0] * (1.0 + P.w[1] * ph)
        den = 2.0 - P.w[0] * (P.w[2] + P.w[3] * ph)
        return num / den

    def _DG_at_critical(ph):
        Th = _Th_c(ph)
        return compute_DeltaG(ph, Th)[0]

    ph_lo, ph_hi = 1.0, 5.0

    f_lo = _DG_at_critical(ph_lo)
    f_hi = _DG_at_critical(ph_hi)
    if f_lo * f_hi > 0:
        for ph_test in np.linspace(0.5, 10.0, 200):
            f_test = _DG_at_critical(ph_test)
            if f_lo * f_test < 0:
                ph_hi = ph_test
                break
            ph_lo = ph_test
            f_lo = f_test

    ph_c = brentq(_DG_at_critical, ph_lo, ph_hi, xtol=1e-12)
    Th_c = _Th_c(ph_c)

    return {
        'T_K': Th_c * P.T_VLCP,
        'p_MPa': ph_c * P.p_VLCP,
        'x': 0.5,
        'Th': Th_c,
        'ph': ph_c,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2–5. Shared phase diagram functions (delegated to watereos.two_state_phase)
# ═══════════════════════════════════════════════════════════════════════════

def _get_compute_batch():
    return _gcb('duska_eos')


def _default_p_grid(p_llcp, n_pressures):
    from watereos.two_state_phase import _default_pressure_grid
    return _default_pressure_grid(p_llcp, n_pressures)


def compute_spinodal_curve(p_range_MPa=None, n_pressures=150):
    """Compute the liquid-liquid spinodal curve."""
    llcp = find_LLCP()
    if p_range_MPa is None:
        p_range_MPa = _default_p_grid(llcp['p_MPa'], n_pressures)
    p_range_MPa = np.asarray(p_range_MPa, dtype=float)
    try:
        from watereos_rs import compute_spinodal_duska
        return compute_spinodal_duska(p_range_MPa, llcp['T_K'], llcp['p_MPa'])
    except ImportError:
        return _spinodal(_get_adapter(), p_range_MPa, n_pressures)


def compute_binodal_curve(p_range_MPa=None, n_pressures=150, spinodal=None):
    """Compute the liquid-liquid binodal (coexistence) curve."""
    llcp = find_LLCP()
    if p_range_MPa is None:
        p_range_MPa = _default_p_grid(llcp['p_MPa'], n_pressures)
    p_range_MPa = np.asarray(p_range_MPa, dtype=float)
    if spinodal is None:
        spinodal = compute_spinodal_curve(p_range_MPa, n_pressures)
    try:
        from watereos_rs import compute_binodal_duska
        return compute_binodal_duska(
            spinodal['p_array'], spinodal['T_upper'], spinodal['T_lower'],
            llcp['T_K'], llcp['p_MPa'])
    except ImportError:
        return _binodal(_get_adapter(), p_range_MPa, n_pressures, spinodal)


def compute_phase_diagram(p_range_MPa=None, n_pressures=150):
    """Compute the full liquid-liquid phase diagram."""
    llcp = find_LLCP()
    if p_range_MPa is None:
        p_range_MPa = _default_p_grid(llcp['p_MPa'], n_pressures)
    p_range_MPa = np.asarray(p_range_MPa, dtype=float)
    spinodal = compute_spinodal_curve(p_range_MPa, n_pressures)
    binodal = compute_binodal_curve(p_range_MPa, n_pressures, spinodal=spinodal)
    return {
        'LLCP': {'T_K': llcp['T_K'], 'p_MPa': llcp['p_MPa'], 'x': 0.5},
        'spinodal': {'T_K': spinodal['T_K'], 'p_MPa': spinodal['p_MPa']},
        'binodal': {'T_K': binodal['T_K'], 'p_MPa': binodal['p_MPa'],
                    'x': binodal['x']},
    }


def compute_tmd_temperature(P_MPa, **kwargs):
    """Compute TMD temperature (alpha=0) at given pressure(s)."""
    return _tmd(P_MPa, _get_compute_batch(), **kwargs)


def compute_kauzmann_temperature(P_MPa, **kwargs):
    """Compute Kauzmann temperature (S_liquid = S_ice_Ih) at given pressure(s)."""
    return _kauz(P_MPa, _get_compute_batch(), **kwargs)
