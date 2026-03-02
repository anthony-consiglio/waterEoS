"""
Liquid-liquid coexistence (binodal) and spinodal curves for the
Caupin & Anisimov (2019) two-state EoS.

The Gibbs free energy of mixing is:
    g(x) = x*GBA + T̂*[x*ln(x) + (1-x)*ln(1-x) + ω̂*x*(1-x)]

Note: ω̂ is INSIDE the T̂*[...] bracket. Since T̂*ω̂ = Ω = 2+ω₀ΔP̂:
    g(x) = x*GBA + T̂*[x*ln(x) + (1-x)*ln(1-x)] + Ω*x*(1-x)

Equilibrium: dg/dx = 0  =>  F(x) = GBA + T̂*[ln(x/(1-x)) + ω̂*(1-2x)] = 0
Spinodal:    d²g/dx² = 0  =>  1/(x*(1-x)) = 2*ω̂
LLCP:        x = 1/2,  GBA = 0,  ω̂ = 2  =>  T̂ = (2+ω₀ΔP̂)/2

Reference: F. Caupin and M. A. Anisimov, J. Chem. Phys. 151, 034503 (2019).
"""

import numpy as np
from scipy.optimize import brentq
from . import params as P
from .core import compute_GBA, compute_omega

from watereos.two_state_phase import (
    get_compute_batch as _gcb,
    compute_spinodal_curve as _spinodal,
    compute_binodal_curve as _binodal,
    compute_phase_diagram as _phase_diag,
    compute_tmd_temperature as _tmd,
    compute_kauzmann_temperature as _kauz,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_reduced(T_K, p_MPa):
    """Convert physical to reduced variables."""
    dTh = (T_K - P.Tc) / P.Tc
    dPh = (p_MPa - P.Pc) / P.P_scale_MPa
    return dTh, dPh


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized adapter for fast_phase_diagram
# ═══════════════════════════════════════════════════════════════════════════

def _get_adapter():
    """Return vectorized adapter for the fast_phase_diagram module."""
    llcp = find_LLCP()

    def omega_vec(T_arr, P_arr):
        dTh = (T_arr - P.Tc) / P.Tc
        dPh = (P_arr - P.Pc) / P.P_scale_MPa
        return (2.0 + P.omega0 * dPh) / (1.0 + dTh)

    def disc_vec(omega, T_arr, P_arr):
        return 1.0 - 2.0 / omega

    def F_eq_vec(x_arr, T_arr, P_arr):
        dTh = (T_arr - P.Tc) / P.Tc
        dPh = (P_arr - P.Pc) / P.P_scale_MPa
        GBA = P.lam * (dTh + P.a * dPh + P.b * dTh * dPh
                       + P.d * dPh**2 + P.f * dTh**2)
        Th = 1.0 + dTh
        om = (2.0 + P.omega0 * dPh) / Th
        safe_x = np.clip(x_arr, 1e-15, 1.0 - 1e-15)
        return GBA + Th * (np.log(safe_x / (1.0 - safe_x))
                           + om * (1.0 - 2.0 * safe_x))

    def g_mix_vec(x_arr, T_arr, P_arr):
        dTh = (T_arr - P.Tc) / P.Tc
        dPh = (P_arr - P.Pc) / P.P_scale_MPa
        GBA = P.lam * (dTh + P.a * dPh + P.b * dTh * dPh
                       + P.d * dPh**2 + P.f * dTh**2)
        Th = 1.0 + dTh
        om = (2.0 + P.omega0 * dPh) / Th
        safe_x = np.clip(x_arr, 1e-15, 1.0 - 1e-15)
        mix_ent = safe_x * np.log(safe_x) + (1.0 - safe_x) * np.log(1.0 - safe_x)
        return safe_x * GBA + Th * (mix_ent + om * safe_x * (1.0 - safe_x))

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
    Find the LLCP where x = 1/2, GBA = 0, ω̂ = 2.

    At x = 1/2:
      GBA(ΔT̂_c, ΔP̂_c) = 0
      ω̂ = 2  →  T̂_c = (2 + ω₀·ΔP̂_c) / 2

    Returns dict with keys 'T_K', 'p_MPa', 'x', 'dTh', 'dPh'.
    """
    def _Th_c(dPh):
        arg = (2.0 + P.omega0 * dPh) / 2.0
        if arg <= 0:
            return None
        return arg

    def _GBA_at_critical(dPh):
        Th = _Th_c(dPh)
        if Th is None:
            return float('inf')
        dTh = Th - 1.0
        return compute_GBA(dTh, dPh)[0]

    dPh_lo, dPh_hi = -0.5, 0.5
    f_lo = _GBA_at_critical(dPh_lo)
    f_hi = _GBA_at_critical(dPh_hi)

    if f_lo * f_hi > 0:
        for dPh_test in np.linspace(-2.0, 5.0, 500):
            f_test = _GBA_at_critical(dPh_test)
            if f_lo * f_test < 0:
                dPh_hi = dPh_test
                break
            dPh_lo = dPh_test
            f_lo = f_test

    dPh_c = brentq(_GBA_at_critical, dPh_lo, dPh_hi, xtol=1e-12)
    Th_c = _Th_c(dPh_c)
    dTh_c = Th_c - 1.0

    T_c = P.Tc * Th_c
    p_c = P.Pc + dPh_c * P.P_scale_MPa

    return {
        'T_K': T_c,
        'p_MPa': p_c,
        'x': 0.5,
        'dTh': dTh_c,
        'dPh': dPh_c,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2–5. Shared phase diagram functions (delegated to watereos.two_state_phase)
# ═══════════════════════════════════════════════════════════════════════════

def _get_compute_batch():
    return _gcb('caupin_eos')


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
        from watereos_rs import compute_spinodal_caupin
        return compute_spinodal_caupin(p_range_MPa, llcp['T_K'], llcp['p_MPa'])
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
        from watereos_rs import compute_binodal_caupin
        return compute_binodal_caupin(
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
