"""
Liquid-liquid coexistence (binodal) and spinodal curves for the
Holten et al. (2014) two-state EoS.

The LLCP is at Tc = 228.2 K, Pc = 0 MPa (mean-field approximation).

The equilibrium condition is:
    L + ln(x/(1-x)) + omega*(1-2x) = 0
where omega = 2 + omega0*p.

Spinodal:  d²g/dx² = 0  =>  1/(x*(1-x)) = 2*omega
LLCP:      x = 1/2, L = 0, omega = 2

Reference: V. Holten, J. V. Sengers, M. A. Anisimov,
           J. Phys. Chem. Ref. Data 43, 014101 (2014).
"""

import numpy as np
from . import params as P

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
    """Convert physical to reduced variables (field coordinates)."""
    t = (T_K - P.Tc) / P.Tc
    p_red = (p_MPa * 1e6 - P.Pc * 1e6) / P.P_scale_Pa
    return t, p_red


def _compute_L_vec(t_arr, p_red_arr):
    """Vectorized computation of the hyperbolic field L."""
    k0, k1, k2, L0 = P.k0, P.k1, P.k2, P.L0

    arg = p_red_arr - k2 * t_arr
    inner = 1.0 + k0 * k2 + k1 * arg
    # Safe sqrt: clamp argument to avoid NaN from numerical issues
    K1_arg = inner**2 - 4.0 * k0 * k1 * k2 * arg
    K1 = np.sqrt(np.maximum(K1_arg, 0.0))
    K2 = np.sqrt(1.0 + k2**2)

    L = L0 * K2 * (1.0 - K1 + k0 * k2 + k1 * (p_red_arr + k2 * t_arr)) / (2.0 * k1 * k2)
    return L


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized adapter for fast_phase_diagram
# ═══════════════════════════════════════════════════════════════════════════

def _get_adapter():
    """Return vectorized adapter for the fast_phase_diagram module."""
    llcp = find_LLCP()

    def omega_vec(T_arr, P_arr):
        _, p_red = _to_reduced(T_arr, P_arr)
        return 2.0 + P.omega0 * p_red

    def disc_vec(omega, T_arr, P_arr):
        return 1.0 - 2.0 / omega

    def F_eq_vec(x_arr, T_arr, P_arr):
        t, p_red = _to_reduced(T_arr, P_arr)
        L = _compute_L_vec(t, p_red)
        omega = 2.0 + P.omega0 * p_red
        safe_x = np.clip(x_arr, 1e-15, 1.0 - 1e-15)
        return L + np.log(safe_x / (1.0 - safe_x)) + omega * (1.0 - 2.0 * safe_x)

    def g_mix_vec(x_arr, T_arr, P_arr):
        t, p_red = _to_reduced(T_arr, P_arr)
        L = _compute_L_vec(t, p_red)
        omega = 2.0 + P.omega0 * p_red
        safe_x = np.clip(x_arr, 1e-15, 1.0 - 1e-15)
        mix_ent = safe_x * np.log(safe_x) + (1.0 - safe_x) * np.log(1.0 - safe_x)
        return safe_x * L + mix_ent + omega * safe_x * (1.0 - safe_x)

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
    Return the LLCP coordinates.
    For Holten's model, Tc and Pc are parameters: Tc=228.2 K, Pc=0 MPa.

    Returns dict with keys 'T_K', 'p_MPa', 'x'.
    """
    return {
        'T_K': P.Tc,
        'p_MPa': P.Pc,
        'x': 0.5,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2–5. Shared phase diagram functions (delegated to watereos.two_state_phase)
# ═══════════════════════════════════════════════════════════════════════════

def _get_compute_batch():
    return _gcb('holten_eos')


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
        from watereos_rs import compute_spinodal_holten
        return compute_spinodal_holten(p_range_MPa, llcp['T_K'], llcp['p_MPa'])
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
        from watereos_rs import compute_binodal_holten
        return compute_binodal_holten(
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
