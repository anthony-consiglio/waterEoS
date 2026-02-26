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

import math
import numpy as np
from scipy.optimize import brentq
from . import params as P
from .core import _compute_L, _reduce


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_reduced(T_K, p_MPa):
    """Convert physical to reduced variables (field coordinates)."""
    t = (T_K - P.Tc) / P.Tc
    p_red = (p_MPa * 1e6 - P.Pc * 1e6) / P.P_scale_Pa
    return t, p_red


def _F_eq(x, t, p_red):
    """Equilibrium condition F(x) = L + ln(x/(1-x)) + omega*(1-2x) = 0."""
    L = _compute_L(t, p_red)[0]
    omega = 2.0 + P.omega0 * p_red
    return L + math.log(x / (1.0 - x)) + omega * (1.0 - 2.0 * x)


def _g_mix(x, t, p_red):
    """Gibbs free energy of mixing g(x) (reduced)."""
    L = _compute_L(t, p_red)[0]
    omega = 2.0 + P.omega0 * p_red
    EPS = 1e-15
    if EPS < x < 1.0 - EPS:
        mix_ent = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    else:
        mix_ent = 0.0
    return x * L + mix_ent + omega * x * (1.0 - x)


def _find_three_roots(t, p_red):
    """
    Find the three roots of F(x) = 0 at given reduced coordinates.
    Returns [x1, x2, x3] with x1 < x2 < x3, or None if < 3 roots.
    """
    omega = 2.0 + P.omega0 * p_red
    if omega <= 0:
        return None
    disc = 1.0 - 2.0 / omega
    if disc <= 0:
        return None

    sqrt_disc = math.sqrt(disc)
    x_infl_lo = (1.0 - sqrt_disc) / 2.0
    x_infl_hi = (1.0 + sqrt_disc) / 2.0

    EPS = 1e-12
    intervals = [(EPS, x_infl_lo), (x_infl_lo, x_infl_hi),
                 (x_infl_hi, 1.0 - EPS)]

    roots = []
    for a, b in intervals:
        try:
            fa = _F_eq(a, t, p_red)
            fb = _F_eq(b, t, p_red)
        except (ValueError, ZeroDivisionError):
            continue
        if fa * fb < 0:
            try:
                r = brentq(lambda x: _F_eq(x, t, p_red), a, b, xtol=1e-13)
                roots.append(r)
            except ValueError:
                pass
        elif abs(fa) < 1e-10:
            roots.append(a)
        elif abs(fb) < 1e-10:
            roots.append(b)

    return roots if len(roots) == 3 else None


def _has_three_roots(t, p_red):
    return _find_three_roots(t, p_red) is not None


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
# 2. Spinodal Curve
# ═══════════════════════════════════════════════════════════════════════════

def _find_spinodal_temps(p_MPa, T_LLCP_K):
    """
    Find the upper and lower spinodal temperatures at a given pressure
    by bisecting on the existence of 3 roots.
    """
    _, p_red = _to_reduced(T_LLCP_K, p_MPa)

    T_inside = None
    scan_temps = np.concatenate([
        np.arange(T_LLCP_K - 0.1, T_LLCP_K - 5.0, -0.5),
        np.arange(T_LLCP_K - 5.0, 50.0, -2.0),
    ])
    for T_test in scan_temps:
        t_test = (T_test - P.Tc) / P.Tc
        if _has_three_roots(t_test, p_red):
            T_inside = T_test
            break

    if T_inside is None:
        return None

    # Upper spinodal
    T_a, T_b = T_inside, T_LLCP_K + 5.0
    for _ in range(60):
        T_mid = (T_a + T_b) / 2.0
        t_mid = (T_mid - P.Tc) / P.Tc
        if _has_three_roots(t_mid, p_red):
            T_a = T_mid
        else:
            T_b = T_mid
    T_upper = (T_a + T_b) / 2.0

    # Lower spinodal
    T_lower_edge = T_inside
    step = 2.0
    while T_lower_edge > 10.0:
        T_test = T_lower_edge - step
        if T_test < 5.0:
            break
        t_test = (T_test - P.Tc) / P.Tc
        if _has_three_roots(t_test, p_red):
            T_lower_edge = T_test
        else:
            T_a, T_b = T_test, T_lower_edge
            for _ in range(50):
                T_mid = (T_a + T_b) / 2.0
                t_mid = (T_mid - P.Tc) / P.Tc
                if _has_three_roots(t_mid, p_red):
                    T_b = T_mid
                else:
                    T_a = T_mid
            T_lower_edge = (T_a + T_b) / 2.0
            break
    T_lower = T_lower_edge

    # Get x values near spinodal boundaries
    roots_upper = _find_three_roots((T_upper - 0.05 - P.Tc) / P.Tc, p_red)
    roots_lower = _find_three_roots((T_lower + 0.05 - P.Tc) / P.Tc, p_red)

    return {
        'T_upper': T_upper,
        'T_lower': T_lower,
        'x_lo_upper': roots_upper[0] if roots_upper else None,
        'x_hi_upper': roots_upper[2] if roots_upper else None,
        'x_lo_lower': roots_lower[0] if roots_lower else None,
        'x_hi_lower': roots_lower[2] if roots_lower else None,
    }


def compute_spinodal_curve(p_range_MPa=None, n_pressures=150):
    """
    Compute the liquid-liquid spinodal curve.

    Returns dict with arrays for the closed T-p curve and separate branches.
    """
    llcp = find_LLCP()

    if p_range_MPa is None:
        p_llcp = llcp['p_MPa']
        n_near = min(n_pressures // 3, 100)
        n_far = n_pressures - n_near
        p_near = np.linspace(p_llcp + 0.05, p_llcp + 5.0, n_near, endpoint=False)
        p_far = np.linspace(p_llcp + 5.0, 200.0, n_far)
        p_range_MPa = np.concatenate([p_near, p_far])

    T_upper, T_lower = [], []
    x_lo_up, x_hi_up = [], []
    x_lo_dn, x_hi_dn = [], []
    p_valid = []

    for p_MPa in p_range_MPa:
        result = _find_spinodal_temps(p_MPa, llcp['T_K'])
        if result is None:
            continue

        T_upper.append(result['T_upper'])
        T_lower.append(result['T_lower'])
        x_lo_up.append(result['x_lo_upper'])
        x_hi_up.append(result['x_hi_upper'])
        x_lo_dn.append(result['x_lo_lower'])
        x_hi_dn.append(result['x_hi_lower'])
        p_valid.append(p_MPa)

    T_upper = np.array(T_upper)
    T_lower = np.array(T_lower)
    p_valid = np.array(p_valid)

    T_curve = np.concatenate([
        [llcp['T_K']], T_upper, T_lower[::-1], [llcp['T_K']]])
    p_curve = np.concatenate([
        [llcp['p_MPa']], p_valid, p_valid[::-1], [llcp['p_MPa']]])

    return {
        'T_K': T_curve, 'p_MPa': p_curve,
        'T_upper': T_upper, 'T_lower': T_lower,
        'x_lo_upper': np.array(x_lo_up), 'x_hi_upper': np.array(x_hi_up),
        'x_lo_lower': np.array(x_lo_dn), 'x_hi_lower': np.array(x_hi_dn),
        'p_array': p_valid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Binodal (Coexistence) Curve
# ═══════════════════════════════════════════════════════════════════════════

def compute_binodal_curve(p_range_MPa=None, n_pressures=150):
    """
    Compute the liquid-liquid binodal (coexistence) curve.
    """
    llcp = find_LLCP()
    spinodal = compute_spinodal_curve(p_range_MPa, n_pressures)

    p_arr = spinodal['p_array']
    T_spin_upper = spinodal['T_upper']
    T_spin_lower = spinodal['T_lower']

    T_binodal, x_binodal_lo, x_binodal_hi, p_valid = [], [], [], []

    for i, p_MPa in enumerate(p_arr):
        _, p_red = _to_reduced(P.Tc, p_MPa)
        T_lo = T_spin_lower[i] + 0.01
        T_hi = T_spin_upper[i] - 0.01

        if T_hi <= T_lo:
            continue

        def _delta_g(T_K, _p_red=p_red):
            t = (T_K - P.Tc) / P.Tc
            roots = _find_three_roots(t, _p_red)
            if roots is None:
                return float('nan')
            return _g_mix(roots[2], t, _p_red) - _g_mix(roots[0], t, _p_red)

        n_scan = 50
        T_scan = np.linspace(T_hi, T_lo, n_scan)
        dg_vals = np.array([_delta_g(T) for T in T_scan])

        for j in range(len(dg_vals) - 1):
            if np.isnan(dg_vals[j]) or np.isnan(dg_vals[j + 1]):
                continue
            if dg_vals[j] * dg_vals[j + 1] < 0:
                T_a = min(T_scan[j], T_scan[j + 1])
                T_b = max(T_scan[j], T_scan[j + 1])
                try:
                    T_eq = brentq(_delta_g, T_a, T_b, xtol=1e-6)
                    t_eq = (T_eq - P.Tc) / P.Tc
                    roots = _find_three_roots(t_eq, p_red)
                    if roots is not None:
                        T_binodal.append(T_eq)
                        x_binodal_lo.append(roots[0])
                        x_binodal_hi.append(roots[2])
                        p_valid.append(p_MPa)
                except (ValueError, RuntimeError):
                    pass
                break

    T_binodal = np.array(T_binodal)
    x_binodal_lo = np.array(x_binodal_lo)
    x_binodal_hi = np.array(x_binodal_hi)
    p_valid = np.array(p_valid)

    if len(T_binodal) > 0:
        T_curve = np.concatenate([
            [llcp['T_K']], T_binodal, T_binodal[::-1], [llcp['T_K']]])
        p_curve = np.concatenate([
            [llcp['p_MPa']], p_valid, p_valid[::-1], [llcp['p_MPa']]])
        x_curve = np.concatenate([
            [0.5], x_binodal_lo, x_binodal_hi[::-1], [0.5]])
    else:
        T_curve = np.array([llcp['T_K']])
        p_curve = np.array([llcp['p_MPa']])
        x_curve = np.array([0.5])

    return {
        'T_K': T_curve, 'p_MPa': p_curve, 'x': x_curve,
        'T_binodal': T_binodal,
        'x_lo': x_binodal_lo, 'x_hi': x_binodal_hi,
        'p_array': p_valid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Full phase diagram
# ═══════════════════════════════════════════════════════════════════════════

def compute_phase_diagram(p_range_MPa=None, n_pressures=150):
    """
    Compute the full liquid-liquid phase diagram.

    Returns dict with 'LLCP', 'spinodal', 'binodal'.
    """
    llcp = find_LLCP()

    if p_range_MPa is None:
        p_llcp = llcp['p_MPa']
        n_near = min(n_pressures // 3, 100)
        n_far = n_pressures - n_near
        p_near = np.linspace(p_llcp + 0.05, p_llcp + 5.0, n_near, endpoint=False)
        p_far = np.linspace(p_llcp + 5.0, 200.0, n_far)
        p_range_MPa = np.concatenate([p_near, p_far])

    spinodal = compute_spinodal_curve(p_range_MPa, n_pressures)
    binodal = compute_binodal_curve(p_range_MPa, n_pressures)

    return {
        'LLCP': {'T_K': llcp['T_K'], 'p_MPa': llcp['p_MPa'], 'x': 0.5},
        'spinodal': {
            'T_K': spinodal['T_K'], 'p_MPa': spinodal['p_MPa'],
        },
        'binodal': {
            'T_K': binodal['T_K'], 'p_MPa': binodal['p_MPa'],
            'x': binodal['x'],
        },
    }
