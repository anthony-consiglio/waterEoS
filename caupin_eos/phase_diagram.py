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

import math
import numpy as np
from scipy.optimize import brentq
from . import params as P
from .core import compute_GBA, compute_omega


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_reduced(T_K, p_MPa):
    """Convert physical to reduced variables."""
    dTh = (T_K - P.Tc) / P.Tc
    dPh = (p_MPa - P.Pc) / P.P_scale_MPa
    return dTh, dPh


def _F_eq(x, dTh, dPh):
    """Equilibrium condition F(x) = dg/dx = 0."""
    GBA = compute_GBA(dTh, dPh)[0]
    om = compute_omega(dTh, dPh)[0]
    Th = 1.0 + dTh
    return GBA + Th * (math.log(x / (1.0 - x)) + om * (1.0 - 2.0 * x))


def _g_mix(x, dTh, dPh):
    """Gibbs free energy of mixing g(x) (relative to state A, reduced)."""
    GBA = compute_GBA(dTh, dPh)[0]
    om = compute_omega(dTh, dPh)[0]
    Th = 1.0 + dTh
    if x < 1e-15 or x > 1.0 - 1e-15:
        mix_ent = 0.0
    else:
        mix_ent = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    return x * GBA + Th * (mix_ent + om * x * (1.0 - x))


def _find_three_roots(dTh, dPh):
    """
    Find the three roots of F(x) = 0 at given reduced coordinates.
    Returns [x1, x2, x3] with x1 < x2 < x3, or None if < 3 roots.
    """
    om = compute_omega(dTh, dPh)[0]
    Th = 1.0 + dTh
    if om <= 0:
        return None
    # Spinodal: d²g/dx² = T̂*(1/(x(1-x)) - 2ω̂) = 0 → x(1-x) = 1/(2ω̂)
    # Quadratic: x² - x + 1/(2ω̂) = 0, disc = 1 - 2/ω̂
    disc = 1.0 - 2.0 / om
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
            fa = _F_eq(a, dTh, dPh)
            fb = _F_eq(b, dTh, dPh)
        except (ValueError, ZeroDivisionError):
            continue
        if fa * fb < 0:
            try:
                r = brentq(lambda x: _F_eq(x, dTh, dPh), a, b, xtol=1e-13)
                roots.append(r)
            except ValueError:
                pass
        elif abs(fa) < 1e-10:
            roots.append(a)
        elif abs(fb) < 1e-10:
            roots.append(b)

    return roots if len(roots) == 3 else None


def _has_three_roots(dTh, dPh):
    return _find_three_roots(dTh, dPh) is not None


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
        """Critical T̂ as function of ΔP̂ from ω̂ = 2 → T̂ = (2+ω₀ΔP̂)/2."""
        arg = (2.0 + P.omega0 * dPh) / 2.0
        if arg <= 0:
            return None
        return arg

    def _GBA_at_critical(dPh):
        """GBA evaluated at (ΔT̂_c(ΔP̂), ΔP̂) — should be 0 at LLCP."""
        Th = _Th_c(dPh)
        if Th is None:
            return float('inf')
        dTh = Th - 1.0
        return compute_GBA(dTh, dPh)[0]

    # Bracket search for zero of GBA
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
# 2. Spinodal Curve
# ═══════════════════════════════════════════════════════════════════════════

def _find_spinodal_temps(dPh, T_LLCP_K):
    """
    Find the upper and lower spinodal temperatures at a given ΔP̂
    by bisecting on the existence of 3 roots.
    """
    T_inside = None
    scan_temps = np.concatenate([
        np.arange(T_LLCP_K - 0.1, T_LLCP_K - 5.0, -0.5),
        np.arange(T_LLCP_K - 5.0, 50.0, -2.0),
    ])
    for T_test in scan_temps:
        dTh_test = (T_test - P.Tc) / P.Tc
        if _has_three_roots(dTh_test, dPh):
            T_inside = T_test
            break

    if T_inside is None:
        return None

    # Upper spinodal
    T_a, T_b = T_inside, T_LLCP_K + 5.0
    for _ in range(60):
        T_mid = (T_a + T_b) / 2.0
        dTh_mid = (T_mid - P.Tc) / P.Tc
        if _has_three_roots(dTh_mid, dPh):
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
        dTh_test = (T_test - P.Tc) / P.Tc
        if _has_three_roots(dTh_test, dPh):
            T_lower_edge = T_test
        else:
            T_a, T_b = T_test, T_lower_edge
            for _ in range(50):
                T_mid = (T_a + T_b) / 2.0
                dTh_mid = (T_mid - P.Tc) / P.Tc
                if _has_three_roots(dTh_mid, dPh):
                    T_b = T_mid
                else:
                    T_a = T_mid
            T_lower_edge = (T_a + T_b) / 2.0
            break
    T_lower = T_lower_edge

    # Get x values at spinodal boundaries
    roots_upper = _find_three_roots((T_upper - 0.05 - P.Tc) / P.Tc, dPh)
    roots_lower = _find_three_roots((T_lower + 0.05 - P.Tc) / P.Tc, dPh)

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
        dPh = (p_MPa - P.Pc) / P.P_scale_MPa
        result = _find_spinodal_temps(dPh, llcp['T_K'])
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

    At each pressure, bisect on temperature within the 3-root region
    to find where g(x1) = g(x3).
    """
    llcp = find_LLCP()
    spinodal = compute_spinodal_curve(p_range_MPa, n_pressures)

    p_arr = spinodal['p_array']
    T_spin_upper = spinodal['T_upper']
    T_spin_lower = spinodal['T_lower']

    T_binodal, x_binodal_lo, x_binodal_hi, p_valid = [], [], [], []

    for i, p_MPa in enumerate(p_arr):
        dPh = (p_MPa - P.Pc) / P.P_scale_MPa
        T_lo = T_spin_lower[i] + 0.01
        T_hi = T_spin_upper[i] - 0.01

        if T_hi <= T_lo:
            continue

        def _delta_g(T_K, _dPh=dPh):
            dTh = (T_K - P.Tc) / P.Tc
            roots = _find_three_roots(dTh, _dPh)
            if roots is None:
                return float('nan')
            return _g_mix(roots[2], dTh, _dPh) - _g_mix(roots[0], dTh, _dPh)

        # Scan from upper spinodal downward — the LLTL is always just
        # below the upper spinodal arm, so searching top-down avoids
        # spurious crossings near a low-T lobe.
        n_scan = 50
        T_scan = np.linspace(T_hi, T_lo, n_scan)  # high → low
        dg_vals = np.array([_delta_g(T) for T in T_scan])

        for j in range(len(dg_vals) - 1):
            if np.isnan(dg_vals[j]) or np.isnan(dg_vals[j + 1]):
                continue
            if dg_vals[j] * dg_vals[j + 1] < 0:
                # Bracket found; T_scan[j] > T_scan[j+1] so order for brentq
                T_a = min(T_scan[j], T_scan[j + 1])
                T_b = max(T_scan[j], T_scan[j + 1])
                try:
                    T_eq = brentq(_delta_g, T_a, T_b, xtol=1e-6)
                    dTh_eq = (T_eq - P.Tc) / P.Tc
                    roots = _find_three_roots(dTh_eq, dPh)
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

    T_curve = np.concatenate([
        [llcp['T_K']], T_binodal, T_binodal[::-1], [llcp['T_K']]])
    p_curve = np.concatenate([
        [llcp['p_MPa']], p_valid, p_valid[::-1], [llcp['p_MPa']]])
    x_curve = np.concatenate([
        [0.5], x_binodal_lo, x_binodal_hi[::-1], [0.5]])

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
