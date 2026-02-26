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

import math
import numpy as np
from scipy.optimize import brentq
from . import params as P
from .core import compute_DeltaG, compute_omega


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _F_eq(x, Th, ph):
    """Equilibrium condition F(x, Th, ph) = dg/dx = 0."""
    DG = compute_DeltaG(ph, Th)[0]
    om = compute_omega(ph, Th)[0]
    return DG + Th * math.log(x / (1.0 - x)) + om * (1.0 - 2.0 * x)


def _g_mix(x, Th, ph):
    """Gibbs free energy of mixing g(x) (relative to state A, reduced)."""
    DG = compute_DeltaG(ph, Th)[0]
    om = compute_omega(ph, Th)[0]
    if x < 1e-15 or x > 1.0 - 1e-15:
        mix_ent = 0.0
    else:
        mix_ent = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    return x * DG + Th * mix_ent + om * x * (1.0 - x)


def _has_three_roots(Th, ph):
    """Check whether F(x)=0 has 3 roots at given (Th, ph)."""
    return _find_three_roots(Th, ph) is not None


def _find_three_roots(Th, ph):
    """
    Find the three roots of F(x) = 0 at given (Th, ph).

    Uses inflection points of g(x) (where d²g/dx² = 0) to bracket the roots.

    Returns
    -------
    list of roots [x1, x2, x3] with x1 < x2 < x3, or None if < 3 roots.
    """
    om = compute_omega(ph, Th)[0]
    if om <= 0:
        return None
    disc = 1.0 - 2.0 * Th / om
    if disc <= 0:
        return None  # No inflection points → single equilibrium

    sqrt_disc = math.sqrt(disc)
    x_infl_lo = (1.0 - sqrt_disc) / 2.0
    x_infl_hi = (1.0 + sqrt_disc) / 2.0

    EPS = 1e-12
    intervals = [(EPS, x_infl_lo), (x_infl_lo, x_infl_hi), (x_infl_hi, 1.0 - EPS)]

    roots = []
    for a, b in intervals:
        try:
            fa = _F_eq(a, Th, ph)
            fb = _F_eq(b, Th, ph)
        except (ValueError, ZeroDivisionError):
            continue
        if fa * fb < 0:
            try:
                r = brentq(lambda x: _F_eq(x, Th, ph), a, b, xtol=1e-13)
                roots.append(r)
            except ValueError:
                pass
        elif abs(fa) < 1e-10:
            roots.append(a)
        elif abs(fb) < 1e-10:
            roots.append(b)

    return roots if len(roots) == 3 else None


# ═══════════════════════════════════════════════════════════════════════════
# 1. Liquid-Liquid Critical Point
# ═══════════════════════════════════════════════════════════════════════════

def find_LLCP():
    """
    Find the liquid-liquid critical point (LLCP).

    At x = 1/2:
      DG(Th_c, ph_c) = 0
      Th_c = omega(Th_c, ph_c) / 2

    Returns
    -------
    dict with keys 'T_K', 'p_MPa', 'x', 'Th', 'ph'
    """
    def _Th_c(ph):
        """Critical Th as explicit function of ph from Th = omega/2."""
        num = P.w[0] * (1.0 + P.w[1] * ph)
        den = 2.0 - P.w[0] * (P.w[2] + P.w[3] * ph)
        return num / den

    def _DG_at_critical(ph):
        """DG evaluated at (Th_c(ph), ph) — should be 0 at LLCP."""
        Th = _Th_c(ph)
        return compute_DeltaG(ph, Th)[0]

    # Bracket: search for sign change in DG across a range of pressures
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
# 2. Spinodal Curve
# ═══════════════════════════════════════════════════════════════════════════

def _find_spinodal_temps(ph, T_LLCP_K):
    """
    Find the upper and lower spinodal temperatures at a given pressure
    by bisecting on the existence of 3 roots of F(x) = 0.

    Finds the contiguous 3-root region connected to the LLCP by scanning
    downward from the LLCP temperature.

    Returns dict or None.
    """
    # First, find a temperature inside the 3-root region by scanning down.
    # Use fine steps near LLCP (dome is narrow there), coarser further away.
    T_inside = None
    scan_temps = np.concatenate([
        np.arange(T_LLCP_K - 0.1, T_LLCP_K - 5.0, -0.5),
        np.arange(T_LLCP_K - 5.0, 50.0, -2.0),
    ])
    for T_test in scan_temps:
        Th = T_test / P.T_VLCP
        if _has_three_roots(Th, ph):
            T_inside = T_test
            break

    if T_inside is None:
        return None

    # Find upper spinodal: bisect between T_inside and T_LLCP+5
    T_a, T_b = T_inside, T_LLCP_K + 5.0
    for _ in range(60):
        T_mid = (T_a + T_b) / 2.0
        if _has_three_roots(T_mid / P.T_VLCP, ph):
            T_a = T_mid
        else:
            T_b = T_mid
    T_upper = (T_a + T_b) / 2.0

    # Find lower spinodal: scan downward from T_inside in small steps
    # to find where the contiguous 3-root region ends (avoid jumping
    # over gaps into separate low-T 3-root regions)
    T_lower_edge = T_inside
    step = 2.0
    while T_lower_edge > 10.0:
        T_test = T_lower_edge - step
        if T_test < 5.0:
            break
        if _has_three_roots(T_test / P.T_VLCP, ph):
            T_lower_edge = T_test
        else:
            # Found the edge — refine with bisection
            T_a, T_b = T_test, T_lower_edge
            for _ in range(50):
                T_mid = (T_a + T_b) / 2.0
                if _has_three_roots(T_mid / P.T_VLCP, ph):
                    T_b = T_mid
                else:
                    T_a = T_mid
            T_lower_edge = (T_a + T_b) / 2.0
            break
    T_lower = T_lower_edge

    # Get the equilibrium x at the spinodal temperatures (just inside)
    roots_upper = _find_three_roots((T_upper - 0.05) / P.T_VLCP, ph)
    roots_lower = _find_three_roots((T_lower + 0.05) / P.T_VLCP, ph)

    x_lo_upper = roots_upper[0] if roots_upper else None
    x_hi_upper = roots_upper[2] if roots_upper else None
    x_lo_lower = roots_lower[0] if roots_lower else None
    x_hi_lower = roots_lower[2] if roots_lower else None

    return {
        'T_upper': T_upper,
        'T_lower': T_lower,
        'x_lo_upper': x_lo_upper,
        'x_hi_upper': x_hi_upper,
        'x_lo_lower': x_lo_lower,
        'x_hi_lower': x_hi_lower,
    }


def compute_spinodal_curve(p_range_MPa=None, n_pressures=150):
    """
    Compute the liquid-liquid spinodal curve.

    At each pressure, finds the upper and lower temperature bounds of the
    3-root region of F(x) = 0 by bisection.

    Parameters
    ----------
    p_range_MPa : array-like or None
        Pressures in MPa. If None, auto-range from LLCP to 200 MPa.
    n_pressures : int
        Number of pressure points if p_range_MPa is None.

    Returns
    -------
    dict with arrays: T_K, p_MPa, x (for both branches, concatenated as
    a closed curve going up one branch and down the other)
    """
    llcp = find_LLCP()

    if p_range_MPa is None:
        p_llcp = llcp['p_MPa']
        # Dense near LLCP (0.05 MPa steps for first 5 MPa), sparser beyond
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
        ph = p_MPa / P.p_VLCP
        result = _find_spinodal_temps(ph, llcp['T_K'])
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
    x_lo_up = np.array(x_lo_up)
    x_hi_up = np.array(x_hi_up)
    x_lo_dn = np.array(x_lo_dn)
    x_hi_dn = np.array(x_hi_dn)

    # Build closed curve in T-p: go along upper branch, then back along lower
    T_curve = np.concatenate([[llcp['T_K']], T_upper, T_lower[::-1], [llcp['T_K']]])
    p_curve = np.concatenate([[llcp['p_MPa']], p_valid, p_valid[::-1], [llcp['p_MPa']]])

    return {
        'T_K': T_curve,
        'p_MPa': p_curve,
        # Separate branches for convenience
        'T_upper': T_upper, 'T_lower': T_lower,
        'x_lo_upper': x_lo_up, 'x_hi_upper': x_hi_up,
        'x_lo_lower': x_lo_dn, 'x_hi_lower': x_hi_dn,
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

    Parameters
    ----------
    p_range_MPa : array-like or None
        Pressures in MPa. If None, auto-range from LLCP to 200 MPa.
    n_pressures : int
        Number of pressure points if p_range_MPa is None.

    Returns
    -------
    dict with arrays: T_K, p_MPa, x_lo, x_hi (coexisting compositions)
    """
    llcp = find_LLCP()
    spinodal = compute_spinodal_curve(p_range_MPa, n_pressures)

    p_arr = spinodal['p_array']
    T_spin_upper = spinodal['T_upper']
    T_spin_lower = spinodal['T_lower']

    T_binodal = []
    x_binodal_lo = []
    x_binodal_hi = []
    p_valid = []

    for i, p_MPa in enumerate(p_arr):
        ph = p_MPa / P.p_VLCP
        T_lo = T_spin_lower[i] + 0.01
        T_hi = T_spin_upper[i] - 0.01

        if T_hi <= T_lo:
            continue

        def _delta_g(T_K):
            """g(x3) - g(x1) as function of temperature."""
            Th = T_K / P.T_VLCP
            roots = _find_three_roots(Th, ph)
            if roots is None:
                return float('nan')
            return _g_mix(roots[2], Th, ph) - _g_mix(roots[0], Th, ph)

        # Scan within the 3-root region for sign change in delta_g
        n_scan = 40
        T_scan = np.linspace(T_lo, T_hi, n_scan)
        dg_vals = np.array([_delta_g(T) for T in T_scan])

        for j in range(len(dg_vals) - 1):
            if np.isnan(dg_vals[j]) or np.isnan(dg_vals[j + 1]):
                continue
            if dg_vals[j] * dg_vals[j + 1] < 0:
                try:
                    T_eq = brentq(_delta_g, T_scan[j], T_scan[j + 1], xtol=1e-6)
                    Th_eq = T_eq / P.T_VLCP
                    roots = _find_three_roots(Th_eq, ph)
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

    # Build closed curve in T-p (binodal is a single T for each p)
    T_curve = np.concatenate([[llcp['T_K']], T_binodal, T_binodal[::-1], [llcp['T_K']]])
    p_curve = np.concatenate([[llcp['p_MPa']], p_valid, p_valid[::-1], [llcp['p_MPa']]])
    x_curve = np.concatenate([[0.5], x_binodal_lo, x_binodal_hi[::-1], [0.5]])

    return {
        'T_K': T_curve,
        'p_MPa': p_curve,
        'x': x_curve,
        'T_binodal': T_binodal,
        'x_lo': x_binodal_lo,
        'x_hi': x_binodal_hi,
        'p_array': p_valid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Convenience: full phase diagram
# ═══════════════════════════════════════════════════════════════════════════

def compute_phase_diagram(p_range_MPa=None, n_pressures=150):
    """
    Compute the full liquid-liquid phase diagram.

    Returns
    -------
    dict with keys:
        'LLCP': dict with T_K, p_MPa, x
        'spinodal': dict with T_K, p_MPa, x (closed curve in T-p)
        'binodal': dict with T_K, p_MPa, x (closed curve in T-p and T-x)
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
            'T_K': spinodal['T_K'],
            'p_MPa': spinodal['p_MPa'],
        },
        'binodal': {
            'T_K': binodal['T_K'],
            'p_MPa': binodal['p_MPa'],
            'x': binodal['x'],
        },
    }
