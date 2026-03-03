"""
Vectorized spinodal/binodal algorithms for two-state EoS models.

All algorithms operate on numpy arrays, bisecting all pressures simultaneously.
Each model provides an "adapter" dict with vectorized callables:

    adapter = {
        'T_LLCP': float,
        'p_LLCP': float,
        'omega_vec':  fn(T_arr, P_arr) -> omega array,
        'disc_vec':   fn(omega_arr, T_arr, P_arr) -> discriminant array,
        'F_eq_vec':   fn(x_arr, T_arr, P_arr) -> F(x) array,
        'g_mix_vec':  fn(x_arr, T_arr, P_arr) -> g(x) array,
    }
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Core vectorized helpers
# ═══════════════════════════════════════════════════════════════════════════

def has_three_roots_vec(T_arr, P_arr, adapter):
    """
    Check whether F(x)=0 has 3 roots at each (T, P) pair.

    F(x) is the equilibrium condition dG_mix/dx = 0.  Inside the spinodal
    dome, G_mix(x) has two local minima and one maximum, giving three roots.
    Outside the dome there is only one root (the equilibrium x).

    The discriminant determines whether the inflection points of F(x) yield
    a composition interval where three sign changes occur.

    Returns boolean array of same length as T_arr.
    """
    omega = adapter['omega_vec'](T_arr, P_arr)
    disc = adapter['disc_vec'](omega, T_arr, P_arr)

    # omega > 0 and disc > 0 are necessary conditions for a spinodal dome
    valid = (omega > 0) & (disc > 0)

    # Inflection points of F(x) in composition space: these divide (0, 1)
    # into intervals where F is monotonic, enabling bisection for each root.
    sqrt_disc = np.sqrt(np.where(disc > 0, disc, 1.0))
    x_lo = (1.0 - sqrt_disc) / 2.0   # lower inflection composition
    x_hi = (1.0 + sqrt_disc) / 2.0   # upper inflection composition

    EPS = 1e-12
    ones = np.ones_like(T_arr)
    F_eq = adapter['F_eq_vec']

    # Evaluate F at interval boundaries: (EPS, x_lo, x_hi, 1-EPS)
    # Three roots exist iff F changes sign in each of the three sub-intervals
    f1 = F_eq(EPS * ones, T_arr, P_arr)
    f2 = F_eq(x_lo, T_arr, P_arr)
    f3 = F_eq(x_hi, T_arr, P_arr)
    f4 = F_eq((1.0 - EPS) * ones, T_arr, P_arr)

    return valid & (f1 * f2 < 0) & (f2 * f3 < 0) & (f3 * f4 < 0)


def find_roots_vec(T_arr, P_arr, adapter):
    """
    Find roots x1 (near 0) and x3 (near 1) of F(x)=0 at each (T, P).

    Uses vectorized bisection on x across all points simultaneously.

    Returns (x1_arr, x3_arr).  Values are NaN where no valid roots exist.
    """
    omega = adapter['omega_vec'](T_arr, P_arr)
    disc = adapter['disc_vec'](omega, T_arr, P_arr)
    valid = (omega > 0) & (disc > 0)

    sqrt_disc = np.sqrt(np.where(disc > 0, disc, 1.0))
    x_infl_lo = (1.0 - sqrt_disc) / 2.0
    x_infl_hi = (1.0 + sqrt_disc) / 2.0

    EPS = 1e-12
    F_eq = adapter['F_eq_vec']
    n = len(T_arr)

    # ── Root 1 (HDL-like, x ≈ 0): bisect in (EPS, x_infl_lo) ──────────
    # 35 iterations on (0, 0.5) yields precision ≈ 0.5/2^35 ≈ 1.5e-11
    x_lo1 = np.full(n, EPS)
    x_hi1 = x_infl_lo.copy()
    f_a1 = F_eq(x_lo1, T_arr, P_arr)

    for _ in range(35):
        x_mid = (x_lo1 + x_hi1) / 2.0
        f_mid = F_eq(x_mid, T_arr, P_arr)
        same_sign = f_a1 * f_mid > 0
        x_lo1 = np.where(same_sign, x_mid, x_lo1)
        f_a1 = np.where(same_sign, f_mid, f_a1)
        x_hi1 = np.where(~same_sign, x_mid, x_hi1)

    x1 = np.where(valid, (x_lo1 + x_hi1) / 2.0, np.nan)

    # ── Root 3 (LDL-like, x ≈ 1): bisect in (x_infl_hi, 1-EPS) ────────
    x_lo3 = x_infl_hi.copy()
    x_hi3 = np.full(n, 1.0 - EPS)
    f_a3 = F_eq(x_lo3, T_arr, P_arr)

    for _ in range(35):
        x_mid = (x_lo3 + x_hi3) / 2.0
        f_mid = F_eq(x_mid, T_arr, P_arr)
        same_sign = f_a3 * f_mid > 0
        x_lo3 = np.where(same_sign, x_mid, x_lo3)
        f_a3 = np.where(same_sign, f_mid, f_a3)
        x_hi3 = np.where(~same_sign, x_mid, x_hi3)

    x3 = np.where(valid, (x_lo3 + x_hi3) / 2.0, np.nan)

    return x1, x3


# ═══════════════════════════════════════════════════════════════════════════
# Spinodal
# ═══════════════════════════════════════════════════════════════════════════

def compute_spinodal_fast(p_arr, adapter):
    """
    Compute the liquid-liquid spinodal curve using parallel bisection.

    Parameters
    ----------
    p_arr : 1D array
        Pressures in MPa (all above LLCP pressure).
    adapter : dict
        Model adapter with vectorized callables.

    Returns
    -------
    dict with same keys as the original compute_spinodal_curve:
        T_K, p_MPa (closed curve), T_upper, T_lower, p_array,
        x_lo_upper, x_hi_upper, x_lo_lower, x_hi_lower.
    """
    T_LLCP = adapter['T_LLCP']
    p_LLCP = adapter['p_LLCP']
    p_arr = np.asarray(p_arr, dtype=float)
    n = len(p_arr)

    def _h3r(T_arr):
        return has_three_roots_vec(T_arr, p_arr, adapter)

    # ── Step 1: Find a temperature inside the 3-root region ──────────────
    # Near the LLCP the 3-root dome is very narrow (< 0.01 K at dP ≈ 0.5),
    # so we need fine steps close to T_LLCP before switching to coarser steps.
    scan_temps = np.concatenate([
        np.arange(T_LLCP - 0.005, T_LLCP - 1.5, -0.005),   # fine near LLCP
        np.arange(T_LLCP - 1.5, T_LLCP - 5.0, -0.5),
        np.arange(T_LLCP - 5.0, 50.0, -2.0),
    ])

    T_inside = np.full(n, np.nan)
    found = np.zeros(n, dtype=bool)

    for T_test in scan_temps:
        T_arr = np.full(n, T_test)
        mask = ~found & _h3r(T_arr)
        T_inside[mask] = T_test
        found |= mask
        if found.all():
            break

    # Filter to pressures where 3-root region was found
    valid = found
    if not valid.any():
        empty = np.array([])
        return {
            'T_K': np.array([T_LLCP]),
            'p_MPa': np.array([p_LLCP]),
            'T_upper': empty, 'T_lower': empty,
            'x_lo_upper': empty, 'x_hi_upper': empty,
            'x_lo_lower': empty, 'x_hi_lower': empty,
            'p_array': empty,
        }

    # ── Step 2: Upper spinodal — bisect [T_inside, T_LLCP + 5] ──────────
    T_lo = np.where(valid, T_inside, T_LLCP)
    T_hi = np.full(n, T_LLCP + 5.0)

    for _ in range(45):
        T_mid = (T_lo + T_hi) / 2.0
        has_roots = _h3r(T_mid) & valid
        T_lo = np.where(has_roots, T_mid, T_lo)
        T_hi = np.where(~has_roots, T_mid, T_hi)

    T_upper_all = (T_lo + T_hi) / 2.0

    # ── Step 3: Lower spinodal — find local lower boundary ──────────────
    # Scan downward from T_inside with exponentially increasing steps to
    # find the nearest T *without* 3 roots.  This gives the local lower
    # boundary of the same contiguous 3-root region containing T_inside,
    # rather than the absolute lowest T with 3 roots (which may belong to
    # a disconnected low-T region, e.g. caupin has a separate 3-root band
    # at 16-127 K with a ~90 K gap from the LLCP dome).
    T_hi_bnd = np.where(valid, T_inside, T_LLCP)   # last T with roots
    T_lo_bnd = np.full(n, 5.0)                       # first T without
    bracketed = np.zeros(n, dtype=bool)
    T_cursor = T_inside.copy()
    step = np.full(n, 0.01)

    for _ in range(40):
        T_test = np.maximum(T_cursor - step, 5.0)
        has = _h3r(T_test) & valid

        # Still has roots: advance cursor, widen step
        advancing = has & ~bracketed
        T_cursor = np.where(advancing, T_test, T_cursor)
        T_hi_bnd = np.where(advancing, T_test, T_hi_bnd)
        step = np.where(advancing, step * 2, step)

        # No roots: gap found — record bracket
        found_now = ~has & valid & ~bracketed
        T_lo_bnd = np.where(found_now, T_test, T_lo_bnd)
        bracketed = bracketed | found_now

        # Hit floor (5 K): region extends to very low T
        at_floor = (T_test <= 5.0) & valid & ~bracketed
        T_lo_bnd = np.where(at_floor, 5.0, T_lo_bnd)
        bracketed = bracketed | at_floor

        if bracketed[valid].all():
            break

    # Bisect to refine the local boundary
    for _ in range(50):
        T_mid = (T_lo_bnd + T_hi_bnd) / 2.0
        has_roots = _h3r(T_mid) & valid
        T_hi_bnd = np.where(has_roots, T_mid, T_hi_bnd)
        T_lo_bnd = np.where(~has_roots, T_mid, T_lo_bnd)

    T_lower_all = (T_lo_bnd + T_hi_bnd) / 2.0

    # ── Filter: remove points from disconnected (non-LLCP) 3-root regions ─
    # Near P_LLCP, the scan may find a low-T 3-root region disconnected from
    # the LLCP dome. Detect this via large jumps in T_upper: scan from high P
    # to low P and remove points where T_upper drops by >30K between neighbors.
    idx = np.where(valid)[0]
    T_upper_raw = T_upper_all[idx]
    T_lower_raw = T_lower_all[idx]
    p_raw = p_arr[idx]

    connected = np.ones(len(idx), dtype=bool)
    if len(idx) > 1:
        # Scan from highest P (last index) to lowest P (first index)
        for i in range(len(idx) - 2, -1, -1):
            if T_upper_raw[i + 1] - T_upper_raw[i] > 30.0:
                connected[i] = False
            elif not connected[i + 1]:
                # If the next point is already disconnected, this might be too
                connected[i] = False

    # Apply filter
    keep = connected
    idx = idx[keep]

    # ── Extract valid pressures ──────────────────────────────────────────
    T_upper = T_upper_all[idx]
    T_lower = T_lower_all[idx]
    p_valid = p_arr[idx]

    # ── Get spinodal compositions (F_eq roots = minima of G(x)) ──────────
    # At the spinodal boundary, the metastable minimum of G(x) is about to
    # vanish.  Evaluate F_eq roots slightly inside the dome where three
    # roots still exist.  x1 (near 0) = HDL-like, x3 (near 1) = LDL-like.
    # Use a fixed small delta (not dome_width-dependent) because caupin has
    # dome_width ~ 200K (T_lower ~ 16K) making 0.1% of dome > 0.2K, which
    # can exceed the near-LLCP dome neck width.
    delta = 0.001

    # Upper spinodal
    T_near_upper = T_upper - delta
    x1_up, x3_up = find_roots_vec(T_near_upper, p_valid, adapter)

    # Lower spinodal
    T_near_lower = T_lower + delta
    x1_dn, x3_dn = find_roots_vec(T_near_lower, p_valid, adapter)

    # ── Build closed curve ───────────────────────────────────────────────
    T_curve = np.concatenate([[T_LLCP], T_upper, T_lower[::-1], [T_LLCP]])
    p_curve = np.concatenate([[p_LLCP], p_valid, p_valid[::-1], [p_LLCP]])

    return {
        'T_K': T_curve,
        'p_MPa': p_curve,
        'T_upper': T_upper,
        'T_lower': T_lower,
        'x_lo_upper': x1_up,
        'x_hi_upper': x3_up,
        'x_lo_lower': x1_dn,
        'x_hi_lower': x3_dn,
        'p_array': p_valid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Binodal
# ═══════════════════════════════════════════════════════════════════════════

def compute_binodal_fast(p_arr, spinodal, adapter):
    """
    Compute the liquid-liquid binodal (coexistence) curve.

    Takes pre-computed spinodal to avoid redundant recomputation.

    Parameters
    ----------
    p_arr : 1D array
        Pressures in MPa.
    spinodal : dict
        Output of compute_spinodal_fast.
    adapter : dict
        Model adapter with vectorized callables.

    Returns
    -------
    dict with same keys as the original compute_binodal_curve.
    """
    T_LLCP = adapter['T_LLCP']
    p_LLCP = adapter['p_LLCP']

    p_spin = spinodal['p_array']
    T_spin_upper = spinodal['T_upper']
    T_spin_lower = spinodal['T_lower']

    if len(p_spin) == 0:
        return {
            'T_K': np.array([T_LLCP]),
            'p_MPa': np.array([p_LLCP]),
            'x': np.array([0.5]),
            'T_binodal': np.array([]),
            'x_lo': np.array([]),
            'x_hi': np.array([]),
            'p_array': np.array([]),
        }

    ns = len(p_spin)

    # T bounds for scan (slightly inside spinodal).
    # Near the LLCP, the binodal is within ~0.002K of T_upper and the
    # dome can be <0.01K wide, so margins must be tiny.
    T_lo_bound = T_spin_lower + 0.0001
    T_hi_bound = T_spin_upper - 0.0001

    # Skip pressures where bounds are invalid
    scan_valid = T_hi_bound > T_lo_bound

    # ── Scan phase: evaluate delta_g at multiple temperatures ────────────
    # delta_g = g_mix(x3) - g_mix(x1) is the Gibbs energy difference between
    # the LDL-like (x3) and HDL-like (x1) minima of G_mix(x).  The binodal
    # is where delta_g = 0 (equal Gibbs energy = thermodynamic coexistence).
    # Use geometric spacing from T_upper downward so the scan concentrates
    # resolution near the upper spinodal — where the binodal actually lives
    # for near-LLCP pressures.
    n_scan = 30
    dome_width = T_hi_bound - T_lo_bound
    # Offsets from T_hi_bound: geomspace from 0.0001K to full dome width
    min_offset = np.full(ns, 0.0001)
    max_offset = np.maximum(dome_width, 0.001)
    # Build scan offsets per pressure (shape: n_scan x ns)
    scan_offsets = np.zeros((n_scan, ns))
    for j in range(n_scan):
        frac = j / (n_scan - 1)
        scan_offsets[j] = min_offset * (max_offset / min_offset) ** frac

    dg_grid = np.full((n_scan, ns), np.nan)
    T_scan_grid = np.full((n_scan, ns), np.nan)

    for j in range(n_scan):
        T_scan = T_hi_bound - scan_offsets[j]
        T_scan = np.where(scan_valid & (T_scan > T_lo_bound), T_scan, T_LLCP)
        T_scan_grid[j] = T_scan

        x1, x3 = find_roots_vec(T_scan, p_spin, adapter)
        g1 = adapter['g_mix_vec'](x1, T_scan, p_spin)
        g3 = adapter['g_mix_vec'](x3, T_scan, p_spin)
        dg_grid[j] = g3 - g1

    # ── Find sign changes in delta_g for each pressure ───────────────────
    T_bracket_lo = np.full(ns, np.nan)
    T_bracket_hi = np.full(ns, np.nan)
    bracket_found = np.zeros(ns, dtype=bool)

    for j in range(n_scan - 1):
        dg_j = dg_grid[j]
        dg_j1 = dg_grid[j + 1]
        sign_change = (
            ~np.isnan(dg_j) & ~np.isnan(dg_j1) &
            (dg_j * dg_j1 < 0) & scan_valid & ~bracket_found
        )
        if sign_change.any():
            T_j = T_scan_grid[j]
            T_j1 = T_scan_grid[j + 1]
            T_bracket_hi = np.where(sign_change, np.maximum(T_j, T_j1), T_bracket_hi)
            T_bracket_lo = np.where(sign_change, np.minimum(T_j, T_j1), T_bracket_lo)
            bracket_found |= sign_change

    if not bracket_found.any():
        return {
            'T_K': np.array([T_LLCP]),
            'p_MPa': np.array([p_LLCP]),
            'x': np.array([0.5]),
            'T_binodal': np.array([]),
            'x_lo': np.array([]),
            'x_hi': np.array([]),
            'p_array': np.array([]),
        }

    # ── Refine with bisection on T ───────────────────────────────────────
    # Only bisect pressures where we found a bracket
    bix = np.where(bracket_found)[0]
    p_bix = p_spin[bix]
    T_lo_b = T_bracket_lo[bix]
    T_hi_b = T_bracket_hi[bix]

    # Evaluate delta_g at T_lo to establish sign
    x1, x3 = find_roots_vec(T_lo_b, p_bix, adapter)
    g1 = adapter['g_mix_vec'](x1, T_lo_b, p_bix)
    g3 = adapter['g_mix_vec'](x3, T_lo_b, p_bix)
    dg_lo = g3 - g1

    # 25 iterations on a ~1 K bracket → precision ≈ 1/2^25 ≈ 3e-8 K
    for _ in range(25):
        T_mid = (T_lo_b + T_hi_b) / 2.0
        x1, x3 = find_roots_vec(T_mid, p_bix, adapter)
        g1 = adapter['g_mix_vec'](x1, T_mid, p_bix)
        g3 = adapter['g_mix_vec'](x3, T_mid, p_bix)
        dg_mid = g3 - g1

        same_sign = dg_lo * dg_mid > 0
        T_lo_b = np.where(same_sign, T_mid, T_lo_b)
        dg_lo = np.where(same_sign, dg_mid, dg_lo)
        T_hi_b = np.where(~same_sign, T_mid, T_hi_b)

    T_eq = (T_lo_b + T_hi_b) / 2.0

    # Get final x values at binodal
    x1_final, x3_final = find_roots_vec(T_eq, p_bix, adapter)

    # Filter out any remaining NaN values
    good = ~np.isnan(T_eq) & ~np.isnan(x1_final) & ~np.isnan(x3_final)
    T_binodal = T_eq[good]
    x_lo = x1_final[good]
    x_hi = x3_final[good]
    p_valid = p_bix[good]

    # Build T-P curve as a single branch (no closed loop).
    # The LLCP itself is NOT included (it's a separate marker).
    if len(T_binodal) > 0:
        T_curve = T_binodal
        p_curve = p_valid
        x_curve = x_lo  # x_lo for reference; both branches at same T,P
    else:
        T_curve = np.array([T_LLCP])
        p_curve = np.array([p_LLCP])
        x_curve = np.array([0.5])

    return {
        'T_K': T_curve,
        'p_MPa': p_curve,
        'x': x_curve,
        'T_binodal': T_binodal,
        'x_lo': x_lo,
        'x_hi': x_hi,
        'p_array': p_valid,
    }
