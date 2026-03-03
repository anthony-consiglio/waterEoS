"""
Kauzmann temperature solver.

Finds T where S_liquid(T, P) = S_ice_Ih(T, P) using SeaFreeze for ice Ih.
Uses vectorized scan + bisection.
"""

import numpy as np
import seafreeze.seafreeze as sf


def _sf_entropy(T_arr, P_arr):
    """Compute ice Ih entropy via SeaFreeze scatter mode."""
    n = len(T_arr)
    PT = np.empty(n, dtype=object)
    PT[:] = list(zip(P_arr, T_arr))
    out = sf.getProp(PT, 'Ih')
    return np.asarray(out.S).ravel()


def compute_kauzmann_temperature(P_MPa, compute_batch,
                                 T_target=185.0,
                                 T_scan_lo=100.0, T_scan_hi=280.0, n_scan=300):
    """Compute Kauzmann temperature (S_liquid = S_ice_Ih) at given pressure(s).

    At each pressure, scans S_liquid(T) - S_ice(T) for zero-crossings and
    picks the crossing nearest to *T_target*. Refines with 30 iterations of
    bisection using both the liquid model and SeaFreeze ice Ih.

    Parameters
    ----------
    P_MPa : float or array_like
        Pressure(s) in MPa.
    compute_batch : callable
        Model's compute_batch(T_arr, P_arr) -> dict with 'S' key.
    T_target : float
        Target temperature (K) for selecting among multiple crossings.
        Use ~185 for HDL Kauzmann, ~155 for LDL Kauzmann.
    T_scan_lo, T_scan_hi : float
        Temperature scan range in K.
    n_scan : int
        Number of temperature points in the initial scan.

    Returns
    -------
    float or numpy.ndarray
        Kauzmann temperature(s) in K. NaN where no crossing is found.
    """
    scalar = np.ndim(P_MPa) == 0
    P_arr = np.atleast_1d(np.asarray(P_MPa, dtype=float))
    n_P = len(P_arr)

    T_scan = np.linspace(T_scan_lo, T_scan_hi, n_scan)

    # Evaluate liquid and ice entropy on full (T, P) meshgrid
    T_grid, P_grid = np.meshgrid(T_scan, P_arr)
    T_flat, P_flat = T_grid.ravel(), P_grid.ravel()

    S_liq = compute_batch(T_flat, P_flat)['S'].reshape(n_P, n_scan)
    S_ice = _sf_entropy(T_flat, P_flat).reshape(n_P, n_scan)
    dS = S_liq - S_ice

    # At each pressure, find the crossing nearest to 185 K
    T_lo_list = []
    T_hi_list = []
    P_list = []
    ip_map = []

    for ip in range(n_P):
        row = dS[ip]
        xings = []
        for j in range(n_scan - 1):
            if np.isfinite(row[j]) and np.isfinite(row[j + 1]) and row[j] * row[j + 1] < 0:
                xings.append(j)
        if xings:
            best_j = min(xings, key=lambda j: abs(0.5 * (T_scan[j] + T_scan[j + 1]) - T_target))
            T_lo_list.append(T_scan[best_j])
            T_hi_list.append(T_scan[best_j + 1])
            P_list.append(P_arr[ip])
            ip_map.append(ip)

    result = np.full(n_P, np.nan)

    if not T_lo_list:
        return float(result[0]) if scalar else result

    T_lo_b = np.array(T_lo_list)
    T_hi_b = np.array(T_hi_list)
    P_b = np.array(P_list)

    # Vectorized bisection using both model and SeaFreeze
    S_lo_liq = compute_batch(T_lo_b, P_b)['S']
    S_lo_ice = _sf_entropy(T_lo_b, P_b)
    sign_lo = np.sign(S_lo_liq - S_lo_ice)

    for _ in range(30):
        T_mid = 0.5 * (T_lo_b + T_hi_b)
        S_mid_liq = compute_batch(T_mid, P_b)['S']
        S_mid_ice = _sf_entropy(T_mid, P_b)
        same = np.sign(S_mid_liq - S_mid_ice) == sign_lo
        T_lo_b = np.where(same, T_mid, T_lo_b)
        T_hi_b = np.where(~same, T_mid, T_hi_b)

    T_final = 0.5 * (T_lo_b + T_hi_b)
    for i, ip in enumerate(ip_map):
        result[ip] = T_final[i]

    return float(result[0]) if scalar else result
