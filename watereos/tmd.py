"""
Temperature of Maximum Density (TMD) solver.

Finds T where the thermal expansion coefficient alpha(T, P) = 0.
Uses vectorized scan + bisection on compute_batch output.
"""

import numpy as np


def compute_tmd_temperature(P_MPa, compute_batch,
                            T_scan_lo=125.0, T_scan_hi=350.0, n_scan=400):
    """Compute TMD temperature (alpha=0) at given pressure(s).

    At each pressure, scans alpha(T) for zero-crossings and picks the
    highest-temperature crossing (the main TMD branch through ~277 K at
    atmospheric pressure). Refines with 30 iterations of bisection.

    Parameters
    ----------
    P_MPa : float or array_like
        Pressure(s) in MPa.
    compute_batch : callable
        Model's compute_batch(T_arr, P_arr) -> dict with 'alpha' key.
    T_scan_lo, T_scan_hi : float
        Temperature scan range in K.
    n_scan : int
        Number of temperature points in the initial scan.

    Returns
    -------
    float or numpy.ndarray
        TMD temperature(s) in K. NaN where no crossing is found.
    """
    scalar = np.ndim(P_MPa) == 0
    P_arr = np.atleast_1d(np.asarray(P_MPa, dtype=float))
    n_P = len(P_arr)

    T_scan = np.linspace(T_scan_lo, T_scan_hi, n_scan)

    # Evaluate alpha on full (T, P) meshgrid in one batch call
    T_grid, P_grid = np.meshgrid(T_scan, P_arr)
    batch = compute_batch(T_grid.ravel(), P_grid.ravel())
    alpha_grid = batch['alpha'].reshape(n_P, n_scan)

    # At each pressure, find the highest-T sign change bracket
    T_lo_list = []
    T_hi_list = []
    P_list = []
    ip_map = []  # maps bracket index -> original pressure index

    for ip in range(n_P):
        row = alpha_grid[ip]
        last_j = None
        for j in range(n_scan - 1):
            if np.isfinite(row[j]) and np.isfinite(row[j + 1]) and row[j] * row[j + 1] < 0:
                last_j = j
        if last_j is not None:
            T_lo_list.append(T_scan[last_j])
            T_hi_list.append(T_scan[last_j + 1])
            P_list.append(P_arr[ip])
            ip_map.append(ip)

    # Initialize result with NaN
    result = np.full(n_P, np.nan)

    if not T_lo_list:
        return float(result[0]) if scalar else result

    T_lo_b = np.array(T_lo_list)
    T_hi_b = np.array(T_hi_list)
    P_b = np.array(P_list)

    # Vectorized bisection
    sign_lo = np.sign(compute_batch(T_lo_b, P_b)['alpha'])
    for _ in range(30):
        T_mid = 0.5 * (T_lo_b + T_hi_b)
        sign_mid = np.sign(compute_batch(T_mid, P_b)['alpha'])
        same = sign_mid == sign_lo
        T_lo_b = np.where(same, T_mid, T_lo_b)
        T_hi_b = np.where(~same, T_mid, T_hi_b)

    T_final = 0.5 * (T_lo_b + T_hi_b)
    for i, ip in enumerate(ip_map):
        result[ip] = T_final[i]

    return float(result[0]) if scalar else result
