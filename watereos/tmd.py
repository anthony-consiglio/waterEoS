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

    # At each pressure, find the highest-T sign change bracket.
    # Vectorized: build an (n_P, n_scan-1) boolean mask of sign-change
    # candidates and use a weighted argmax to pick the rightmost True per row.
    # Replaces an O(n_P * n_scan) Python double-loop with one numpy pass.
    left, right = alpha_grid[:, :-1], alpha_grid[:, 1:]
    crossings = np.isfinite(left) & np.isfinite(right) & (left * right < 0)
    # Weight True cells by their column index (+1 so all-False stays at 0);
    # argmax then yields the rightmost True column. Rows with no crossing
    # are filtered out by the has_crossing mask.
    col_idx = np.arange(1, n_scan)  # +1 offset so weight 0 stays distinct
    last_j = np.argmax(crossings * col_idx, axis=1)
    has_crossing = crossings.any(axis=1)

    result = np.full(n_P, np.nan)

    if not has_crossing.any():
        return float(result[0]) if scalar else result

    ip_map = np.where(has_crossing)[0]
    j_sel = last_j[has_crossing]
    T_lo_b = T_scan[j_sel]
    T_hi_b = T_scan[j_sel + 1]
    P_b = P_arr[has_crossing]

    # Vectorized bisection
    sign_lo = np.sign(compute_batch(T_lo_b, P_b)['alpha'])
    for _ in range(30):
        T_mid = 0.5 * (T_lo_b + T_hi_b)
        sign_mid = np.sign(compute_batch(T_mid, P_b)['alpha'])
        same = sign_mid == sign_lo
        T_lo_b = np.where(same, T_mid, T_lo_b)
        T_hi_b = np.where(~same, T_mid, T_hi_b)

    T_final = 0.5 * (T_lo_b + T_hi_b)
    result[ip_map] = T_final  # fancy-indexed assign, no Python loop

    return float(result[0]) if scalar else result
