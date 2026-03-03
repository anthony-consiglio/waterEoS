"""
Kauzmann temperature solver.

The Kauzmann temperature is where the entropy of the metastable liquid
equals that of ice Ih: S_liquid(T, P) = S_ice_Ih(T, P).  Below this
temperature the liquid would have less entropy than the crystal, which
is considered a thermodynamic lower bound on the metastable liquid.

Algorithm: evaluate dS = S_liquid - S_ice on a (T, P) meshgrid, detect
sign changes in dS along the T axis for each pressure, then refine with
vectorized bisection.  SeaFreeze provides ice Ih entropy; the liquid
model's compute_batch provides liquid entropy.
"""

import numpy as np
import seafreeze.seafreeze as sf


def _sf_entropy(T_arr, P_arr):
    """Compute ice Ih entropy via SeaFreeze scatter mode.

    SeaFreeze scatter mode uses an object array of (P, T) tuples, one per
    point, rather than the grid-mode [P_1d, T_1d] format which would create
    a full meshgrid.  This is appropriate when we have matched (T, P) pairs.
    """
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

    # Evaluate liquid and ice entropy on the full (n_P × n_scan) meshgrid
    # in a single vectorized call for speed.
    T_grid, P_grid = np.meshgrid(T_scan, P_arr)
    T_flat, P_flat = T_grid.ravel(), P_grid.ravel()

    S_liq = compute_batch(T_flat, P_flat)['S'].reshape(n_P, n_scan)
    S_ice = _sf_entropy(T_flat, P_flat).reshape(n_P, n_scan)
    # dS > 0: liquid has more entropy (normal); dS = 0: Kauzmann point
    dS = S_liq - S_ice

    # At each pressure, find all sign changes in dS and pick the one
    # whose midpoint is closest to T_target.  Multiple crossings can
    # occur (e.g. HDL Kauzmann ~185 K, LDL Kauzmann ~155 K).
    T_lo_list = []   # lower bracket endpoints
    T_hi_list = []   # upper bracket endpoints
    P_list = []      # pressures with valid crossings
    ip_map = []      # maps bracket index -> original pressure index

    for ip in range(n_P):
        row = dS[ip]     # dS vs T at this pressure
        # Detect sign changes: product < 0 means dS crosses zero
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

    # Vectorized bisection: evaluate both models at each midpoint and
    # narrow the bracket toward dS = 0.
    # 30 iterations on a ~1 K bracket → precision ≈ 1/2^30 ≈ 1e-9 K.
    S_lo_liq = compute_batch(T_lo_b, P_b)['S']
    S_lo_ice = _sf_entropy(T_lo_b, P_b)
    sign_lo = np.sign(S_lo_liq - S_lo_ice)  # track which side has dS > 0

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
