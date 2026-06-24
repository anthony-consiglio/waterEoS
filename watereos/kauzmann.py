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

from watereos._seafreeze_native import getProp_scatter as _sf_scatter


def _sf_entropy(T_arr, P_arr):
    """Compute ice Ih entropy at matched (P, T) pairs.

    Uses the native (Rust) SeaFreeze drop-in: it returns the full property
    set in one call, but evaluating only the d/dT G derivative is cheap
    enough that the broader output set isn't worth narrowing for. The
    previous code had to pass ``tdvSpec=('S',)`` to the Python SeaFreeze
    because the default tdv set was ~6 derivatives slower; the Rust port
    evaluates all properties from the same six base derivatives anyway.
    """
    out = _sf_scatter('Ih', np.asarray(P_arr, dtype=float),
                      np.asarray(T_arr, dtype=float))
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

    # At each pressure, find the sign-change in dS whose midpoint is
    # closest to T_target.  Multiple crossings can occur (e.g. HDL
    # Kauzmann ~185 K, LDL Kauzmann ~155 K); T_target picks a branch.
    #
    # Vectorized: build a boolean (n_P, n_scan-1) mask of crossing
    # candidates, compute |T_midpoint - T_target| for each, mask
    # non-crossings to +inf, and argmin per row.
    left, right = dS[:, :-1], dS[:, 1:]
    crossings = np.isfinite(left) & np.isfinite(right) & (left * right < 0)
    T_mid_all = 0.5 * (T_scan[:-1] + T_scan[1:])  # (n_scan-1,)
    distances = np.where(crossings, np.abs(T_mid_all - T_target), np.inf)
    best_j = np.argmin(distances, axis=1)
    has_crossing = crossings.any(axis=1)

    result = np.full(n_P, np.nan)

    if not has_crossing.any():
        return float(result[0]) if scalar else result

    ip_map = np.where(has_crossing)[0]
    j_sel = best_j[has_crossing]
    T_lo_b = T_scan[j_sel]
    T_hi_b = T_scan[j_sel + 1]
    P_b = P_arr[has_crossing]

    # Vectorized bisection: liquid is cheap to re-evaluate per step (the
    # model's compute_batch is the Rust/JAX fast path). Ice is expensive
    # (SeaFreeze scatter mode), so we reuse the S_ice grid already
    # computed for bracket detection via linear interpolation in T.
    # Accuracy: with ~1 K T_scan spacing the interp error in dS is
    # ~1e-3 J/(kg·K), translating to ~1e-4 K uncertainty in T_kauz, far
    # below any physically relevant precision.
    S_ice_rows = S_ice[has_crossing]  # (n_kept, n_scan)

    def _interp_ice_S(T_in):
        j = np.searchsorted(T_scan, T_in) - 1
        j = np.clip(j, 0, n_scan - 2)
        t0 = T_scan[j]
        t1 = T_scan[j + 1]
        s0 = np.take_along_axis(S_ice_rows, j[:, None], axis=1).ravel()
        s1 = np.take_along_axis(S_ice_rows, (j + 1)[:, None], axis=1).ravel()
        return s0 + (s1 - s0) * (T_in - t0) / (t1 - t0)

    S_lo_liq = compute_batch(T_lo_b, P_b)['S']
    S_lo_ice = _interp_ice_S(T_lo_b)
    sign_lo = np.sign(S_lo_liq - S_lo_ice)

    for _ in range(30):
        T_mid = 0.5 * (T_lo_b + T_hi_b)
        S_mid_liq = compute_batch(T_mid, P_b)['S']
        S_mid_ice = _interp_ice_S(T_mid)
        same = np.sign(S_mid_liq - S_mid_ice) == sign_lo
        T_lo_b = np.where(same, T_mid, T_lo_b)
        T_hi_b = np.where(~same, T_mid, T_hi_b)

    T_final = 0.5 * (T_lo_b + T_hi_b)
    result[ip_map] = T_final

    return float(result[0]) if scalar else result
