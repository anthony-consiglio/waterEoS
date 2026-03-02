"""
Computation helpers — wraps watereos.getProp and phase diagram functions.
"""

import numpy as np
from watereos_gui.utils.model_registry import MODEL_REGISTRY


def compute_property_curves(model_key, prop_key, T_range, P_range,
                            n_curves, n_points, isobar_mode):
    """
    Compute a family of isobars or isotherms for a single model.

    Returns
    -------
    dict with keys:
        'x_values' : list of 1-D arrays (T or P values along each curve)
        'y_values' : list of 1-D arrays (property values)
        'curve_labels' : list of str labels
        'x_label' : str
        'y_label' : str
        'title'   : str
    """
    from watereos import getProp

    T_min, T_max = T_range
    P_min, P_max = P_range

    T_pts = np.linspace(T_min, T_max, n_points)
    P_pts = np.linspace(P_min, P_max, n_points)

    x_values = []
    y_values = []
    curve_labels = []

    info = MODEL_REGISTRY[model_key]

    if isobar_mode:
        # Each curve is an isobar (fixed P, sweep T)
        pressures = np.linspace(P_min, P_max, n_curves)
        for P in pressures:
            PT = np.array([np.array([P]), T_pts], dtype=object)
            result = getProp(PT, model_key)
            y = getattr(result, prop_key, None)
            if y is None:
                continue
            y = np.asarray(y).flatten()
            x_values.append(T_pts)
            y_values.append(y)
            curve_labels.append(f'{P:.1f} MPa')
        x_label = 'Temperature [K]'
    else:
        # Each curve is an isotherm (fixed T, sweep P)
        temperatures = np.linspace(T_min, T_max, n_curves)
        for T in temperatures:
            PT = np.array([P_pts, np.array([T])], dtype=object)
            result = getProp(PT, model_key)
            y = getattr(result, prop_key, None)
            if y is None:
                continue
            y = np.asarray(y).flatten()
            x_values.append(P_pts)
            y_values.append(y)
            curve_labels.append(f'{T:.1f} K')
        x_label = 'Pressure [MPa]'

    from watereos_gui.utils.model_registry import get_display_label
    y_label = get_display_label(prop_key)
    title = f'{info.display_name} — {y_label}'

    return {
        'x_values': x_values,
        'y_values': y_values,
        'curve_labels': curve_labels,
        'x_label': x_label,
        'y_label': y_label,
        'title': title,
        'model_key': model_key,
    }


def compute_multi_model_curves(model_keys, prop_key, T_range, P_range,
                               n_curves, n_points, isobar_mode):
    """
    Compute property curves for multiple models.

    Returns
    -------
    dict mapping model_key -> compute_property_curves result
    """
    results = {}
    for mk in model_keys:
        results[mk] = compute_property_curves(
            mk, prop_key, T_range, P_range, n_curves, n_points, isobar_mode,
        )
    return results


def compute_property_surface(model_key, prop_key, T_range, P_range, n_points):
    """
    Compute a 2-D grid of a property over T and P.

    Returns
    -------
    dict with keys:
        'T_grid' : 2-D array (n_P, n_T)
        'P_grid' : 2-D array (n_P, n_T)
        'Z'      : 2-D array (n_P, n_T) — property values
        'prop_key', 'model_key'
    """
    from watereos import getProp

    T_min, T_max = T_range
    P_min, P_max = P_range

    T_pts = np.linspace(T_min, T_max, n_points)
    P_pts = np.linspace(P_min, P_max, n_points)

    PT = np.array([P_pts, T_pts], dtype=object)
    result = getProp(PT, model_key)
    Z = getattr(result, prop_key, None)
    if Z is None:
        raise ValueError(f'Property {prop_key} not available for model {model_key}')
    Z = np.asarray(Z, dtype=float)

    T_grid, P_grid = np.meshgrid(T_pts, P_pts)

    return {
        'T_grid': T_grid,
        'P_grid': P_grid,
        'Z': Z,
        'prop_key': prop_key,
        'model_key': model_key,
    }


def compute_phase_diagram_data(model_key, n_pressures=150):
    """
    Compute full phase diagram for a two-state model.

    Returns
    -------
    dict with keys: 'LLCP', 'spinodal', 'binodal' (model-dependent),
    'tmd', 'widom' (model-dependent), 'ice_ih_liquidus', 'ice_iii_liquidus',
    'nucleation_ih', 'nucleation_iii', 'kauzmann', 'triple_point',
    'melting_point' (model-independent).
    """
    if model_key == 'duska2020':
        from duska_eos.phase_diagram import find_LLCP, compute_spinodal_curve, compute_binodal_curve
    elif model_key == 'holten2014':
        from holten_eos.phase_diagram import find_LLCP, compute_spinodal_curve, compute_binodal_curve
    elif model_key == 'caupin2019':
        from caupin_eos.phase_diagram import find_LLCP, compute_spinodal_curve, compute_binodal_curve
    else:
        raise ValueError(f'No phase diagram for model {model_key}')

    llcp = find_LLCP()
    P_llcp = float(llcp['p_MPa'])
    P_max = 300.0

    # Dense spacing near LLCP for smooth curve onset, sparser far away
    n_near = n_pressures // 3
    n_far = n_pressures - n_near
    p_near = np.linspace(P_llcp + 0.05, P_llcp + 10.0, n_near, endpoint=False)
    p_far = np.linspace(P_llcp + 10.0, P_max, n_far)
    p_arr = np.concatenate([p_near, p_far])
    spinodal = compute_spinodal_curve(p_range_MPa=p_arr)
    binodal = compute_binodal_curve(p_range_MPa=p_arr, spinodal=spinodal)

    # Split spinodal into HDL (upper) and LDL (lower) branches
    hdl_spinodal = None
    ldl_spinodal = None
    if 'T_upper' in spinodal and len(spinodal['T_upper']) > 0:
        hdl_spinodal = {'T_K': spinodal['T_upper'],
                        'p_MPa': spinodal['p_array']}
    if 'T_lower' in spinodal and len(spinodal['T_lower']) > 0:
        ldl_spinodal = {'T_K': spinodal['T_lower'],
                        'p_MPa': spinodal['p_array']}

    tmd = _compute_tmd_curve(model_key, llcp=llcp)
    widom = _compute_widom_line(model_key, llcp)

    nuc = _nucleation_curves()

    # Compute raw Kauzmann curves, then truncate at spinodals
    kauzmann_hdl = _kauzmann_hdl_curve()
    kauzmann_ldl = _kauzmann_ldl_curve()
    kauzmann_hdl = _truncate_at_spinodal(kauzmann_hdl, hdl_spinodal)
    kauzmann_ldl = _truncate_at_spinodal(kauzmann_ldl, ldl_spinodal)

    return {
        'LLCP': llcp,
        'spinodal': spinodal,
        'hdl_spinodal': hdl_spinodal,
        'ldl_spinodal': ldl_spinodal,
        'binodal': binodal,
        'tmd': tmd,
        'widom': widom,
        'ice_ih_liquidus': _ice_ih_liquidus(),
        'ice_iii_liquidus': _ice_iii_liquidus(),
        'nucleation_ih': nuc['ih'],
        'nucleation_iii': nuc['iii'],
        'kauzmann_hdl': kauzmann_hdl,
        'kauzmann_ldl': kauzmann_ldl,
        'triple_point': {'T_K': 251.165, 'p_MPa': 209.9},
    }


# ---------------------------------------------------------------------------
# Model-dependent curves
# ---------------------------------------------------------------------------

def _get_compute_batch(model_key):
    """Return the model-specific vectorized compute_batch function."""
    if model_key == 'caupin2019':
        from caupin_eos.core import compute_batch
    elif model_key == 'holten2014':
        from holten_eos.core import compute_batch
    elif model_key == 'duska2020':
        from duska_eos.core import compute_batch
    else:
        raise ValueError(f'No compute_batch for model {model_key}')
    return compute_batch


def _compute_tmd_curve(model_key, llcp=None, n_points=80):
    """Find the Temperature of Maximum Density (alpha=0) at each pressure.

    TMD terminates at the LLCP pressure — above the LLCP, the liquid-liquid
    transition makes the single-phase TMD concept ill-defined.

    Uses vectorized grid evaluation + parallel bisection for speed.
    """
    compute_batch = _get_compute_batch(model_key)
    info = MODEL_REGISTRY[model_key]
    P_lo = max(info.P_min, -140.0)
    # Terminate TMD at LLCP pressure
    if llcp is not None:
        P_hi = float(llcp['p_MPa'])
    else:
        P_hi = min(info.P_max, 300.0)

    if P_lo >= P_hi:
        return None

    P_pts = np.linspace(P_lo, P_hi, n_points)
    n_T_scan = 60
    T_lo, T_hi = 125.0, info.T_max
    T_scan = np.linspace(T_lo, T_hi, n_T_scan)

    # --- Single vectorized call: alpha at all (P, T) grid points ---
    T_grid, P_grid = np.meshgrid(T_scan, P_pts)
    try:
        batch = compute_batch(T_grid.ravel(), P_grid.ravel())
        alpha_grid = batch['alpha'].reshape(n_points, n_T_scan)
    except Exception:
        return None

    # --- Find first sign change along T axis for each pressure ---
    valid = ~np.isnan(alpha_grid[:, :-1]) & ~np.isnan(alpha_grid[:, 1:])
    sign_change = (alpha_grid[:, :-1] * alpha_grid[:, 1:] < 0) & valid

    has_change = sign_change.any(axis=1)
    if not has_change.any():
        return None

    first_idx = sign_change.argmax(axis=1)  # first True per row
    mask = has_change
    row_idx = np.where(mask)[0]
    j_bracket = first_idx[mask]

    P_bracket = P_pts[mask]
    T_lo_bracket = T_scan[j_bracket]
    T_hi_bracket = T_scan[j_bracket + 1]

    # Sign of alpha at the lower bracket endpoint (for bisection direction)
    alpha_at_lo = alpha_grid[row_idx, j_bracket]
    sign_lo = np.sign(alpha_at_lo)

    # --- Vectorized bisection: refine all brackets simultaneously ---
    for _ in range(15):
        T_mid = 0.5 * (T_lo_bracket + T_hi_bracket)
        batch_mid = compute_batch(T_mid, P_bracket)
        alpha_mid = batch_mid['alpha']

        nan_mid = np.isnan(alpha_mid)
        same_sign = np.sign(alpha_mid) == sign_lo
        update = ~nan_mid
        T_lo_bracket = np.where(update & same_sign, T_mid, T_lo_bracket)
        T_hi_bracket = np.where(update & ~same_sign, T_mid, T_hi_bracket)

    T_tmd = 0.5 * (T_lo_bracket + T_hi_bracket)
    return {'T_K': T_tmd, 'p_MPa': P_bracket}


def _compute_widom_line(model_key, llcp, n_points=60):
    """Find the Cp maximum locus below the LLCP pressure.

    Uses a single vectorized grid evaluation over a common T range,
    with per-pressure windowed argmax.
    """
    if llcp is None:
        return None

    compute_batch = _get_compute_batch(model_key)

    T_llcp = float(llcp['T_K'])
    P_llcp = float(llcp['p_MPa'])

    info = MODEL_REGISTRY[model_key]
    # Widom line extends from LLCP toward lower pressures and higher T
    P_lo = max(P_llcp - 100.0, info.P_min)
    P_hi = P_llcp - 0.5  # just below LLCP

    if P_lo >= P_hi:
        return None

    P_pts = np.linspace(P_hi, P_lo, n_points)

    # Compute the per-pressure T window bounds to determine global range
    dP = P_llcp - P_pts
    T_centers = T_llcp + dP * 0.3
    T_global_lo = max(float(np.min(T_centers)) - 30.0, info.T_min)
    T_global_hi = min(float(np.max(T_centers)) + 30.0, info.T_max)

    n_T_scan = 120
    T_scan = np.linspace(T_global_lo, T_global_hi, n_T_scan)

    # --- Single vectorized call: Cp at all (P, T) grid points ---
    T_grid, P_grid = np.meshgrid(T_scan, P_pts)
    try:
        batch = compute_batch(T_grid.ravel(), P_grid.ravel())
        Cp_grid = batch['Cp'].reshape(n_points, n_T_scan)
    except Exception:
        return None

    # --- Per-pressure window mask: restrict each row to [T_center±30] ---
    T_lo_per_P = np.maximum(T_centers - 30.0, info.T_min)[:, np.newaxis]
    T_hi_per_P = np.minimum(T_centers + 30.0, info.T_max)[:, np.newaxis]
    T_row = T_scan[np.newaxis, :]
    window_mask = (T_row >= T_lo_per_P) & (T_row <= T_hi_per_P)

    # Replace out-of-window and NaN values with -inf for argmax
    Cp_masked = np.where(window_mask & ~np.isnan(Cp_grid), Cp_grid, -np.inf)

    # Find argmax per pressure
    idx_max = np.argmax(Cp_masked, axis=1)
    peak_Cp = Cp_masked[np.arange(n_points), idx_max]
    has_valid = peak_Cp > -np.inf

    # Interior check: peak must not be at the edge of the valid window
    first_valid = np.argmax(window_mask, axis=1)
    last_valid = n_T_scan - 1 - np.argmax(window_mask[:, ::-1], axis=1)
    interior = (idx_max > first_valid) & (idx_max < last_valid) & has_valid

    if not interior.any():
        return None

    T_widom = T_scan[idx_max[interior]]
    P_widom = P_pts[interior]
    return {'T_K': T_widom, 'p_MPa': P_widom}


# ---------------------------------------------------------------------------
# Model-independent curves (IAPWS melting equations & empirical fits)
# ---------------------------------------------------------------------------

def _ice_ih_liquidus(n_points=100):
    """IAPWS Ice Ih melting pressure equation."""
    T_star = 273.16    # K
    p_star = 611.657e-6  # MPa
    a1, b1 = 0.119539337e7, 3.0
    a2, b2 = 0.808183159e5, 25.75
    a3, b3 = 0.333826860e4, 103.75

    T = np.linspace(251.165, 273.16, n_points)
    theta = T / T_star
    pi = 1.0 + a1 * (1.0 - theta**b1) + a2 * (1.0 - theta**b2) + a3 * (1.0 - theta**b3)
    P = pi * p_star
    return {'T_K': T, 'p_MPa': P}


def _ice_iii_liquidus(n_points=50):
    """IAPWS Ice III melting pressure equation.

    Reference: IAPWS R14-08(2011).
    Valid from 251.165 K (Ih/III/Liquid triple point) to
    256.164 K (III/V/Liquid triple point).
    """
    T_star = 251.165  # K
    p_star = 208.566  # MPa
    a, b = 0.299948, 60.0

    T = np.linspace(251.165, 256.164, n_points)
    theta = T / T_star
    pi = 1.0 - a * (1.0 - theta**b)
    P = pi * p_star
    return {'T_K': T, 'p_MPa': P}


def _nucleation_curves():
    """Homogeneous ice nucleation boundaries from Holten et al. (2014).

    Appendix A, Eqs. (A1) and (A2).
    Returns dict with 'ih' and 'iii' sub-dicts.

    Below ~200 MPa (Ice Ih nucleation), Eq. (A1):
        P_H / P_0 = 1 + 2282.7*(1 - theta^6.243) + 157.24*(1 - theta^79.81)
        theta = T / T_0,  T_0 = 235.15 K,  P_0 = 0.1 MPa

    Above ~200 MPa (Ice III nucleation), Eq. (A2):
        T_H / K = 172.82 + 0.03718*p1 + 3.403e-5*p1^2 - 1.573e-8*p1^3
        p1 = P / MPa
    """
    T_0 = 235.15  # K
    P_0 = 0.1     # MPa

    # --- Ice Ih nucleation: Eq. (A1), parametric in T ---
    T_ih = np.linspace(181.0, 235.15, 150)
    theta = T_ih / T_0
    P_ih = P_0 * (1.0 + 2282.7 * (1.0 - theta**6.243)
                  + 157.24 * (1.0 - theta**79.81))
    # Keep only the part below the break point (~200 MPa)
    mask_ih = P_ih <= 200.0
    T_ih = T_ih[mask_ih]
    P_ih = P_ih[mask_ih]

    # --- Ice III nucleation: Eq. (A2), parametric in P ---
    P_iii = np.linspace(200.0, 300.0, 60)
    T_iii = (172.82 + 0.03718 * P_iii + 3.403e-5 * P_iii**2
             - 1.573e-8 * P_iii**3)

    return {
        'ih':  {'T_K': T_ih,  'p_MPa': P_ih},
        'iii': {'T_K': T_iii, 'p_MPa': P_iii},
    }


def _kauzmann_hdl_curve():
    """HDL Kauzmann temperature T_K(P) — empirical estimate.

    Approximate locus where entropy of the high-density liquid (HDL)
    extrapolates to crystal entropy. Based on Caupin & Anisimov (2019).
    """
    P = np.linspace(0.0, 300.0, 80)
    T = 183.0 - 0.085 * P - 5.0e-5 * P**2
    return {'T_K': T, 'p_MPa': P}


def _kauzmann_ldl_curve():
    """LDL Kauzmann temperature T_K(P) — empirical estimate.

    Approximate locus where entropy of the low-density liquid (LDL)
    extrapolates to crystal entropy. Based on Caupin & Anisimov (2019).
    """
    P = np.linspace(0.0, 300.0, 80)
    T = 155.0 - 0.065 * P - 3.0e-5 * P**2
    return {'T_K': T, 'p_MPa': P}


def _truncate_at_spinodal(kauzmann, spinodal):
    """Truncate a Kauzmann curve at the pressure where it crosses the spinodal.

    Only keep Kauzmann points where T_kauzmann < T_spinodal at matching P,
    or where P is below the spinodal range (P < P_LLCP).
    """
    if kauzmann is None or spinodal is None:
        return kauzmann

    T_k = np.asarray(kauzmann['T_K'])
    P_k = np.asarray(kauzmann['p_MPa'])
    T_s = np.asarray(spinodal['T_K'])
    P_s = np.asarray(spinodal['p_MPa'])

    if T_k.size == 0 or T_s.size == 0:
        return kauzmann

    P_s_min = P_s.min()

    # Interpolate spinodal T at the Kauzmann pressures
    # Sort spinodal by pressure for interpolation
    order = np.argsort(P_s)
    P_s_sorted = P_s[order]
    T_s_sorted = T_s[order]

    keep = np.ones(len(P_k), dtype=bool)
    for i, (pk, tk) in enumerate(zip(P_k, T_k)):
        if pk < P_s_min:
            # Below spinodal range — Kauzmann is fine
            continue
        # Interpolate spinodal T at this pressure
        t_spin = np.interp(pk, P_s_sorted, T_s_sorted)
        if tk > t_spin:
            # Kauzmann crosses above spinodal — truncate here and beyond
            keep[i:] = False
            break

    T_k = T_k[keep]
    P_k = P_k[keep]
    if T_k.size == 0:
        return None
    return {'T_K': T_k, 'p_MPa': P_k}


def _get_compute_at_x(model_key):
    """Return the model-specific compute_properties_at_x function, or None."""
    try:
        if model_key == 'duska2020':
            from duska_eos.core import compute_properties_at_x
        elif model_key == 'holten2014':
            from holten_eos.core import compute_properties_at_x
        elif model_key == 'caupin2019':
            from caupin_eos.core import compute_properties_at_x
        else:
            return None
        return compute_properties_at_x
    except ImportError:
        return None


def compute_property_at_forced_x(model_key, prop_key, T_arr, P_arr, x_arr):
    """
    Compute a property at forced composition x for each (T, P, x) triple.

    Uses model-specific compute_properties_at_x (includes excess volume).
    Returns 1-D array of property values, or None if not supported.
    """
    func = _get_compute_at_x(model_key)
    if func is None:
        return None

    n = len(T_arr)
    result = np.full(n, np.nan)
    for i in range(n):
        try:
            props = func(float(T_arr[i]), float(P_arr[i]), float(x_arr[i]))
            val = props.get(prop_key)
            if val is not None:
                result[i] = float(val)
        except Exception:
            pass
    return result


def compute_point_properties(model_keys, T_K, P_MPa):
    """
    Compute all properties at a single (T, P) for multiple models.

    Returns
    -------
    dict mapping model_key -> {prop_key: float_value}
    """
    from watereos import getProp

    results = {}
    for mk in model_keys:
        PT = np.array([np.array([P_MPa]), np.array([T_K])], dtype=object)
        out = getProp(PT, mk)
        props = {}
        for attr in MODEL_REGISTRY[mk].properties:
            val = getattr(out, attr, None)
            if val is not None:
                val = np.asarray(val).flatten()
                props[attr] = float(val[0]) if val.size > 0 else None
            else:
                props[attr] = None
        results[mk] = props
    return results
