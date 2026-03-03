"""
Computation helpers — wraps watereos.getProp and phase diagram functions.
"""

import json
import importlib.resources
import numpy as np
from watereos.model_registry import MODEL_REGISTRY


# --- Phase diagram cache (module-level, populated on first access) ---
_phase_diagram_cache = {}


def _load_precomputed(model_key):
    """Load precomputed phase diagram from watereos/data/ JSON file."""
    try:
        data_dir = importlib.resources.files('watereos') / 'data'
        path = data_dir / f'{model_key}_phase_diagram.json'
        with importlib.resources.as_file(path) as f:
            raw = json.loads(f.read_text())
        return _deserialize_precomputed(raw)
    except Exception:
        return None


def _deserialize_precomputed(raw):
    """Convert JSON dict (lists) back to numpy arrays recursively."""
    if not isinstance(raw, dict):
        return raw
    result = {}
    for k, v in raw.items():
        if v is None:
            result[k] = None
        elif isinstance(v, dict):
            result[k] = _deserialize_precomputed(v)
        elif isinstance(v, list):
            result[k] = np.array(v)
        else:
            result[k] = v
    return result


# --- Phase diagram default parameters ---
_PHASE_DIAGRAM_DEFAULTS = {
    'P_max': 300.0,
    'P_llcp_offset': 0.05,
    'tmd_n_T_scan': 60,
    'tmd_T_lo': 125.0,
    'tmd_bisection_iters': 15,
    'widom_slope': 0.3,
    'widom_window': 30.0,
    'widom_n_T_scan': 120,
}


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

    from watereos.model_registry import get_display_label
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


def compute_phase_diagram_data(model_key, n_pressures=150,
                               P_max=None, P_llcp_offset=None,
                               tmd_n_T_scan=None, tmd_T_lo=None,
                               tmd_bisection_iters=None,
                               widom_slope=None, widom_window=None,
                               widom_n_T_scan=None):
    """
    Compute full phase diagram for a two-state model.

    Checks for precomputed JSON first, then module-level cache, then
    falls back to live computation.

    Returns
    -------
    dict with keys: 'LLCP', 'spinodal', 'binodal' (model-dependent),
    'tmd', 'widom' (model-dependent), 'ice_ih_liquidus', 'ice_iii_liquidus',
    'nucleation_ih', 'nucleation_iii', 'kauzmann_hdl', 'kauzmann_ldl',
    'triple_point' (model-independent).
    """
    # Check caches first (precomputed JSON or previous live computation)
    if model_key in _phase_diagram_cache:
        return _phase_diagram_cache[model_key]

    precomputed = _load_precomputed(model_key)
    if precomputed is not None:
        _phase_diagram_cache[model_key] = precomputed
        return precomputed

    # Apply defaults for any unspecified parameters
    d = _PHASE_DIAGRAM_DEFAULTS
    if P_max is None:
        P_max = d['P_max']
    if P_llcp_offset is None:
        P_llcp_offset = d['P_llcp_offset']
    if tmd_n_T_scan is None:
        tmd_n_T_scan = d['tmd_n_T_scan']
    if tmd_T_lo is None:
        tmd_T_lo = d['tmd_T_lo']
    if tmd_bisection_iters is None:
        tmd_bisection_iters = d['tmd_bisection_iters']
    if widom_slope is None:
        widom_slope = d['widom_slope']
    if widom_window is None:
        widom_window = d['widom_window']
    if widom_n_T_scan is None:
        widom_n_T_scan = d['widom_n_T_scan']

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

    # Dense spacing near LLCP for smooth curve onset, sparser far away
    n_near = n_pressures // 3
    n_far = n_pressures - n_near
    p_near = np.linspace(P_llcp + P_llcp_offset, P_llcp + 10.0, n_near, endpoint=False)
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

    tmd = _compute_tmd_curve(model_key, llcp=llcp,
                             n_T_scan=tmd_n_T_scan, T_lo=tmd_T_lo,
                             bisection_iters=tmd_bisection_iters)

    widom = _compute_widom_line(model_key, llcp,
                                slope=widom_slope, window=widom_window,
                                n_T_scan=widom_n_T_scan)

    nuc = _nucleation_curves()

    # Model-specific Kauzmann: where total S_liquid = S_ice_Ih
    kauzmann = _compute_kauzmann_curve(model_key, P_range=(0.0, P_max),
                                       S_key='S')

    result = {
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
        'kauzmann': kauzmann,
        'triple_point': {'T_K': 251.165, 'p_MPa': 209.9},
    }
    _phase_diagram_cache[model_key] = result
    return result


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


def _compute_tmd_curve(model_key, llcp=None, n_points=80,
                       n_T_scan=60, T_lo=125.0, bisection_iters=15):
    """Find the Temperature of Maximum Density (alpha=0) at each pressure.

    Uses vectorized grid evaluation + parallel bisection for speed.
    Picks the highest-T crossing (~277 K at 0.1 MPa) which is the physical TMD.
    """
    compute_batch = _get_compute_batch(model_key)
    info = MODEL_REGISTRY[model_key]
    P_lo = -140.0
    P_hi = min(info.P_max, 300.0)

    if P_lo >= P_hi:
        return None

    P_pts = np.linspace(P_lo, P_hi, n_points)
    T_hi = info.T_max
    T_scan = np.linspace(T_lo, T_hi, n_T_scan)

    # --- Single vectorized call: alpha at all (P, T) grid points ---
    T_grid, P_grid = np.meshgrid(T_scan, P_pts)
    try:
        batch = compute_batch(T_grid.ravel(), P_grid.ravel())
        alpha_grid = batch['alpha'].reshape(n_points, n_T_scan)
    except Exception:
        return None

    # --- Find last (highest-T) sign change along T axis for each pressure ---
    valid = ~np.isnan(alpha_grid[:, :-1]) & ~np.isnan(alpha_grid[:, 1:])
    sign_change = (alpha_grid[:, :-1] * alpha_grid[:, 1:] < 0) & valid

    has_change = sign_change.any(axis=1)
    if not has_change.any():
        return None

    # Flip along T axis, argmax finds first-from-right, convert back
    n_cols = sign_change.shape[1]  # n_T_scan - 1
    flipped = sign_change[:, ::-1]
    last_from_right = flipped.argmax(axis=1)
    last_idx = (n_cols - 1) - last_from_right

    mask = has_change
    row_idx = np.where(mask)[0]
    j_bracket = last_idx[mask]

    P_bracket = P_pts[mask]
    T_lo_bracket = T_scan[j_bracket]
    T_hi_bracket = T_scan[j_bracket + 1]

    # Sign of alpha at the lower bracket endpoint (for bisection direction)
    alpha_at_lo = alpha_grid[row_idx, j_bracket]
    sign_lo = np.sign(alpha_at_lo)

    # --- Vectorized bisection: refine all brackets simultaneously ---
    for _ in range(bisection_iters):
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


def _compute_widom_line(model_key, llcp, n_points=60,
                        slope=0.3, window=30.0, n_T_scan=120):
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
    # Widom line extends from LLCP toward lower pressures and higher T.
    # Allow negative pressures for models whose LLCP is near 0 MPa.
    P_lo = max(P_llcp - 100.0, -140.0)
    P_hi = P_llcp - 0.5  # just below LLCP

    if P_lo >= P_hi:
        return None

    P_pts = np.linspace(P_hi, P_lo, n_points)

    # Compute the per-pressure T window bounds to determine global range
    dP = P_llcp - P_pts
    T_centers = T_llcp + dP * slope
    T_global_lo = max(float(np.min(T_centers)) - window, info.T_min)
    T_global_hi = min(float(np.max(T_centers)) + window, info.T_max)

    T_scan = np.linspace(T_global_lo, T_global_hi, n_T_scan)

    # --- Single vectorized call: Cp at all (P, T) grid points ---
    T_grid, P_grid = np.meshgrid(T_scan, P_pts)
    try:
        batch = compute_batch(T_grid.ravel(), P_grid.ravel())
        Cp_grid = batch['Cp'].reshape(n_points, n_T_scan)
    except Exception:
        return None

    # --- Per-pressure window mask: restrict each row to [T_center±window] ---
    T_lo_per_P = np.maximum(T_centers - window, info.T_min)[:, np.newaxis]
    T_hi_per_P = np.minimum(T_centers + window, info.T_max)[:, np.newaxis]
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


def _compute_kauzmann_curve(model_key, P_range=(0.0, 300.0), n_points=80,
                            S_key='S', T_target=185.0,
                            T_scan_lo=100.0, T_scan_hi=260.0, n_scan=200):
    """Find Kauzmann temperature (S_liquid = S_ice_Ih) for a specific model.

    Model entropies are already aligned to IAPWS-95 reference and use the
    same units (J/(kg·K)) as SeaFreeze ice Ih.

    Parameters
    ----------
    model_key : str
        Model identifier.
    P_range : tuple
        (P_min, P_max) in MPa.
    S_key : str
        Entropy key from compute_batch: 'S' for total (HDL Kauzmann),
        'S_B' for LDL component (LDL Kauzmann).
    T_target : float
        Target temperature (K) for selecting among multiple crossings.
    """
    import seafreeze.seafreeze as sf

    compute_batch = _get_compute_batch(model_key)
    P_pts = np.linspace(P_range[0], P_range[1], n_points)
    T_scan = np.linspace(T_scan_lo, T_scan_hi, n_scan)

    # SeaFreeze grid mode: PT = np.array([P_1d, T_1d], dtype=object)
    PT_grid = np.array([P_pts, T_scan], dtype=object)
    try:
        ice_out = sf.getProp(PT_grid, 'Ih')
        S_ice = np.asarray(ice_out.S, dtype=float)  # shape (n_points, n_scan)
    except Exception:
        return None

    # Model entropy on same meshgrid
    T_grid, P_grid = np.meshgrid(T_scan, P_pts)
    try:
        batch = compute_batch(T_grid.ravel(), P_grid.ravel())
        S_liq = batch[S_key].reshape(n_points, n_scan)
    except Exception:
        return None

    dS = S_liq - S_ice

    # At each pressure, find crossing nearest to T_target
    T_lo_list = []
    T_hi_list = []
    P_list = []
    ip_map = []

    for ip in range(n_points):
        row = dS[ip]
        xings = []
        for j in range(n_scan - 1):
            if (np.isfinite(row[j]) and np.isfinite(row[j + 1])
                    and row[j] * row[j + 1] < 0):
                xings.append(j)
        if xings:
            best_j = min(xings,
                         key=lambda j: abs(0.5 * (T_scan[j] + T_scan[j + 1]) - T_target))
            T_lo_list.append(T_scan[best_j])
            T_hi_list.append(T_scan[best_j + 1])
            P_list.append(P_pts[ip])
            ip_map.append(ip)

    if not T_lo_list:
        return None

    T_lo_b = np.array(T_lo_list)
    T_hi_b = np.array(T_hi_list)
    P_b = np.array(P_list)

    # Vectorized bisection
    batch_lo = compute_batch(T_lo_b, P_b)
    S_lo_liq = batch_lo[S_key]

    # SeaFreeze for bisection points — use scatter mode (small arrays)
    def _sf_S(T_arr, P_arr):
        n = len(T_arr)
        PT = np.empty(n, dtype=object)
        PT[:] = list(zip(P_arr, T_arr))
        out = sf.getProp(PT, 'Ih')
        return np.asarray(out.S).ravel()

    S_lo_ice = _sf_S(T_lo_b, P_b)
    sign_lo = np.sign(S_lo_liq - S_lo_ice)

    for _ in range(30):
        T_mid = 0.5 * (T_lo_b + T_hi_b)
        S_mid_liq = compute_batch(T_mid, P_b)[S_key]
        S_mid_ice = _sf_S(T_mid, P_b)
        same = np.sign(S_mid_liq - S_mid_ice) == sign_lo
        T_lo_b = np.where(same, T_mid, T_lo_b)
        T_hi_b = np.where(~same, T_mid, T_hi_b)

    T_final = 0.5 * (T_lo_b + T_hi_b)
    return {'T_K': T_final, 'p_MPa': P_b}


def _truncate_at_binodal(curve, binodal):
    """Truncate a curve so it doesn't extend into the two-phase region.

    Keeps only points at pressures below the binodal's minimum pressure
    (i.e. below the LLCP), which is the single-phase region.
    """
    if curve is None or binodal is None:
        return curve

    T_c = np.asarray(curve['T_K'])
    P_c = np.asarray(curve['p_MPa'])
    P_bn = np.asarray(binodal.get('p_array', binodal.get('p_MPa', [])))

    if T_c.size == 0 or P_bn.size == 0:
        return curve

    P_bn_min = P_bn.min()  # LLCP pressure (start of two-phase region)
    keep = P_c < P_bn_min

    T_c = T_c[keep]
    P_c = P_c[keep]
    if T_c.size == 0:
        return None
    return {'T_K': T_c, 'p_MPa': P_c}


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
