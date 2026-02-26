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
    Compute full phase diagram (LLCP, spinodal, binodal) for a two-state model.

    Returns
    -------
    dict with keys 'LLCP', 'spinodal', 'binodal' (each a dict).
    spinodal/binodal include both closed-curve arrays (T_K, p_MPa) and
    per-branch arrays (T_upper, T_lower, T_binodal, p_array, etc.).
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
    spinodal = compute_spinodal_curve(n_pressures=n_pressures)
    binodal = compute_binodal_curve(n_pressures=n_pressures)

    return {
        'LLCP': llcp,
        'spinodal': spinodal,
        'binodal': binodal,
    }


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
