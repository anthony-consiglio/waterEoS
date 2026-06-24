"""
Shared phase diagram functions for two-state EoS models.

All three models (Caupin, Holten, Duška) have identical logic for
spinodal, binodal, phase diagram, TMD, and Kauzmann computations.
The only model-specific parts are _get_adapter() and find_LLCP(),
which remain in each model's phase_diagram.py.
"""

import numpy as np


def get_compute_batch(model_package):
    """Import compute_batch from a model package.

    Parameters
    ----------
    model_package : str
        Package name, e.g. 'caupin_eos', 'holten_eos', 'duska_eos'.
    """
    return __import__(f'{model_package}.core',
                      fromlist=['compute_batch']).compute_batch


def compute_spinodal_curve(adapter, p_range_MPa=None, n_pressures=150):
    """Compute the liquid-liquid spinodal curve.

    Parameters
    ----------
    adapter : dict
        Vectorized adapter from the model's _get_adapter().
    p_range_MPa : array_like, optional
        Pressure values in MPa. Auto-generated if None.
    n_pressures : int
        Number of pressure points (used when p_range_MPa is None).

    Returns
    -------
    dict
        Arrays for the closed T-p curve and separate branches.
    """
    from watereos.fast_phase_diagram import compute_spinodal_fast

    if p_range_MPa is None:
        p_range_MPa = _default_pressure_grid(adapter['p_LLCP'], n_pressures)

    return compute_spinodal_fast(p_range_MPa, adapter)


def compute_binodal_curve(adapter, p_range_MPa=None, n_pressures=150,
                          spinodal=None):
    """Compute the liquid-liquid binodal (coexistence) curve.

    Parameters
    ----------
    adapter : dict
        Vectorized adapter from the model's _get_adapter().
    p_range_MPa : array_like, optional
        Pressure values in MPa. Auto-generated if None.
    n_pressures : int
        Number of pressure points (used when p_range_MPa is None).
    spinodal : dict, optional
        Pre-computed spinodal. Computed if None.

    Returns
    -------
    dict
        Binodal arrays with T_K, p_MPa, x keys.
    """
    from watereos.fast_phase_diagram import compute_binodal_fast

    if p_range_MPa is None:
        p_range_MPa = _default_pressure_grid(adapter['p_LLCP'], n_pressures)

    if spinodal is None:
        spinodal = compute_spinodal_curve(adapter, p_range_MPa, n_pressures)

    return compute_binodal_fast(p_range_MPa, spinodal, adapter)


def compute_phase_diagram(adapter, find_llcp_fn, p_range_MPa=None,
                          n_pressures=150):
    """Compute the full liquid-liquid phase diagram.

    Parameters
    ----------
    adapter : dict
        Vectorized adapter from the model's _get_adapter().
    find_llcp_fn : callable
        Model's find_LLCP function.
    p_range_MPa : array_like, optional
        Pressure values in MPa. Auto-generated if None.
    n_pressures : int
        Number of pressure points (used when p_range_MPa is None).

    Returns
    -------
    dict
        Keys: 'LLCP', 'spinodal', 'binodal'.
    """
    llcp = find_llcp_fn()

    if p_range_MPa is None:
        p_range_MPa = _default_pressure_grid(llcp['p_MPa'], n_pressures)

    spinodal = compute_spinodal_curve(adapter, p_range_MPa, n_pressures)
    binodal = compute_binodal_curve(adapter, p_range_MPa, n_pressures,
                                    spinodal=spinodal)

    return {
        'LLCP': {'T_K': llcp['T_K'], 'p_MPa': llcp['p_MPa'], 'x': 0.5},
        'spinodal': {
            'T_K': spinodal['T_K'], 'p_MPa': spinodal['p_MPa'],
        },
        'binodal': {
            'T_K': binodal['T_K'], 'p_MPa': binodal['p_MPa'],
            'x': binodal['x'],
        },
    }


def compute_tmd_temperature(P_MPa, compute_batch, **kwargs):
    """Compute TMD temperature (alpha=0) at given pressure(s).

    Parameters
    ----------
    P_MPa : float or array_like
        Pressure(s) in MPa.
    compute_batch : callable
        Model's compute_batch function.

    Returns
    -------
    float or numpy.ndarray
        TMD temperature(s) in K. NaN where no crossing found.
    """
    from watereos.tmd import compute_tmd_temperature as _tmd
    return _tmd(P_MPa, compute_batch, **kwargs)


def compute_kauzmann_temperature(P_MPa, compute_batch, **kwargs):
    """Compute Kauzmann temperature (S_liquid = S_ice_Ih) at given pressure(s).

    Parameters
    ----------
    P_MPa : float or array_like
        Pressure(s) in MPa.
    compute_batch : callable
        Model's compute_batch function.

    Returns
    -------
    float or numpy.ndarray
        Kauzmann temperature(s) in K. NaN where no crossing found.
    """
    from watereos.kauzmann import compute_kauzmann_temperature as _kauz
    return _kauz(P_MPa, compute_batch, **kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────

def _default_pressure_grid(p_llcp, n_pressures):
    """Build the default pressure grid below the LLCP."""
    n_near = min(n_pressures // 3, 100)
    n_far = n_pressures - n_near
    p_near = np.linspace(p_llcp + 0.05, p_llcp + 5.0, n_near, endpoint=False)
    p_far = np.linspace(p_llcp + 5.0, 200.0, n_far)
    return np.concatenate([p_near, p_far])
