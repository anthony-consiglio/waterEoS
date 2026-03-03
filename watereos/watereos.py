"""
Unified water equation-of-state dispatcher.

Provides a single ``getProp(PT, model)`` interface that dispatches to:

  - ``'water1'``      -- SeaFreeze water1
  - ``'IAPWS95'``     -- SeaFreeze IAPWS-95
  - ``'holten2014'``  -- Holten, Sengers & Anisimov (2014)
  - ``'caupin2019'``  -- Caupin & Anisimov (2019)
  - ``'duska2020'``   -- Duska (2020) EOS-VaT
  - ``'grenke2025'``  -- Grenke & Elliott (2025) Tait-Tammann
  - ``'singh2017'``   -- Singh, Issenmann & Caupin (2017) transport
"""

import warnings

import numpy as np

from watereos._common import _is_grid_input
from watereos.model_registry import MODEL_REGISTRY

_MODELS = ['water1', 'IAPWS95', 'holten2014', 'caupin2019', 'duska2020', 'grenke2025', 'singh2017']

# Case-insensitive lookup  ->  canonical name
_CANONICAL = {name.lower(): name for name in _MODELS}

_PT_FORMAT_MSG = (
    "PT must be a numpy array in one of these formats:\n"
    "  Grid mode:    np.array([P_array, T_array], dtype=object)\n"
    "  Scatter mode:  np.array([(P1,T1), (P2,T2), ...], dtype=object)\n"
    "Example:\n"
    "  PT = np.array([np.array([0.1, 100]), np.array([250, 300])], dtype=object)"
)


def _validate_PT(PT):
    """Validate PT input and return it as a numpy array.

    Raises TypeError with a helpful message if PT is not in the expected
    SeaFreeze format.
    """
    if not isinstance(PT, np.ndarray):
        try:
            PT = np.asarray(PT)
        except Exception:
            raise TypeError(_PT_FORMAT_MSG)

    if PT.ndim == 0:
        raise TypeError(
            f"PT is a scalar ({PT}), but it must contain (P, T) pairs.\n"
            + _PT_FORMAT_MSG
        )

    # A 1-D numeric array (e.g. [210, 220, 240]) is always wrong — the user
    # likely passed bare temperatures or pressures without pairing them.
    if PT.ndim == 1 and PT.dtype.kind in ('i', 'f'):  # int or float
        raise TypeError(
            f"PT looks like a flat numeric array (shape {PT.shape}, "
            f"dtype {PT.dtype}).\n"
            "Each element must be a (P, T) pair, not a single number.\n"
            + _PT_FORMAT_MSG
        )

    return PT


def _check_bounds(PT, model):
    """Warn if any (T, P) values fall outside the model's suggested range."""
    info = MODEL_REGISTRY.get(model)
    if info is None:
        return
    T_min, T_max, P_min, P_max = info.T_min, info.T_max, info.P_min, info.P_max

    try:
        if _is_grid_input(PT):
            P_vals = np.asarray(PT[0], dtype=float)
            T_vals = np.asarray(PT[1], dtype=float)
        else:
            pairs = np.array(PT.tolist(), dtype=float)
            P_vals = pairs[:, 0]
            T_vals = pairs[:, 1]
    except Exception:
        return

    n_T = int(np.sum((T_vals < T_min) | (T_vals > T_max)))
    n_P = int(np.sum((P_vals < P_min) | (P_vals > P_max)))

    if n_T:
        warnings.warn(
            f"waterEoS: {n_T} temperature value(s) outside suggested range "
            f"for {model} [{T_min}-{T_max} K]",
            stacklevel=3,
        )
    if n_P:
        warnings.warn(
            f"waterEoS: {n_P} pressure value(s) outside suggested range "
            f"for {model} [{P_min}-{P_max} MPa]",
            stacklevel=3,
        )


def list_models():
    """Return list of available model name strings."""
    return list(_MODELS)


def getProp(PT, model):
    """Compute thermodynamic properties for water.

    Parameters
    ----------
    PT : numpy array
        Pressure (MPa) and temperature (K) input.  Two formats are accepted
        (matching the SeaFreeze convention):

        **Grid mode** — every combination of P and T is evaluated::

            PT = np.array([P_array, T_array], dtype=object)

        **Scatter mode** — specific (P, T) pairs::

            PT = np.array([(P1, T1), (P2, T2), ...], dtype=object)

    model : str
        One of: 'water1', 'IAPWS95', 'holten2014', 'caupin2019', 'duska2020',
        'grenke2025', 'singh2017'.  Matching is case-insensitive.

    Returns
    -------
    out : object
        Result object whose attributes (``rho``, ``Cp``, ``vel``, etc.) depend
        on the chosen backend.
    """
    PT = _validate_PT(PT)

    canonical = _CANONICAL.get(model.lower())
    if canonical is not None:
        _check_bounds(PT, canonical)
    if canonical is None:
        raise ValueError(
            f"Unknown model '{model}'. Choose from: {', '.join(_MODELS)}"
        )

    if canonical == 'water1':
        from seafreeze.seafreeze import getProp as _sf
        return _sf(PT, 'water1')

    if canonical == 'IAPWS95':
        from seafreeze.seafreeze import getProp as _sf
        return _sf(PT, 'water_IAPWS95')

    if canonical == 'holten2014':
        from holten_eos import getProp as _gp
        return _gp(PT)

    if canonical == 'caupin2019':
        from caupin_eos import getProp as _gp
        return _gp(PT)

    if canonical == 'duska2020':
        from duska_eos import getProp as _gp
        return _gp(PT)

    if canonical == 'grenke2025':
        from grenke_eos import getProp as _gp
        return _gp(PT)

    if canonical == 'singh2017':
        from singh_viscosity import getProp as _gp
        return _gp(PT)


def compute(T_K, P_MPa, model):
    """Compute thermodynamic properties for water.

    A simpler alternative to getProp() — pass T and P directly
    as scalars, lists, or arrays.  Returns a grid over all combinations.

    Parameters
    ----------
    T_K : float or array_like
        Temperature(s) in Kelvin.
    P_MPa : float or array_like
        Pressure(s) in MPa.
    model : str
        One of: 'water1', 'IAPWS95', 'holten2014', 'caupin2019', 'duska2020',
        'grenke2025', 'singh2017'.  Matching is case-insensitive.

    Returns
    -------
    out : object
        Result object whose attributes (``rho``, ``Cp``, ``vel``, etc.) depend
        on the chosen backend.
    """
    PT = np.array([np.atleast_1d(P_MPa), np.atleast_1d(T_K)], dtype=object)
    return getProp(PT, model)
