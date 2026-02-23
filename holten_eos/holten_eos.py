"""
SeaFreeze-compatible interface for the Holten et al. (2014) EoS.

Usage (identical to SeaFreeze, duska_eos, and caupin_eos):

    import numpy as np
    from holten_eos import getProp

    # Grid mode
    PT = np.array([P_MPa_array, T_K_array], dtype=object)
    out = getProp(PT)
    out.rho   # shape (len(P), len(T))

    # Scatter mode
    PT = np.empty((N,), dtype=object)
    PT[0] = (P0_MPa, T0_K)
    out = getProp(PT)
    out.rho   # shape (N,)
"""

import numpy as np
from .core import compute_properties, compute_batch


class ThermodynamicStates:
    """Container for thermodynamic output, accessed via attributes."""
    pass


_MIX_KEYS = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv', 'Kt', 'Ks', 'Kp', 'alpha', 'vel', 'x']
_STATE_KEYS = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv', 'Kt', 'Ks', 'Kp', 'alpha', 'vel']
_ALL_KEYS = (
    _MIX_KEYS
    + [k + '_A' for k in _STATE_KEYS]
    + [k + '_B' for k in _STATE_KEYS]
)

# Keys returned by compute_batch (all except Kp variants)
_BATCH_KEYS = [k for k in _ALL_KEYS if k not in ('Kp', 'Kp_A', 'Kp_B')]


def _is_grid_input(PT):
    """Detect whether PT is grid format or scatter format."""
    if PT.dtype == object and len(PT) == 2:
        try:
            if hasattr(PT[0], '__len__') and not isinstance(PT[0], tuple):
                return True
        except (TypeError, IndexError):
            pass
    return False


def getProp(PT, phase=None):
    """
    Compute thermodynamic properties using the Holten EoS.

    Parameters
    ----------
    PT : numpy array
        Grid mode: np.array([P_MPa, T_K], dtype=object)
        Scatter mode: object array of (P_MPa, T_K) tuples
    phase : str, optional
        Ignored. Accepted for SeaFreeze API compatibility.

    Returns
    -------
    ThermodynamicStates
        Object with attributes: rho, V, S, Cp, Cv, Kt, Ks, alpha, vel,
        x, and _A / _B suffixed versions for each pure state.
    """
    out = ThermodynamicStates()

    if _is_grid_input(PT):
        P_arr = np.asarray(PT[0], dtype=float)
        T_arr = np.asarray(PT[1], dtype=float)
        nP, nT = len(P_arr), len(T_arr)

        # meshgrid: T varies along axis 1, P along axis 0
        T_grid, P_grid = np.meshgrid(T_arr, P_arr)
        T_flat = T_grid.ravel()
        P_flat = P_grid.ravel()

        batch = compute_batch(T_flat, P_flat)

        arrays = {}
        for k in _BATCH_KEYS:
            arrays[k] = batch[k].reshape(nP, nT)

        # Kp not computed in batch mode — fill with NaN
        for kp_key in ('Kp', 'Kp_A', 'Kp_B'):
            arrays[kp_key] = np.full((nP, nT), np.nan)

    else:
        N = len(PT)
        T_flat = np.empty(N)
        P_flat = np.empty(N)
        for i in range(N):
            point = PT[i]
            P_flat[i] = float(point[0])
            T_flat[i] = float(point[1])

        batch = compute_batch(T_flat, P_flat)

        arrays = {}
        for k in _BATCH_KEYS:
            arrays[k] = batch[k]

        # Kp not computed in batch mode — fill with NaN
        for kp_key in ('Kp', 'Kp_A', 'Kp_B'):
            arrays[kp_key] = np.full(N, np.nan)

    for k, v in arrays.items():
        setattr(out, k, v)

    out.PTM = PT
    return out
