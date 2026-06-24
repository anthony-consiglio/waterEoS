"""
SeaFreeze-compatible interface for the Grenke & Elliott (2025) EoS.

Usage (identical to SeaFreeze and other *_eos modules):

    import numpy as np
    from grenke_eos import getProp

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
from watereos._common import ThermodynamicStates, _is_grid_input

_KEYS = [
    'rho', 'V', 'S', 'G', 'H', 'U', 'A',
    'Cp', 'Cv', 'Kt', 'Ks', 'Kp', 'alpha', 'vel',
]

# Keys returned by compute_batch (all except Kp)
_BATCH_KEYS = [k for k in _KEYS if k != 'Kp']


def getProp(PT, phase=None):
    """
    Compute thermodynamic properties using the Grenke EoS.

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
        Object with attributes: rho, V, S, G, H, U, A, Cp, Cv,
        Kt, Ks, Kp, alpha, vel.
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

        # Kp not computed in batch — fill with NaN
        arrays['Kp'] = np.full((nP, nT), np.nan)

    else:
        N = len(PT)
        pairs = np.array(PT.tolist(), dtype=float)
        P_flat = pairs[:, 0]
        T_flat = pairs[:, 1]

        batch = compute_batch(T_flat, P_flat)

        arrays = {}
        for k in _BATCH_KEYS:
            arrays[k] = batch[k]

        # Kp not computed in batch — fill with NaN
        arrays['Kp'] = np.full(N, np.nan)

    for k, v in arrays.items():
        setattr(out, k, v)

    out.PTM = PT
    return out
