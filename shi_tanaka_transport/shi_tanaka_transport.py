"""
SeaFreeze-compatible interface for Shi & Tanaka (2020) transport + thermo.

Usage (identical to singh_viscosity / the other model wrappers):

    import numpy as np
    from shi_tanaka_transport import getProp

    # Grid mode
    PT = np.array([P_MPa_array, T_K_array], dtype=object)
    out = getProp(PT)
    out.eta    # viscosity, shape (len(P), len(T))
    out.rho    # density (from Shi-Tanaka backbone), same shape

    # Scatter mode
    PT = np.empty((N,), dtype=object)
    PT[0] = (P0_MPa, T0_K)
    out = getProp(PT)
    out.eta    # shape (N,)
"""

import numpy as np

from watereos._common import _is_grid_input

from .core import compute_batch


class TransportStates:
    """Container exposing transport + thermodynamic output as attributes."""
    pass


_THERMO_MIX = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
               'Kt', 'Ks', 'alpha', 'vel', 'x']
_THERMO_STATE = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
                 'Kt', 'Ks', 'alpha', 'vel']
_TRANSPORT = ['eta', 'D', 'tau_r', 'f']

_BATCH_KEYS = list(dict.fromkeys(
    _TRANSPORT + _THERMO_MIX
    + [k + '_A' for k in _THERMO_STATE]
    + [k + '_B' for k in _THERMO_STATE]
))


def getProp(PT, phase=None):
    """Compute Shi-Tanaka transport + thermodynamic properties on PT input."""
    out = TransportStates()

    if _is_grid_input(PT):
        P_arr = np.asarray(PT[0], dtype=float)
        T_arr = np.asarray(PT[1], dtype=float)
        nP, nT = len(P_arr), len(T_arr)

        T_grid, P_grid = np.meshgrid(T_arr, P_arr)
        batch = compute_batch(T_grid.ravel(), P_grid.ravel())

        for k in _BATCH_KEYS:
            if k in batch:
                setattr(out, k, np.asarray(batch[k]).reshape(nP, nT))

        for kp_key in ('Kp', 'Kp_A', 'Kp_B'):
            setattr(out, kp_key, np.full((nP, nT), np.nan))
    else:
        N = len(PT)
        pairs = np.array(PT.tolist(), dtype=float)
        P_flat = pairs[:, 0]
        T_flat = pairs[:, 1]

        batch = compute_batch(T_flat, P_flat)

        for k in _BATCH_KEYS:
            if k in batch:
                setattr(out, k, np.asarray(batch[k]))

        for kp_key in ('Kp', 'Kp_A', 'Kp_B'):
            setattr(out, kp_key, np.full(N, np.nan))

    out.PTM = PT
    return out
