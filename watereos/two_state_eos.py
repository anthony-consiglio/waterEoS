"""
Shared SeaFreeze-compatible getProp implementation for two-state EoS models.

All three models (Caupin, Holten, Duška) use identical getProp logic:
grid/scatter input detection, compute_batch dispatch, output assembly.
Each model's *_eos.py becomes a thin wrapper that supplies its own
compute_batch callable.
"""

import numpy as np

from watereos._common import ThermodynamicStates, _is_grid_input

_MIX_KEYS = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
             'Kt', 'Ks', 'Kp', 'alpha', 'vel', 'x']
_STATE_KEYS = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
               'Kt', 'Ks', 'Kp', 'alpha', 'vel']
_ALL_KEYS = (
    _MIX_KEYS
    + [k + '_A' for k in _STATE_KEYS]
    + [k + '_B' for k in _STATE_KEYS]
)

# Keys returned by compute_batch (all except Kp variants)
_BATCH_KEYS = [k for k in _ALL_KEYS if k not in ('Kp', 'Kp_A', 'Kp_B')]


def getProp(PT, compute_batch, phase=None):
    """Compute thermodynamic properties for a two-state EoS model.

    Parameters
    ----------
    PT : numpy array
        Grid mode: np.array([P_MPa, T_K], dtype=object)
        Scatter mode: object array of (P_MPa, T_K) tuples
    compute_batch : callable
        Model's compute_batch(T_flat, P_flat) -> dict of arrays.
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

        T_grid, P_grid = np.meshgrid(T_arr, P_arr)
        T_flat = T_grid.ravel()
        P_flat = P_grid.ravel()

        batch = compute_batch(T_flat, P_flat)

        arrays = {}
        for k in _BATCH_KEYS:
            arrays[k] = batch[k].reshape(nP, nT)

        # Kp not computed in batch mode -- fill with NaN
        for kp_key in ('Kp', 'Kp_A', 'Kp_B'):
            arrays[kp_key] = np.full((nP, nT), np.nan)

    else:
        N = len(PT)
        pairs = np.array(PT.tolist(), dtype=float)
        P_flat = np.ascontiguousarray(pairs[:, 0])
        T_flat = np.ascontiguousarray(pairs[:, 1])

        batch = compute_batch(T_flat, P_flat)

        arrays = {}
        for k in _BATCH_KEYS:
            arrays[k] = batch[k]

        # Kp not computed in batch mode -- fill with NaN
        for kp_key in ('Kp', 'Kp_A', 'Kp_B'):
            arrays[kp_key] = np.full(N, np.nan)

    for k, v in arrays.items():
        setattr(out, k, v)

    out.PTM = PT
    return out
