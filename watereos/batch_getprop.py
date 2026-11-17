"""
SeaFreeze-compatible ``getProp`` for any model with a ``compute_batch``.

``watereos.two_state_eos.getProp`` assumes the full two-state key set.  The
models added from the literature return only what their source supports
(some only volumetric properties), so this version reshapes whatever arrays
the batch function returns.
"""

import numpy as np

from watereos._common import ThermodynamicStates, _is_grid_input


def getProp(PT, compute_batch, nan_keys=(), **kwargs):
    """Evaluate ``compute_batch(T_flat, P_flat, **kwargs)`` on SeaFreeze input.

    Parameters
    ----------
    PT : numpy array
        Grid mode ``np.array([P_MPa, T_K], dtype=object)`` or scatter mode
        (object array of ``(P_MPa, T_K)`` pairs).
    compute_batch : callable
        Returns a dict of 1-D arrays, one value per (T, P) point.
    nan_keys : iterable of str
        Keys to add as all-NaN arrays when the batch does not return them
        (``Kp`` for models that, like the other two-state models, do not
        compute it in batch mode).
    **kwargs
        Passed to ``compute_batch`` (for example ``branch``).

    Returns
    -------
    ThermodynamicStates
        One attribute per returned key, shaped ``(len(P), len(T))`` in grid
        mode and ``(N,)`` in scatter mode, plus ``PTM``.
    """
    out = ThermodynamicStates()
    if _is_grid_input(PT):
        P_arr = np.atleast_1d(np.asarray(PT[0], dtype=float))
        T_arr = np.atleast_1d(np.asarray(PT[1], dtype=float))
        shape = (len(P_arr), len(T_arr))
        T_grid, P_grid = np.meshgrid(T_arr, P_arr)
        T_flat = T_grid.ravel()
        P_flat = P_grid.ravel()
    else:
        pairs = np.array(PT.tolist(), dtype=float)
        P_flat = np.ascontiguousarray(pairs[:, 0])
        T_flat = np.ascontiguousarray(pairs[:, 1])
        shape = (len(PT),)

    batch = compute_batch(T_flat, P_flat, **kwargs)
    for key, val in batch.items():
        arr = np.asarray(val)
        setattr(out, key, arr.reshape(shape))
    for key in nan_keys:
        if key not in batch:
            setattr(out, key, np.full(shape, np.nan))
    out.PTM = PT
    return out
