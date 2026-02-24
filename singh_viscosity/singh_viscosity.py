"""
SeaFreeze-compatible interface for the Singh et al. (2017) transport model.

Usage (identical to SeaFreeze and other EoS modules):

    import numpy as np
    from singh_viscosity import getProp

    # Grid mode
    PT = np.array([P_MPa_array, T_K_array], dtype=object)
    out = getProp(PT)
    out.eta    # viscosity, shape (len(P), len(T))
    out.rho    # density (from Holten backbone), same shape

    # Scatter mode
    PT = np.empty((N,), dtype=object)
    PT[0] = (P0_MPa, T0_K)
    out = getProp(PT)
    out.eta    # shape (N,)
"""

import numpy as np
from .core import compute_batch


class TransportStates:
    """Container for transport + thermodynamic output, accessed via attributes."""
    pass


# Thermodynamic keys from the Holten backbone
_THERMO_MIX_KEYS = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel', 'x']
_THERMO_STATE_KEYS = ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel']

# Transport property keys (from Singh model)
_TRANSPORT_KEYS = ['eta', 'D', 'tau_r', 'f']

# All keys returned by compute_batch (excluding Kp variants)
_BATCH_KEYS = (
    _TRANSPORT_KEYS
    + _THERMO_MIX_KEYS
    + [k + '_A' for k in _THERMO_STATE_KEYS]
    + [k + '_B' for k in _THERMO_STATE_KEYS]
)
# Deduplicate (x appears in both thermo and transport as f)
_BATCH_KEYS = list(dict.fromkeys(_BATCH_KEYS))


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
    Compute transport and thermodynamic properties.

    Parameters
    ----------
    PT : numpy array
        Grid mode: np.array([P_MPa, T_K], dtype=object)
        Scatter mode: object array of (P_MPa, T_K) tuples
    phase : str, optional
        Ignored. Accepted for SeaFreeze API compatibility.

    Returns
    -------
    TransportStates
        Object with attributes:
          Transport: eta (Pa*s), D (m^2/s), tau_r (s), f (LDS fraction)
          Thermodynamic (from Holten backbone): rho, V, S, Cp, Cv, Kt, Ks,
            alpha, vel, x, G, H, U, A, and _A/_B suffixed state properties.
    """
    out = TransportStates()

    if _is_grid_input(PT):
        P_arr = np.asarray(PT[0], dtype=float)
        T_arr = np.asarray(PT[1], dtype=float)
        nP, nT = len(P_arr), len(T_arr)

        # meshgrid: T varies along axis 1, P along axis 0
        T_grid, P_grid = np.meshgrid(T_arr, P_arr)
        T_flat = T_grid.ravel()
        P_flat = P_grid.ravel()

        batch = compute_batch(T_flat, P_flat)

        for k in _BATCH_KEYS:
            if k in batch:
                setattr(out, k, batch[k].reshape(nP, nT))

        # Kp not computed in batch mode — fill with NaN
        for kp_key in ('Kp', 'Kp_A', 'Kp_B'):
            setattr(out, kp_key, np.full((nP, nT), np.nan))

    else:
        N = len(PT)
        T_flat = np.empty(N)
        P_flat = np.empty(N)
        for i in range(N):
            point = PT[i]
            P_flat[i] = float(point[0])
            T_flat[i] = float(point[1])

        batch = compute_batch(T_flat, P_flat)

        for k in _BATCH_KEYS:
            if k in batch:
                setattr(out, k, batch[k])

        # Kp not computed in batch mode — fill with NaN
        for kp_key in ('Kp', 'Kp_A', 'Kp_B'):
            setattr(out, kp_key, np.full(N, np.nan))

    out.PTM = PT
    return out
