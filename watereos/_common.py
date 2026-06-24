"""Shared utilities for waterEoS model interfaces."""

import numpy as np


class ThermodynamicStates:
    """Container for thermodynamic output, accessed via attributes.

    Attributes are set dynamically by each model backend.  For two-state
    models the full set is:

    Mixture (equilibrium) properties
        rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, Kp, alpha, vel, x

    Per-state properties (suffixed ``_A`` for HDL, ``_B`` for LDL)
        rho_A, V_A, S_A, G_A, ... vel_A, rho_B, V_B, ... vel_B

    Transport properties (``singh2017`` only)
        eta, D, tau_r, f

    Each attribute is a numpy array whose shape depends on the input:

    * **Grid mode** ``PT = [P_array, T_array]`` — shape ``(len(P), len(T))``
    * **Scatter mode** ``PT = [(P1,T1), (P2,T2), ...]`` — shape ``(N,)``

    Units follow the SeaFreeze convention: kg, m, s, K, MPa, J.
    """
    pass


def _is_grid_input(PT):
    """Detect whether *PT* uses grid format or scatter format.

    Parameters
    ----------
    PT : numpy.ndarray (dtype=object)
        Either ``[P_array, T_array]`` (grid) or ``[(P,T), ...]`` (scatter).

    Returns
    -------
    bool
        ``True`` if *PT* looks like a two-element grid input.
    """
    if PT.dtype == object and len(PT) == 2:
        try:
            if hasattr(PT[0], '__len__') and not isinstance(PT[0], tuple):
                return True
        except (TypeError, IndexError):
            pass
    return False
