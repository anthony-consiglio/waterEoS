"""
SeaFreeze-compatible interface for the Shi & Tanaka (2020) hierarchical
two-state EoS.

Usage (identical to SeaFreeze and the other two-state wrappers):

    import numpy as np
    from shi_tanaka_eos import getProp

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

from watereos.two_state_eos import getProp as _getProp

try:
    from watereos_rs import compute_batch_shi_tanaka as compute_batch
except ImportError:
    from .core import compute_batch


def getProp(PT, phase=None):
    """Compute thermodynamic properties using the Shi-Tanaka 2020 EoS."""
    return _getProp(PT, compute_batch, phase)
