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

from watereos.two_state_eos import getProp as _getProp

try:
    from watereos_rs import compute_batch_holten as compute_batch
except ImportError:
    from .core import compute_batch


def getProp(PT, phase=None):
    """Compute thermodynamic properties using the Holten EoS."""
    return _getProp(PT, compute_batch, phase)
