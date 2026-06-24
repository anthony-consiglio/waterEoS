"""
SeaFreeze-compatible interface for the Caupin & Anisimov (2019) EoS.

Usage (identical to SeaFreeze and duska_eos):

    import numpy as np
    from caupin_eos import getProp

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

# Backend dispatch (priority order):
#   1. Rust (watereos_rs) — fastest; built-in to the published wheel
#   2. JAX (core_ad)      — 2-5x faster than numpy on supported platforms;
#                           only imported if Rust is unavailable, so Rust users
#                           never pay JAX's import or trace-compile cost.
#   3. numpy (core)       — hand-coded, last-resort pure-Python path.
try:
    from watereos_rs import compute_batch_caupin as compute_batch
except ImportError:
    try:
        from .core_ad import compute_batch
    except ImportError:
        from .core import compute_batch


def getProp(PT, phase=None):
    """Compute thermodynamic properties using the Caupin EoS."""
    return _getProp(PT, compute_batch, phase)
