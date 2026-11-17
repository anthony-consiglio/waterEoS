"""
SeaFreeze-compatible interface for the Moynihan (1996) model.

    import numpy as np
    from moynihan_eos import getProp

    PT = np.array([np.array([0.1, 100.0]), np.array([200.0, 250.0])],
                  dtype=object)
    out = getProp(PT)
    out.x, out.rho, out.Cp_ex      # shape (len(P), len(T))
"""

from watereos.batch_getprop import getProp as _getProp

from .core import compute_batch


def getProp(PT, phase=None, branch='stable'):
    """Compute the properties the Moynihan (1996) model determines.

    Returns x, rho, V, alpha, Kt and Cp_ex (see ``moynihan_eos.core``), and
    ``metastable`` when ``branch`` is 'hdl' or 'ldl'.  ``phase`` is ignored
    (SeaFreeze compatibility).
    """
    return _getProp(PT, compute_batch, branch=branch)
