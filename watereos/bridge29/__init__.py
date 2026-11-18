"""Tony Consiglio's Bridge29 equation of state and transport model."""
import numpy as np

from watereos.batch_getprop import getProp as _get_prop

from .core import compute_batch as _numpy_batch

# Backend dispatch: the Rust extension when it has Bridge29, else the NumPy
# core. An extension built before Bridge29 was added lacks the function.
try:
    from watereos_rs import compute_batch_bridge29 as _backend_compute_batch
except ImportError:
    _backend_compute_batch = _numpy_batch


def compute_batch(T, P, branch="stable", *, x=None):
    """Bridge29 properties at temperatures T [K] and pressures P [MPa].

    Same arguments and keys as ``watereos.bridge29.core.compute_batch``;
    T and P broadcast against each other. A forced composition ``x`` is
    evaluated by the NumPy core.
    """
    if x is not None or _backend_compute_batch is _numpy_batch:
        return _numpy_batch(T, P, branch, x=x)
    T, P = np.broadcast_arrays(np.asarray(T, float), np.asarray(P, float))
    out = _backend_compute_batch(np.ravel(T), np.ravel(P), branch=branch)
    return {key: np.asarray(value).reshape(T.shape) for key, value in out.items()}


def getProp(PT, phase=None, branch="stable"):
    return _get_prop(PT, compute_batch, branch=branch, nan_keys=("Kp", "Kp_A", "Kp_B"))
