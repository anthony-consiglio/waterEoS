"""Tony Consiglio's Bridge29 equation of state and transport model."""
from watereos.batch_getprop import getProp as _get_prop

from .core import compute_batch


def getProp(PT, phase=None, branch="stable"):
    return _get_prop(PT, compute_batch, branch=branch, nan_keys=("Kp", "Kp_A", "Kp_B"))
