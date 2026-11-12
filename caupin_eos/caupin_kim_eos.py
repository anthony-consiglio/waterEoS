"""
SeaFreeze-compatible interface for the Caupin & Anisimov (2019) EoS,
WITH the Kim et al. (2017) compressibility data (Table II fit).

The default ``caupin_eos.getProp`` uses Table III (without Kim data),
which the paper treats as its preferred result. This variant uses the
Table II parameter set and is exposed in the unified API as the
``'caupin2019_kim'`` model.

Backend dispatch mirrors caupin_eos.caupin_eos but binds the Table II
parameters. The JAX path is intentionally skipped: core_ad is hardwired
to Table III, so a missing Rust build falls back to the numpy core with
the Table II parameter set rather than to JAX.
"""

from watereos.two_state_eos import getProp as _getProp

from . import params_kim as _pk
from .core import compute_batch as _cb


def _numpy_batch(T_K, p_MPa, branch='stable'):
    return _cb(T_K, p_MPa, pset=_pk, branch=branch)


try:
    from watereos_rs import compute_batch_caupin_kim as compute_batch
except ImportError:
    def compute_batch(T_K, p_MPa, branch='stable'):
        return _numpy_batch(T_K, p_MPa, branch=branch)


def getProp(PT, phase=None, branch='stable'):
    """Compute thermodynamic properties using the Caupin (2019) with-Kim fit.

    ``branch`` selects the liquid state: ``'stable'`` (default, lowest
    Gibbs energy), ``'hdl'`` or ``'ldl'``; see
    ``watereos.two_state_branch``.
    """
    return _getProp(PT, compute_batch, phase, branch=branch,
                    numpy_batch=_numpy_batch)
