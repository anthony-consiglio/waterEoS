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

try:
    from watereos_rs import compute_batch_caupin_kim as compute_batch
except ImportError:
    from . import params_kim as _pk
    from .core import compute_batch as _cb

    def compute_batch(T_K, p_MPa):
        return _cb(T_K, p_MPa, pset=_pk)


def getProp(PT, phase=None):
    """Compute thermodynamic properties using the Caupin (2019) with-Kim fit."""
    return _getProp(PT, compute_batch, phase)
