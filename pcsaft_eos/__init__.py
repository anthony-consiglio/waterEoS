"""Novak et al. PC-SAFT models for pure water."""

from watereos.batch_getprop import getProp as _getProp

from .core import compute_batch, pressure
from .params import FITS


def getProp(PT, model='novak2025_pcsaft_ts_caf'):
    """Evaluate a dense-fluid model on a SeaFreeze PT grid or point list."""
    return _getProp(PT, compute_batch, model=model)


__all__ = ['FITS', 'compute_batch', 'getProp', 'pressure']
