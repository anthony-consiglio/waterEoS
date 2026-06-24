"""
Native drop-in for ``seafreeze.getProp`` using the Rust evaluator.

The upstream SeaFreeze Python package wraps a tensor-product B-spline
evaluation in numpy that becomes a hot spot when called inside the
T-V phase-diagram and Kauzmann solvers. The same evaluation has been
ported to Rust in ``watereos_rs::seafreeze`` (see
``watereos/data/seafreeze_splines.bin`` for the bundled spline data),
and this module exposes that backend with a SeaFreeze-shaped API:

  * Accepts the same two PTm conventions: object-array grid input
    ``np.array([P_1d, T_1d], dtype=object)`` and scatter input
    ``np.empty(n, dtype=object); PT[i] = (Pi, Ti)``.
  * Returns an object whose attributes (``.rho``, ``.G``, ``.S``, ...)
    match the upstream SeaFreeze result.

If the Rust extension isn't importable for some reason (e.g. wheel
build failed), we transparently fall back to the upstream package so
the rest of watereos keeps working.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    import watereos_rs as _wrs
    _HAVE_RUST = True
except ImportError:  # pragma: no cover - exercised only when Rust missing
    _wrs = None
    _HAVE_RUST = False

if not _HAVE_RUST:  # pragma: no cover
    try:
        import seafreeze.seafreeze as _sf_ref
    except ImportError as exc:
        raise ImportError(
            "Neither watereos_rs (Rust seafreeze backend) nor the upstream "
            "'seafreeze' package is available; install one of them."
        ) from exc


def _is_scatter_PTm(PTm: Any) -> bool:
    """True when PTm matches SeaFreeze's scatter-input shape.

    Mirrors ``seafreeze.seafreeze._is_scatter`` exactly:

      * ``PTm[0]`` is a tuple ``(P, T)``, **or**
      * ``PTm.shape == (1, 2)`` (or ``(1, 3)``) holding scalars.

    Anything else is treated as grid input
    (``np.array([P_1d, T_1d], dtype=object)``).
    """
    try:
        first = PTm[0]
    except (IndexError, KeyError, TypeError):
        return False
    if isinstance(first, tuple):
        return True
    shape = getattr(PTm, "shape", None)
    if shape in ((1, 2), (1, 3)):
        try:
            return all(np.isscalar(PTm[i]) for i in range(shape[1]))
        except (IndexError, TypeError):
            return False
    return False


def _split_scatter(PTm: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract contiguous P and T 1-D arrays from a scatter PTm object array."""
    shape = getattr(PTm, "shape", None)
    if shape == (1, 2) or shape == (1, 3):
        # Singleton scatter: a single (P, T[, m]) row of scalars.
        return (
            np.ascontiguousarray(np.array([float(PTm[0])])),
            np.ascontiguousarray(np.array([float(PTm[1])])),
        )
    pts = np.asarray([(t[0], t[1]) for t in PTm], dtype=np.float64)
    return np.ascontiguousarray(pts[:, 0]), np.ascontiguousarray(pts[:, 1])


def getProp(PTm: Any, phase: str) -> SimpleNamespace:
    """SeaFreeze-compatible ``getProp`` powered by the Rust backend.

    Parameters mirror ``seafreeze.seafreeze.getProp``.
    """
    if not _HAVE_RUST:  # pragma: no cover
        return _sf_ref.getProp(PTm, phase)

    if _is_scatter_PTm(PTm):
        P, T = _split_scatter(PTm)
        result = _wrs.sf_getprop_scatter(phase, P, T)
    else:
        # Grid input: PTm[0] = pressures (1-D), PTm[1] = temperatures (1-D).
        P = np.ascontiguousarray(np.asarray(PTm[0], dtype=np.float64).ravel())
        T = np.ascontiguousarray(np.asarray(PTm[1], dtype=np.float64).ravel())
        result = _wrs.sf_getprop_grid(phase, P, T)
    return SimpleNamespace(**result)


def getProp_grid(phase: str, P: np.ndarray, T: np.ndarray) -> SimpleNamespace:
    """Direct grid-mode entry that skips PTm packing/parsing overhead."""
    if not _HAVE_RUST:  # pragma: no cover
        PT = np.array([np.asarray(P), np.asarray(T)], dtype=object)
        return _sf_ref.getProp(PT, phase)
    P = np.ascontiguousarray(np.asarray(P, dtype=np.float64))
    T = np.ascontiguousarray(np.asarray(T, dtype=np.float64))
    return SimpleNamespace(**_wrs.sf_getprop_grid(phase, P, T))


def getProp_scatter(phase: str, P: np.ndarray, T: np.ndarray) -> SimpleNamespace:
    """Direct scatter-mode entry for paired (P, T) inputs."""
    if not _HAVE_RUST:  # pragma: no cover
        PT = np.empty(len(P), dtype=object)
        PT[:] = list(zip(np.asarray(P), np.asarray(T)))
        return _sf_ref.getProp(PT, phase)
    P = np.ascontiguousarray(np.asarray(P, dtype=np.float64))
    T = np.ascontiguousarray(np.asarray(T, dtype=np.float64))
    if P.shape != T.shape:
        raise ValueError("scatter getProp requires equal-shape P and T")
    return SimpleNamespace(**_wrs.sf_getprop_scatter(phase, P, T))


def have_rust() -> bool:
    """True iff the native Rust backend is in use."""
    return _HAVE_RUST
