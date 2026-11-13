"""
SeaFreeze-compatible interface for water vapor.

Usage (identical to SeaFreeze and the other *_eos modules):

    import numpy as np
    from vapor_eos import getProp

    # Grid mode
    PT = np.array([P_MPa_array, T_K_array], dtype=object)
    out = getProp(PT)
    out.G   # shape (len(P), len(T))

    # Scatter mode
    PT = np.empty((N,), dtype=object)
    PT[0] = (P0_MPa, T0_K)
    out = getProp(PT)
    out.G   # shape (N,)

For batched evaluation without the PT packing:

    from vapor_eos import compute_batch, compute_grid
    out = compute_batch(T_K, P_MPa)    # paired 1-D arrays -> dict of arrays
    out['G'], out['V'], out['S']
    out = compute_grid(T_K, P_MPa)     # arrays of shape (len(P), len(T))

The vapor is also reachable by name wherever the ices are, e.g.
``watereos._seafreeze_native.getProp_scatter('vapor', P, T)``.
"""

import numpy as np

from watereos._common import ThermodynamicStates, _is_grid_input

from . import params as P
from .core import KEYS, sublimation_pressure, saturation_pressure

# Backend dispatch: Rust, else NumPy. An installed watereos_rs that
# predates the vapor raises ImportError here too and falls through.
try:
    from watereos_rs import compute_batch_vapor as _backend_compute_batch
    from watereos_rs import compute_grid_vapor as _backend_compute_grid
except ImportError:
    from .core import compute_batch as _backend_compute_batch
    from .core import compute_grid as _backend_compute_grid


def _f64(x):
    return np.ascontiguousarray(x, dtype=np.float64)


def compute_batch(T_K, P_MPa):
    """Vapor properties at paired (T, P) points.

    Parameters
    ----------
    T_K, P_MPa : 1-D array_like of equal length
        Temperature [K] and pressure [MPa].

    Returns
    -------
    dict of 1-D arrays: G, V, rho, S, H, U, A, Cp, Cv, Kt, Ks, Kp, alpha,
    vel (SeaFreeze units). NaN where T or P is not positive and where the
    truncated virial series has no meaning (far beyond saturation).
    """
    return _backend_compute_batch(_f64(T_K), _f64(P_MPa))


def compute_grid(T_K, P_MPa):
    """Vapor properties on the grid of every P with every T.

    Parameters
    ----------
    T_K, P_MPa : 1-D array_like
        Temperatures [K] and pressures [MPa].

    Returns
    -------
    dict of arrays of shape ``(len(P_MPa), len(T_K))``, keyed as in
    ``compute_batch``.
    """
    return _backend_compute_grid(_f64(T_K), _f64(P_MPa))


def getProp(PT, phase=None):
    """Compute thermodynamic properties of water vapor.

    Parameters
    ----------
    PT : numpy array
        Grid mode: np.array([P_MPa, T_K], dtype=object)
        Scatter mode: object array of (P_MPa, T_K) tuples
    phase : str, optional
        Ignored. Accepted for SeaFreeze API compatibility.

    Returns
    -------
    ThermodynamicStates
        Object with attributes: G, V, rho, S, H, U, A, Cp, Cv,
        Kt, Ks, Kp, alpha, vel.
    """
    out = ThermodynamicStates()

    if _is_grid_input(PT):
        batch = compute_grid(np.ravel(PT[1]), np.ravel(PT[0]))
    else:
        pairs = np.array(PT.tolist(), dtype=float)
        batch = compute_batch(pairs[:, 1], pairs[:, 0])

    for k in KEYS:
        setattr(out, k, batch[k])

    out.PTM = PT
    return out


def _condensed(phase):
    """(T, p) -> (G, V) evaluator of a liquid model or a SeaFreeze phase."""
    from watereos.watereos import _CANONICAL

    model = _CANONICAL.get(str(phase).lower())
    if model is not None and model != 'vapor':
        from watereos import getProp as _getProp

        def evaluate(T, p):
            PT = np.empty(T.size, dtype=object)
            PT[:] = list(zip(p.tolist(), T.tolist()))
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                o = _getProp(PT, model)
            return np.asarray(o.G, dtype=float), np.asarray(o.V, dtype=float)

        return evaluate

    from watereos._seafreeze_native import getProp_scatter

    def evaluate(T, p):
        o = getProp_scatter(phase, p, T)
        return np.asarray(o.G, dtype=float), np.asarray(o.V, dtype=float)

    return evaluate


def vapor_pressure(T_K, phase='Ih', tol=1e-13, max_iter=30):
    """Pressure at which the vapor coexists with a condensed phase.

    Solves G_vapor(T, p) = G_phase(T, p) for p by Newton's method in ln p,
    so the result is the vapor pressure implied by the equations of state
    themselves: the sublimation pressure for an ice, the saturation
    pressure for a liquid (also when supercooled).

    Parameters
    ----------
    T_K : float or array_like
        Temperature(s) in K.
    phase : str
        A liquid model key of ``watereos.getProp`` (``'holten2014'``,
        ``'water1'``, ...) or a SeaFreeze ice name (``'Ih'``, ...).

    Returns
    -------
    ndarray, same shape as ``T_K``
        Pressure in MPa; NaN where the iteration finds no solution.
    """
    T_in = np.asarray(T_K, dtype=float)
    T = _f64(T_in.ravel())
    evaluate = _condensed(phase)

    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        p = _f64(np.where(T < P.T_TRIPLE, sublimation_pressure(T),
                          saturation_pressure(T)))
        for _ in range(max_iter):
            vap = compute_batch(T, p)
            g_c, v_c = evaluate(T, p)
            # d(G_vap - G_c)/d(ln p) = p (V_vap - V_c), within 1e-2 of R T
            step = (vap['G'] - g_c) / ((vap['V'] - v_c) * p * 1.0e6)
            p = _f64(p * np.exp(-step))
            if not np.any(np.abs(step) >= tol):
                break
        else:
            p = np.where(np.abs(step) < 1e-9, p, np.nan)
    return p.reshape(T_in.shape)
