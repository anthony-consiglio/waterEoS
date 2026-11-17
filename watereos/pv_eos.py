"""
Volume roots of equations of state written as pressure P(V, T).

Some published equations give the pressure as an explicit function of
volume and temperature, with no Gibbs energy: the empirical polynomial of
Mishima & Sumita (2023) and the potential-energy-landscape equations of the
TIP4P/2005 and q-TIP4P/F model waters.  At each temperature these are
polynomials in V.  To evaluate them at a given (T, P) the isotherm has to be
solved for V, and below a liquid-liquid critical point the isotherm has a
van der Waals loop with more than one root.

This module finds every mechanically stable root (dP/dV < 0) inside a volume
window and picks the one with the lowest Gibbs energy along the isotherm,

    G(V) - G(V_lo) = P V - P V_lo - integral_{V_lo}^{V} P(V') dV',

which is the Maxwell construction.  Only differences of G at the same
temperature are needed, so a function of T alone, which a P(V, T) equation
does not determine, cancels.

Roots are bracketed either on a volume grid or between supplied stationary
points and refined by safeguarded regula falsi. Brackets keep the search
inside the volume window.
"""

import numpy as np

_N_GRID = 801
_MAX_ITER = 100


def polyval_rows(coeffs, V):
    """Evaluate one polynomial per row: sum_k coeffs[:, k] V^k.

    Parameters
    ----------
    coeffs : array (N, d+1)
        Coefficients in ascending powers of V, one row per point.
    V : array (N,) or (N, K)
        Volumes; a 2-D array evaluates each row at K volumes.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        out = np.zeros_like(V)
        for k in range(coeffs.shape[1] - 1, -1, -1):
            out = out * V + coeffs[:, k]
        return out
    out = np.zeros_like(V)
    for k in range(coeffs.shape[1] - 1, -1, -1):
        out = out * V + coeffs[:, k][:, None]
    return out


def deriv_coeffs(coeffs):
    """Coefficients of dP/dV from those of P(V) (ascending powers)."""
    coeffs = np.asarray(coeffs, dtype=float)
    k = np.arange(1, coeffs.shape[1])
    return coeffs[:, 1:] * k


def integral_coeffs(coeffs):
    """Coefficients of the antiderivative of P(V) with zero constant."""
    coeffs = np.asarray(coeffs, dtype=float)
    k = np.arange(1, coeffs.shape[1] + 1)
    return np.concatenate([np.zeros((coeffs.shape[0], 1)), coeffs / k],
                          axis=1)


def volume_roots(coeffs, P, V_lo, V_hi, n_grid=_N_GRID, max_roots=6,
                 stationary=None):
    """All mechanically stable volume roots of P(V) = P in [V_lo, V_hi].

    Parameters
    ----------
    coeffs : array (N, d+1)
        P(V) at each point's temperature, ascending powers of V.
    P : array (N,)
        Target pressures, in the units of P(V).
    V_lo, V_hi : float
        Volume window, in the units of V.
    n_grid : int
        Grid points used to bracket the roots.
    max_roots : int
        Maximum number of stable roots kept per point.
    stationary : array (N, K), optional
        Sorted stationary volumes inside the window, padded with V_hi.
        These partition each isotherm into monotonic intervals and avoid
        missing two roots close together near a critical point.

    Returns
    -------
    V_roots : array (N, max_roots)
        Roots with dP/dV < 0 in ascending order, NaN-padded.
    G_rel : array (N, max_roots)
        Gibbs energy of each root relative to the volume V_lo on the same
        isotherm (units of P times V), NaN-padded.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    P = np.asarray(P, dtype=float)
    N = coeffs.shape[0]
    if stationary is None:
        Vg = np.broadcast_to(np.linspace(V_lo, V_hi, n_grid), (N, n_grid))
    else:
        Vg = np.column_stack((np.full(N, V_lo), stationary, np.full(N, V_hi)))
    Q = polyval_rows(coeffs, Vg) - P[:, None]

    # Stable roots: P falls through the target as V increases.
    cross = (Q[:, :-1] > 0.0) & (Q[:, 1:] <= 0.0)
    cross &= np.isfinite(P)[:, None]
    rows, cols = np.nonzero(cross)

    V_roots = np.full((N, max_roots), np.nan)
    G_rel = np.full((N, max_roots), np.nan)
    if rows.size == 0:
        return V_roots, G_rel

    a = Vg[rows, cols]
    b = Vg[rows, cols + 1]
    fa = Q[rows, cols].copy()
    fb = Q[rows, cols + 1].copy()
    c_r = coeffs[rows]
    p_r = P[rows]
    x = b.copy()
    for _ in range(_MAX_ITER):
        with np.errstate(divide='ignore', invalid='ignore'):
            x = b - fb * (b - a) / (fb - fa)
        bad = ~np.isfinite(x) | (x <= np.minimum(a, b)) | (x >= np.maximum(a, b))
        x = np.where(bad, 0.5 * (a + b), x)
        fx = polyval_rows(c_r, x) - p_r
        # Illinois modification keeps both ends moving.
        same = np.sign(fx) == np.sign(fb)
        a_new = np.where(same, a, b)
        fa_new = np.where(same, 0.5 * fa, fb)
        a, fa = a_new, fa_new
        b, fb = x, fx
        if np.all(np.abs(b - a) <= 1e-13 * np.maximum(1.0, np.abs(b))):
            break
    stable = polyval_rows(deriv_coeffs(c_r), b) < 0
    rows, roots, c_r, p_r = rows[stable], b[stable], c_r[stable], p_r[stable]

    # Gibbs energy along the isotherm relative to V_lo.
    ic = integral_coeffs(c_r)
    F_rel = -(polyval_rows(ic, roots) - polyval_rows(ic, np.full_like(roots, V_lo)))
    g = F_rel + p_r * (roots - V_lo)

    # Scatter into the padded arrays in ascending order of volume.
    order = np.lexsort((roots, rows))
    rows, roots, g = rows[order], roots[order], g[order]
    rank = np.zeros(rows.size, dtype=int)
    if rows.size > 1:
        starts = np.r_[True, rows[1:] != rows[:-1]]
        idx = np.arange(rows.size)
        first = np.maximum.accumulate(np.where(starts, idx, 0))
        rank = idx - first
    keep = rank < max_roots
    V_roots[rows[keep], rank[keep]] = roots[keep]
    G_rel[rows[keep], rank[keep]] = g[keep]
    return V_roots, G_rel


def stable_volume(V_roots, G_rel):
    """The root with the lowest Gibbs energy; NaN where there is none.

    Returns
    -------
    V : array (N,)
    n_roots : array (N,) of int
        Number of mechanically stable roots found at each point.
    """
    n_roots = np.sum(np.isfinite(V_roots), axis=1)
    G = np.where(np.isfinite(G_rel), G_rel, np.inf)
    idx = np.argmin(G, axis=1)
    V = V_roots[np.arange(V_roots.shape[0]), idx]
    V = np.where(n_roots > 0, V, np.nan)
    return V, n_roots


def volumetric_properties(V, P_V, P_T, P_VV):
    """Response functions from derivatives of P(V, T) at a root.

    Parameters
    ----------
    V : volume at the root.
    P_V, P_T, P_VV : (dP/dV)_T, (dP/dT)_V, (d2P/dV2)_T at the root.

    Returns
    -------
    Kt : isothermal bulk modulus -V (dP/dV)_T, in the units of P.
    alpha : isobaric expansivity (1/V)(dV/dT)_P = -(dP/dT)_V / (V (dP/dV)_T).
    Kp : (dKt/dP)_T = -1 - V (d2P/dV2) / (dP/dV).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        Kt = -V * P_V
        alpha = -P_T / (V * P_V)
        Kp = -1.0 - V * P_VV / P_V
    return Kt, alpha, Kp
