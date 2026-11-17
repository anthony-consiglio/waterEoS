"""Mishima & Sumita (2023), case 3, 7 × 7 volumetric equation."""

from functools import lru_cache
from fractions import Fraction
from math import comb

import numpy as np
from numpy.polynomial.polynomial import polyroots, polyval

from watereos import pv_eos
from watereos.batch_getprop import getProp as _getProp

from .params import COEFFS

# Retain the dense branches while excluding remote polynomial oscillations.
_VOLUME_WINDOW = (0.70, 1.25)  # cm³/g
_CHUNK_SIZE = 2048


@lru_cache(maxsize=1)
def _shifted_coefficients():
    """Recenter at V=1 and T/300=1 before rounding the transformed table."""
    a = [[Fraction(str(c)) for c in row] for row in COEFFS]
    return np.array([
        [float(sum(a[i][j] * comb(i, k) * comb(j, l)
                   for i in range(k, 8) for j in range(l, 8)))
         for l in range(8)]
        for k in range(8)
    ])


def _isotherms(T):
    b = _shifted_coefficients()
    t = (T - 300.0) / 300.0
    c = polyval(t, b.T).T
    c_t = polyval(t, (b[:, 1:] * np.arange(1, 8)).T).T / 300.0
    return c, c_t


def _pressure_derivatives(T, v):
    c, c_t = _isotherms(T)
    p_v = pv_eos.deriv_coeffs(c)
    p_vv = pv_eos.deriv_coeffs(p_v)
    u = v - 1.0
    p, p_v, p_vv, p_t = (pv_eos.polyval_rows(a, u)
                         for a in (c, p_v, p_vv, c_t))
    return p, p_v, p_vv, p_t


def _solve(T, P):
    unique_T, index = np.unique(T, return_inverse=True)
    c, _ = _isotherms(unique_T)
    lo, hi = (v - 1.0 for v in _VOLUME_WINDOW)
    stationary = np.full((unique_T.size, 6), hi)
    for row, derivative in enumerate(pv_eos.deriv_coeffs(c)):
        if np.all(np.isfinite(derivative)):
            roots = polyroots(derivative)
            real = roots.real[(np.abs(roots.imag) < 1e-10)
                              & (roots.real > lo) & (roots.real < hi)]
            stationary[row, :real.size] = np.sort(real)
    roots, gibbs = pv_eos.volume_roots(c[index], P, lo, hi,
                                      stationary=stationary[index])
    return roots + 1.0, gibbs


def _flat_inputs(first, second):
    a, b = np.broadcast_arrays(np.asarray(first, float), np.asarray(second, float))
    return a.ravel(), b.ravel()


def volume_roots(T_K, P_MPa):
    """Stable polynomial volume roots in cm³/g, ordered and padded with NaN."""
    T, P = _flat_inputs(T_K, P_MPa)
    roots = np.full((T.size, 6), np.nan)
    for start in range(0, T.size, _CHUNK_SIZE):
        block = slice(start, start + _CHUNK_SIZE)
        roots[block], _ = _solve(T[block], P[block])
    return roots


def compute_batch(T_K, P_MPa):
    """Return rho, V, Kt and alpha at the lowest-Gibbs stable root.

    Outputs are in kg/m³, m³/kg, MPa and 1/K, respectively. Missing
    roots in the search window yield NaN. The fit gives only volumetric
    properties, and the 135 K atmospheric-pressure root is the dense one.
    """
    T, P = _flat_inputs(T_K, P_MPa)
    v = np.empty_like(T)
    for start in range(0, T.size, _CHUNK_SIZE):
        block = slice(start, start + _CHUNK_SIZE)
        v[block], _ = pv_eos.stable_volume(*_solve(T[block], P[block]))
    _, p_v, p_vv, p_t = _pressure_derivatives(T, v)
    kt, alpha, _ = pv_eos.volumetric_properties(v, p_v, p_t, p_vv)
    return {'rho': 1000.0 / v, 'V': v * 1e-3, 'Kt': kt, 'alpha': alpha}


def getProp(PT, phase=None):
    """Evaluate pressure-temperature grid or scatter input."""
    return _getProp(PT, compute_batch)
