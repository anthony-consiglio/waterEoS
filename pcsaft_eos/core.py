"""Pure-water PC-SAFT and PC-SAFT-TS. Internal density is mol/m³."""

import numpy as np
from numpy.polynomial.polynomial import polyval

from vapor_eos import params as ideal
from watereos.two_state_general import M_H2O, R

from .params import FITS

# Gross and Sadowski (2001), Table 1. Columns index powers of packing
# fraction and rows contain the three chain-length contributions.
_A = np.array([
    [0.9105631445, 0.6361281449, 2.6861347891, -26.547362491,
     97.759208784, -159.59154087, 91.297774084],
    [-0.3084016918, 0.1860531159, -2.5030047259, 21.419793629,
     -65.255885330, 83.318680481, -33.746922930],
    [-0.0906148351, 0.4527842806, 0.5962700728, -1.7241829131,
     -4.1302112531, 13.776631870, -8.6728470368],
])
_B = np.array([
    [0.7240946941, 2.2382791861, -4.0025849485, -21.003576815,
     26.855641363, 206.55133841, -355.60235612],
    [-0.5755498075, 0.6995095521, 3.8925673390, -17.215471648,
     192.67226447, -161.82646165, -165.20769346],
    [0.0976883116, -0.2557574982, -9.1558561530, 20.642075974,
     -38.804430052, 93.626774077, -29.666905585],
])
_NA_A3 = 6.02214076e-7       # (molecule/Å³) / (mol/m³)
_DENSITY_STEP = 1e-5        # relative imaginary step for pressure
_ETA_MIN = 0.12             # restrict the T–P inversion to dense roots
_ETA_MAX = 0.68
_BATCH_SIZE = 256


def _packing_scale(T, fit):
    d = fit.sigma * (1.0 - 0.12 * np.exp(-3.0 * fit.epsilon / T))
    return np.pi / 6 * _NA_A3 * fit.m * d**3


def _fraction(T, n, fit):
    if not fit.l_bond[1]:
        return np.zeros_like(n + T)
    # Eqs. 11–12 agree with the cold-side fractions in 2025 Fig. 12.
    # The sign in the printed 2024 Eq. 13 would give 1-f.
    exponent = (fit.dh - fit.dp / n) / T - fit.ds
    # Stable logistic for real and complex input used in differentiation.
    q = np.exp(np.where(exponent.real >= 0, -exponent, exponent))
    return np.where(exponent.real >= 0, q / (1 + q), 1 / (1 + q))


def _unbonded(a):
    """Positive solution of a X² + X = 1."""
    return 2.0 / (1.0 + np.sqrt(1.0 + 4.0 * a))


def _association(T, n_sigma3, g, f, fit):
    """Association free energy with one unbonded fraction per water type."""
    eh, kh = fit.h_bond
    h = n_sigma3 * g * kh * np.expm1(eh / T)
    if not fit.l_bond[1]:
        xh = _unbonded(2 * h)
        return 4 * (np.log(xh) + (1 - xh) / 2)

    el, kl = fit.l_bond
    ec, kc = fit.cross_bond or ((eh + el) / 2.0, np.sqrt(kh * kl))
    ell = n_sigma3 * g * kl * np.expm1(el / T)
    cross = n_sigma3 * g * kc * np.expm1(ec / T)
    # Each site bonds only to the opposite partner kind. 2B has one of
    # each kind and 4C has two. Donor–acceptor symmetry reduces Eq. 17.
    a, b = 2 * (1 - f) * h, f * cross
    c, d = 2 * (1 - f) * cross, f * ell
    xh, xl = _unbonded(a + b), _unbonded(c + d)
    for _ in range(50):
        fh = xh * (1 + a * xh + b * xl) - 1
        fl = xl * (1 + c * xh + d * xl) - 1
        jh, jl = 1 + 2 * a * xh + b * xl, 1 + c * xh + 2 * d * xl
        det = jh * jl - b * c * xh * xl
        dh = (jl * fh - b * xh * fl) / det
        dl = (jh * fl - c * xl * fh) / det
        # Limit the step to keep both unbonded fractions positive.
        step = 1 / np.maximum(1.0, 2.0 * np.maximum(
            dh.real / xh.real, dl.real / xl.real))
        xh, xl = xh - step * dh, xl - step * dl
        if np.all(np.maximum(np.abs(dh), np.abs(dl)) < 1e-13):
            break
    else:
        error = np.maximum(abs(xh * (1 + a * xh + b * xl) - 1),
                           abs(xl * (1 + c * xh + d * xl) - 1))
        xh, xl = np.where(error < 1e-10, xh, np.nan), np.where(error < 1e-10, xl, np.nan)
    return (4 * (1 - f) * (np.log(xh) + (1 - xh) / 2)
            + 2 * f * (np.log(xl) + (1 - xl) / 2))


def _residual(T, n, fit):
    """Dimensionless residual Helmholtz energy Aʳ/(RT) per mole."""
    m = fit.m
    eta = n * _packing_scale(T, fit)
    om = 1 - eta
    g = (1 - eta / 2) / om**3
    hc = m * (4 * eta - 3 * eta**2) / om**2 - (m - 1) * np.log(g)
    weights = np.array([1.0, (m - 1) / m, (m - 1) * (m - 2) / m**2])
    i1, i2 = polyval(eta, weights @ _A), polyval(eta, weights @ _B)
    c1 = 1 / (1 + m * (8 * eta - 2 * eta**2) / om**4
              + (1 - m) * (20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4)
              / (om * (2 - eta))**2)
    n_sigma3 = n * _NA_A3 * fit.sigma**3
    e = fit.epsilon / T
    dispersion = -np.pi * n_sigma3 * m**2 * e * (2 * i1 + m * c1 * e * i2)
    return hc + dispersion + _association(T, n_sigma3, g, _fraction(T, n, fit), fit)


def _pressure(T, n, fit):
    # Aʳ differentiated in complex density; the fraction varies along the
    # isotherm for TS-TV. This is first-derivative accurate to O(step²).
    slope = _residual(T, n * (1 + 1j * _DENSITY_STEP), fit).imag / _DENSITY_STEP
    return n * R * T * (1 + slope)


def pressure(T_K, rho, model='novak2025_pcsaft_ts_caf'):
    """Pressure (MPa) at specified T (K) and mass density (kg/m³)."""
    return _pressure(np.asarray(T_K, float), np.asarray(rho, float) / M_H2O,
                     FITS[model]) / 1e6


def _liquid_density(T, P, fit):
    """Find the highest-density downward pressure crossing on each isotherm."""
    scale = _packing_scale(T, fit)
    grid = np.linspace(_ETA_MAX, _ETA_MIN, 64)
    p = _pressure(T[:, None], grid[None, :] / scale[:, None], fit) - P[:, None]
    cross = (p[:, :-1] > 0) & (p[:, 1:] <= 0)
    available = cross.any(axis=1)
    n = np.full(T.shape, np.nan)
    index = cross[available].argmax(axis=1)
    scale = scale[available]
    lo, hi = grid[index + 1] / scale, grid[index] / scale
    t, target = T[available], P[available]
    for _ in range(29):
        if not lo.size:
            break
        mid = (lo + hi) / 2
        above = _pressure(t, mid, fit) > target
        lo, hi = np.where(above, lo, mid), np.where(above, mid, hi)
    n[available] = (lo + hi) / 2
    return n


def _ideal_temperature_terms(T):
    """IAPWS-95 ideal Helmholtz temperature function and its derivatives."""
    tau = ideal.T_C / T
    x = np.asarray(ideal.GAMMA0)[:, None] * tau
    weights = np.asarray(ideal.N0_EXP)[:, None]
    ex = np.exp(-x)
    om = -np.expm1(-x)
    f = (ideal.N0_1 + ideal.N0_2 * tau + ideal.N0_3 * np.log(tau)
         + np.sum(weights * np.log(om), axis=0))
    tf = ideal.N0_2 * tau + ideal.N0_3 + np.sum(weights * x * ex / om, axis=0)
    ttf = -ideal.N0_3 - np.sum(weights * x**2 * ex / om**2, axis=0)
    return f, tf, ttf


def _properties(T, P, n, fit):
    # Local differences of the smooth Helmholtz and pressure functions.
    h = 0.01
    dn = 0.01
    res = _residual(T, n, fit)
    rm, rp = _residual(T - h, n, fit), _residual(T + h, n, fit)
    p_m, p_p = _pressure(T, n - dn, fit), _pressure(T, n + dn, fit)
    dp_dn = (p_p - p_m) / (2 * dn)
    dp_dt = (_pressure(T + h, n, fit) - _pressure(T - h, n, fit)) / (2 * h)
    f, tf, ttf = _ideal_temperature_terms(T)
    mass = M_H2O
    volume = 1 / (n * mass)
    ar = R * T * res / mass
    a = ar + R * T / mass * (np.log(n * mass / ideal.RHO_C) + f)
    s = -(R / mass) * (np.log(n * mass / ideal.RHO_C) + f - tf
                        + res + T * (rp - rm) / (2 * h))
    cvr = -R * T / mass * ((rp - rm) / h + T * (rp - 2 * res + rm) / h**2)
    cv = cvr - R / mass * ttf
    kt = n * dp_dn
    alpha = dp_dt / kt
    cp = cv + T * volume * dp_dt**2 / kt
    ks = kt * cp / cv
    u = a + T * s
    props = {'rho': n * mass, 'V': volume, 'Kt': kt / 1e6,
             'alpha': alpha, 'S': s, 'Cp': cp, 'Cv': cv, 'Ks': ks / 1e6,
             'vel': np.sqrt(np.where((cv > 0) & (ks > 0), ks * volume, np.nan)),
             'A': a, 'G': a + P * volume, 'U': u, 'H': u + P * volume}
    if fit.l_bond[1]:
        props['x'] = _fraction(T, n, fit)
    return {k: np.where(kt > 0, v, np.nan) for k, v in props.items()}


PROPERTY_KEYS = ('rho', 'V', 'Kt', 'alpha', 'S', 'Cp', 'Cv', 'Ks',
                 'vel', 'A', 'G', 'U', 'H')


def compute_batch(T_K, P_MPa, model='novak2025_pcsaft_ts_caf'):
    """Compute dense-fluid properties from T in K and P in MPa."""
    fit = FITS[model]
    T, p = np.broadcast_arrays(np.asarray(T_K, float), np.asarray(P_MPa, float))
    T, p = T.ravel(), p.ravel() * 1e6
    keys = PROPERTY_KEYS + (('x',) if fit.l_bond[1] else ())
    out = {k: np.full(T.size, np.nan) for k in keys}
    valid = np.flatnonzero(np.isfinite(T) & np.isfinite(p) & (T >= 100) & (T <= 2000))
    for start in range(0, valid.size, _BATCH_SIZE):
        i = valid[start:start + _BATCH_SIZE]
        n = _liquid_density(T[i], p[i], fit)
        found = np.isfinite(n)
        j = i[found]
        for key, value in _properties(T[j], p[j], n[found], fit).items():
            out[key][j] = value
    return out
