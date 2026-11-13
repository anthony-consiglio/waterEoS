"""
Water vapor as a Gibbs function of temperature and pressure (NumPy backend).

Formulation
-----------
The IAPWS-95 Helmholtz function is phi0(tau, delta) + phir(tau, delta)
with tau = T_c/T and delta = rho/rho_c. At low density the residual part
is a power series, phir = b1(tau) delta + b2(tau) delta^2 + ..., whose
first two coefficients are the second and third virial coefficients.
Reverting the series from density to pressure gives a Gibbs function that
is explicit in (T, p), with no density iteration:

    g / (R T) = 1 + phi0(tau, q) + b1 q + (b2 - b1^2 / 2) q^2,

    q = p / (rho_c R T)        (the reduced ideal-gas density at T, p)

Every property below is an analytic derivative of this one function, so
the set is thermodynamically consistent to rounding. With

    z = 1 + b1 q + 2 k q^2,   k = b2 - b1^2 / 2      (compressibility factor)
    e = 1 - 2 k q^2
    a = 1 - (tau b1') q - 2 (tau k' + k) q^2

the volume is V = z R T / p, the isothermal bulk modulus p z / e, the
expansivity a / (T z) and Cp - Cv = R a^2 / e.

The ideal-gas part carries the IAPWS-95 reference state (zero internal
energy and entropy of the saturated liquid at the triple point), which is
the reference state of the liquids and ices of this package. The vapor is
therefore one phase, the same whichever liquid model it is paired with,
and equating its Gibbs energy with that of ice Ih or of a liquid gives
the sublimation or saturation pressure.

Range
-----
T from 130 K (the limit of the IAPWS-95 ideal-gas part) to about 450 K,
p from zero to somewhat above saturation. Points where the truncated
series has no meaning (V <= 0 or dV/dp >= 0, far beyond saturation) and
non-positive T or p return NaN.

Units follow SeaFreeze: T [K], p [MPa], G H U A [J/kg], S Cp Cv [J/(kg K)],
V [m^3/kg], rho [kg/m^3], Kt Ks [MPa], alpha [1/K], vel [m/s].
"""

import numpy as np

from . import params as P

KEYS = ('G', 'V', 'rho', 'S', 'H', 'U', 'A', 'Cp', 'Cv',
        'Kt', 'Ks', 'Kp', 'alpha', 'vel')

_N0E = np.array(P.N0_EXP)
_G0 = np.array(P.GAMMA0)


def _series(terms, powers):
    """f, tau f' and tau^2 f'' of f = sum n tau^t, from the powers tau^t."""
    f = d1 = d2 = 0.0
    for n, t in terms:
        u = n * powers[t]
        f = f + u
        d1 = d1 + t * u
        d2 = d2 + t * (t - 1.0) * u
    return f, d1, d2


def _tau_terms(T):
    """The temperature-only parts of the Gibbs function.

    Returns (tau, F, tF, ttF, b1, tb1, ttb1, k, tk, ttk): the ideal-gas
    function F(tau) = phi0 - ln(delta) with tau F' and tau^2 F'', then b1
    and k = b2 - b1^2/2 with the same two derivative combinations.
    """
    tau = P.T_C / T

    x = _G0[:, None] * tau.reshape(1, -1)
    ex = np.exp(-x)
    om = 1.0 - ex
    w = _N0E[:, None]
    lnt = np.log(tau)
    F = (P.N0_1 + P.N0_2 * tau + P.N0_3 * lnt
         + (w * np.log1p(-ex)).sum(axis=0).reshape(tau.shape))
    tF = (P.N0_2 * tau + P.N0_3
          + (w * x * ex / om).sum(axis=0).reshape(tau.shape))
    ttF = (-P.N0_3
           - (w * x * x * ex / (om * om)).sum(axis=0).reshape(tau.shape))

    s = np.sqrt(tau)           # tau^(1/2)
    r4 = np.sqrt(s)            # tau^(1/4)
    r8 = np.sqrt(r4)           # tau^(1/8)
    t2 = tau * tau
    t4 = t2 * t2
    t5 = t4 * tau
    t6 = t4 * t2
    powers = {-0.5: 1.0 / s, 0.5: s, 0.75: s * r4, 0.875: tau / r8,
              1.0: tau, 4.0: t4, 5.0: t5, 6.0: t6, 7.0: t6 * tau,
              9.0: t5 * t4, 10.0: t5 * t5, 12.0: t6 * t6}
    b1, tb1, ttb1 = _series(P.B1_TERMS, powers)
    b2, tb2, ttb2 = _series(P.B2_TERMS, powers)
    k = b2 - 0.5 * b1 * b1
    tk = tb2 - b1 * tb1
    ttk = ttb2 - tb1 * tb1 - b1 * ttb1
    return tau, F, tF, ttF, b1, tb1, ttb1, k, tk, ttk


def _assemble(T, p, terms):
    """All properties from the temperature terms; T, p and terms broadcast."""
    tau, F, tF, ttF, b1, tb1, ttb1, k, tk, ttk = terms
    p_pa = p * 1.0e6
    q = tau * (p_pa / P.P_STAR)
    q2 = q * q

    gam = 1.0 + np.log(q) + F + b1 * q + k * q2
    tgam = 1.0 + tF + (tb1 + b1) * q + (tk + 2.0 * k) * q2
    ttgam = -1.0 + ttF + (ttb1 + 2.0 * tb1) * q + (ttk + 4.0 * tk + 2.0 * k) * q2

    z = 1.0 + b1 * q + 2.0 * k * q2
    e = 1.0 - 2.0 * k * q2
    a = 1.0 - tb1 * q - 2.0 * (tk + k) * q2

    RT = P.R * T
    G = RT * gam
    H = RT * tgam
    S = P.R * (tgam - gam)
    Cp = -P.R * ttgam
    Cv = Cp - P.R * a * a / e
    V = z * RT / p_pa
    pV = z * RT
    Kt = p * z / e
    Ks = Kt * Cp / Cv
    out = {
        'G': G, 'V': V, 'rho': 1.0 / V, 'S': S, 'H': H,
        'U': H - pV, 'A': G - pV, 'Cp': Cp, 'Cv': Cv,
        'Kt': Kt, 'Ks': Ks, 'Kp': 2.0 * z / (e * e) - 1.0,
        'alpha': a / (T * z), 'vel': np.sqrt(Ks * 1.0e6 * V),
    }
    ok = (T > 0.0) & (p > 0.0) & (z > 0.0) & (e > 0.0) & (Cv > 0.0)
    if not ok.all():
        ok = np.broadcast_to(ok, G.shape)
        for key in KEYS:
            out[key] = np.where(ok, out[key], np.nan)
    return out


def compute_batch(T_K, P_MPa):
    """Vapor properties at paired points.

    Parameters
    ----------
    T_K, P_MPa : 1-D arrays of equal length
        Temperature [K] and pressure [MPa].

    Returns
    -------
    dict of 1-D arrays keyed by ``KEYS``.
    """
    T = np.asarray(T_K, dtype=float)
    p = np.asarray(P_MPa, dtype=float)
    if T.ndim != 1 or T.shape != p.shape:
        raise ValueError("compute_batch needs 1-D T and P of equal length")
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return _assemble(T, p, _tau_terms(T))


def compute_grid(T_K, P_MPa):
    """Vapor properties on the grid of every P with every T.

    The temperature-only terms are evaluated once per temperature.

    Returns
    -------
    dict of arrays of shape ``(len(P_MPa), len(T_K))`` keyed by ``KEYS``.
    """
    T = np.asarray(T_K, dtype=float)
    p = np.asarray(P_MPa, dtype=float)
    if T.ndim != 1 or p.ndim != 1:
        raise ValueError("compute_grid needs 1-D T and P")
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        terms = tuple(t[None, :] for t in _tau_terms(T))
        out = _assemble(T[None, :], p[:, None], terms)
    shape = (p.size, T.size)
    return {key: np.ascontiguousarray(np.broadcast_to(val, shape))
            for key, val in out.items()}


def sublimation_pressure(T_K):
    """Vapor pressure of ice Ih [MPa], IAPWS R14-08(2011); 50 K to 273.16 K."""
    theta = np.asarray(T_K, dtype=float) / P.T_TRIPLE
    s = sum(a * theta ** b for a, b in zip(P.SUBLIMATION_A, P.SUBLIMATION_B))
    return P.P_TRIPLE * np.exp(s / theta)


def saturation_pressure(T_K):
    """Vapor pressure of liquid water [MPa], Wagner & Pruss (1993), the
    auxiliary saturation equation of IAPWS-95; 273.16 K to 647.096 K."""
    T = np.asarray(T_K, dtype=float)
    v = 1.0 - T / P.T_C
    s = sum(a * v ** e for a, e in zip(P.SATURATION_A, P.SATURATION_E))
    return P.P_C * np.exp(P.T_C / T * s)
