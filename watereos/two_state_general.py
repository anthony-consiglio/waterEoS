"""
General two-state (two-structure) liquid with a regular-solution mixing term.

Many published equations of state for water and for model waters write the
molar Gibbs energy of the liquid as a mixture of two interconvertible local
structures, A (high density) and B (low density):

    G(T, P; x) = G_A(T, P) + x * dG(T, P)
                 + (R T / N) * [x ln x + (1 - x) ln(1 - x)]
                 + W(T, P) * x * (1 - x)

where x is the fraction of the low-density structure B, dG = G_B - G_A, W is
the interaction (non-ideality) term in energy units, and N is the number of
molecules per mixing unit (N = 1 for the usual form; the "clustering"
variants divide the ideal mixing entropy by N).  The equilibrium fraction
minimizes G at fixed (T, P):

    dG + (R T / N) ln(x / (1 - x)) + W (1 - 2x) = 0.

Dividing by R T / N gives the normalized form of ``watereos.two_state_branch``
with L = N dG / (R T) and omega = N W / (R T), so the same branch selection
(``'stable'``, ``'hdl'``, ``'ldl'``) applies.

A model supplies G_A, dG and W as "fields": 6-tuples

    (f, df/dT, df/dP, d2f/dT2, d2f/dTdP, d2f/dP2)

of 1-D arrays, in J/mol with T in K and P in Pa.  This module then returns
all mixture and pure-state properties per kilogram, in the units of the rest
of the package (kg, m, s, K, MPa, J).

First derivatives of G use the envelope theorem (dG/dx = 0 at equilibrium).
Second derivatives include the relaxation of x:

    G_ab = G_ab|_x - (d2G/dx da)(d2G/dx db) / (d2G/dx2)

with a, b in {T, P}.

The helpers at the end build fields from the forms used in the literature:
polynomials in reduced variables, products with T, sums and multiples.
"""

import numpy as np

from watereos.two_state_branch import normalize_branch, select_branch

R = 8.314462618          # J/(mol K)
M_H2O = 0.018015268      # kg/mol
M_D2O = 0.020027508      # kg/mol (IAPWS)

_EPS = 1e-15

TWO_STATE_KEYS = (
    'rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv', 'Kt', 'Ks', 'alpha',
    'vel',
)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------

def zero_field(n):
    """A field that is identically zero."""
    z = np.zeros(n)
    return (z, z, z, z, z, z)


def constant_field(c, n):
    """A field equal to the constant c."""
    z = np.zeros(n)
    return (np.full(n, float(c)), z, z, z, z, z)


def add_fields(*fields):
    """Sum of fields."""
    return tuple(sum(parts) for parts in zip(*fields))


def scale_field(f, c):
    """c times a field (c a scalar or an array)."""
    return tuple(c * v for v in f)


def times_T(f, T):
    """The field T * f(T, P)."""
    v, vT, vP, vTT, vTP, vPP = f
    return (T * v, v + T * vT, T * vP, 2.0 * vT + T * vTT, vP + T * vTP,
            T * vPP)


def poly_field(coeffs, u, v, du_dT, dv_dP, scale=1.0):
    """Polynomial sum_{m,n} c_mn u^m v^n and its derivatives in (T, P).

    Parameters
    ----------
    coeffs : dict {(m, n): c_mn}
    u, v : 1-D arrays
        Reduced variables, linear in T and in P respectively
        (u = (T - T0) / sT, v = (P - P0) / sP).
    du_dT, dv_dP : float
        1 / sT and 1 / sP.
    scale : float
        Overall factor (for example R * Tc to give J/mol).
    """
    n = u.shape[0]
    f = np.zeros(n)
    fu = np.zeros(n)
    fv = np.zeros(n)
    fuu = np.zeros(n)
    fuv = np.zeros(n)
    fvv = np.zeros(n)

    def pw(base, k):
        if k < 0:
            return np.zeros(n)
        return base ** k

    for (m, k), c in coeffs.items():
        if c == 0.0:
            continue
        f += c * pw(u, m) * pw(v, k)
        if m >= 1:
            fu += c * m * pw(u, m - 1) * pw(v, k)
        if k >= 1:
            fv += c * k * pw(u, m) * pw(v, k - 1)
        if m >= 2:
            fuu += c * m * (m - 1) * pw(u, m - 2) * pw(v, k)
        if m >= 1 and k >= 1:
            fuv += c * m * k * pw(u, m - 1) * pw(v, k - 1)
        if k >= 2:
            fvv += c * k * (k - 1) * pw(u, m) * pw(v, k - 2)
    return (scale * f, scale * fu * du_dT, scale * fv * dv_dP,
            scale * fuu * du_dT ** 2, scale * fuv * du_dT * dv_dP,
            scale * fvv * dv_dP ** 2)


# ---------------------------------------------------------------------------
# Properties from second-order derivatives of a molar Gibbs energy
# ---------------------------------------------------------------------------

def properties_from_gibbs(G, G_T, G_P, G_TT, G_TP, G_PP, T, P_Pa, M):
    """Per-kilogram properties from a molar Gibbs energy and its derivatives.

    Returns a dict with the keys of ``TWO_STATE_KEYS``.  Where the liquid is
    mechanically unstable (V <= 0 or compressibility <= 0) the response
    functions are NaN.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        V = G_P / M
        rho = 1.0 / V
        S = -G_T / M
        Cp = -T * G_TT / M
        kappa_T = -G_PP / G_P                    # 1/Pa
        alpha = G_TP / G_P                       # 1/K
        Kt = 1.0 / kappa_T / 1e6                 # MPa
        Cv = Cp - T * V * alpha ** 2 / kappa_T
        kappa_S = kappa_T - T * V * alpha ** 2 / Cp
        Ks = 1.0 / kappa_S / 1e6
        vel = np.sqrt(1.0 / (rho * kappa_S))
        Gm = G / M
        H = Gm + T * S
        U = H - P_Pa * V
        A = Gm - P_Pa * V
    bad = ~(V > 0)
    for arr in (rho, V):
        arr[bad] = np.nan
    return {'rho': rho, 'V': V, 'S': S, 'G': Gm, 'H': H, 'U': U, 'A': A,
            'Cp': Cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks, 'alpha': alpha,
            'vel': vel}


# ---------------------------------------------------------------------------
# The two-state mixture
# ---------------------------------------------------------------------------

def compute_two_state(T_K, P_MPa, terms, M=M_H2O, N=1.0, branch='stable'):
    """Mixture and pure-state properties of a two-state liquid.

    Parameters
    ----------
    T_K, P_MPa : 1-D arrays
        Temperature (K) and pressure (MPa).
    terms : callable
        ``terms(T_K, P_Pa) -> (G_A, dG, W)``, each a field (see module
        docstring) in J/mol with derivatives with respect to T in K and P in
        Pa.
    M : float
        Molar mass in kg/mol (``M_H2O`` or ``M_D2O``).
    N : float
        Molecules per mixing unit; the ideal entropy of mixing is divided by
        N.  1 for the usual form.
    branch : {'stable', 'hdl', 'ldl'}
        Liquid branch; see ``watereos.two_state_branch``.

    Returns
    -------
    dict of 1-D arrays
        Mixture keys (rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, alpha, vel, x)
        and the same keys with suffix ``_A`` (pure high-density structure,
        x = 0) and ``_B`` (pure low-density structure, x = 1).  With a
        branch other than ``'stable'`` also ``metastable``.  The mixture
        properties are NaN where the requested branch does not exist.
    """
    branch = normalize_branch(branch)
    T = np.ascontiguousarray(np.atleast_1d(np.asarray(T_K, dtype=float)))
    P_MPa = np.ascontiguousarray(np.atleast_1d(np.asarray(P_MPa, dtype=float)))
    T, P_MPa = np.broadcast_arrays(T, P_MPa)
    T = np.array(T, dtype=float)
    P_Pa = np.array(P_MPa, dtype=float) * 1e6

    GA, dG, W = terms(T, P_Pa)
    GA = tuple(np.asarray(a, dtype=float) for a in GA)
    dG = tuple(np.asarray(a, dtype=float) for a in dG)
    W = tuple(np.asarray(a, dtype=float) for a in W)

    RTn = R * T / N
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        L = dG[0] / RTn
        omega = W[0] / RTn
    x, metastable = select_branch(L, omega, branch)
    missing = ~np.isfinite(x)
    xs = np.clip(np.where(missing, 0.5, x), _EPS, 1.0 - _EPS)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        mix = xs * np.log(xs) + (1.0 - xs) * np.log(1.0 - xs)
        lnr = np.log(xs / (1.0 - xs))
        xx = xs * (1.0 - xs)

        G = GA[0] + xs * dG[0] + RTn * mix + W[0] * xx
        G_T = GA[1] + xs * dG[1] + (R / N) * mix + W[1] * xx
        G_P = GA[2] + xs * dG[2] + W[2] * xx

        g_xT = dG[1] + (R / N) * lnr + W[1] * (1.0 - 2.0 * xs)
        g_xP = dG[2] + W[2] * (1.0 - 2.0 * xs)
        g_xx = RTn / xx - 2.0 * W[0]

        G_TT = GA[3] + xs * dG[3] + W[3] * xx - g_xT ** 2 / g_xx
        G_TP = GA[4] + xs * dG[4] + W[4] * xx - g_xT * g_xP / g_xx
        G_PP = GA[5] + xs * dG[5] + W[5] * xx - g_xP ** 2 / g_xx

    out = properties_from_gibbs(G, G_T, G_P, G_TT, G_TP, G_PP, T, P_Pa, M)
    for key in out:
        out[key] = np.where(missing, np.nan, out[key])
    out['x'] = np.where(missing, np.nan, xs)

    GB = add_fields(GA, dG)
    for suffix, f in (('_A', GA), ('_B', GB)):
        pure = properties_from_gibbs(*f, T, P_Pa, M)
        for key, val in pure.items():
            out[key + suffix] = val

    if branch != 'stable':
        out['metastable'] = np.asarray(metastable, dtype=bool)
    return out
