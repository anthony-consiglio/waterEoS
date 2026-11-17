"""
Moynihan (1996) two-species / nonideal-solution model: core.

The relative molar Gibbs energy of the mixture of species A (LDA-like) and
B (HDA-like), Eq. 4 (p. 412), with X the mole fraction of B:

    dG(X) = G - G_A0 - V_A0 (P - P0)
          = X [dH0 - T dS0 + dV0 (P - P0)]
            + R T [X ln X + (1 - X) ln(1 - X)] + W X (1 - X)

with dH0, dS0, dV0 and W constant.  The equilibrium fraction solves Eq. 11

    dH0 - T dS0 + dV0 (P - P0) + R T ln(X / (1 - X)) + W (1 - 2X) = 0,

the liquid-liquid (amorphous-amorphous) transition is the line
dH0 - T dS0 + dV0 (P - P0) = 0 below Tc = W / 2R (Eqs. 12, 7), and the
excess heat capacity and the excess parts of V alpha and V kappa follow in
closed form (Eqs. 17-19).

What this model determines, and so what this module returns
-----------------------------------------------------------
The pure-species Gibbs energies G_A0(T) and V_A0 are not modeled, so the
model gives no entropy, enthalpy, Gibbs energy or total heat capacity.  It
determines:

- ``x``: the fraction of the low-density species, x = 1 - X.
- ``Cp_ex``: the excess (conformational) heat capacity of Eq. 17, per kg.
  The paper compares Cp(T) - Cp(277 K) with Cp_ex(T) - Cp_ex(277 K) (Fig. 5).
- ``V_alpha_ex`` and ``V_kappa_ex``: the excess volume-weighted
  expansivity and compressibility of Eqs. 18 and 19, on a mass basis.
- ``V``, ``rho``, ``alpha``, ``Kt``: from the construction the paper uses
  for its temperature-volume phase diagram (p. 421, Fig. 7): background
  expansivity 0.46e-3 1/K and compressibility 1.62e-5 1/atm, independent of
  T and P and equal for both species, around a critical specific volume of
  1.01 cm3/g.  Here that is written as

      v(T, P) = v_bkgr(T, P) + (X - 1/2) dV0 / M,
      v_bkgr  = v_c exp[alpha_b (T - Tc) - kappa_b (P - Pc)],

  so that v = v_c at the model's critical point and the excess expansivity
  and compressibility are exactly Eqs. 18 and 19.  The absolute volume
  is anchored at the paper's reported v_c. The paper starts its coexistence
  construction from the 135 K volumes, so the expression above is an
  approximate reconstruction using the same constant background coefficients.

Phase lines (functions below): the critical point, the coexistence
pressure, both spinodal pressures and the coexisting volumes, all in
closed form.
"""

import numpy as np

from watereos.two_state_branch import normalize_branch, select_branch

from . import params as P

_EPS = 1e-15


def _reaction_gibbs(T, P_Pa):
    """dG0 + dV0 (P - P0) of A -> B, J/mol."""
    return P.DH0 - T * P.DS0 + P.DV0 * (P_Pa - P.P0)


def critical_point():
    """Liquid-liquid critical point of the model.

    Returns
    -------
    dict with T_K, p_MPa, rho (kg/m3) and x.  Tc = W / 2R (Eq. 7) and Pc on
    the coexistence line at Tc (Eq. 12).  The paper prints 217 +- 10 K and
    380 atm (38.5 MPa); with the printed constants Pc is 38.1 MPa.
    """
    Tc = P.W / (2.0 * P.R)
    Pc = P.P0 - (P.DH0 - Tc * P.DS0) / P.DV0
    return {'T_K': Tc, 'p_MPa': Pc / 1e6, 'rho': 1.0 / P.V_C, 'x': 0.5}


def coexistence_pressure(T_K):
    """Pressure (MPa) of the liquid-liquid transition, Eq. 12.

    NaN above the critical temperature, where there is no transition.
    """
    T = np.asarray(T_K, dtype=float)
    Pa = P.P0 - (P.DH0 - T * P.DS0) / P.DV0
    Tc = P.W / (2.0 * P.R)
    return np.where((T > 0) & (T <= Tc), Pa / 1e6, np.nan)


def spinodal_pressures(T_K):
    """Spinodal pressures (MPa) at temperature T, below Tc.

    Returns
    -------
    p_hdl, p_ldl : arrays
        p_hdl is the lowest pressure at which the high-density liquid
        exists (its spinodal; Mishima's HDA -> LDA transformation on
        decompression, 500 atm at 135 K); p_ldl is the highest pressure at
        which the low-density liquid exists (LDA -> HDA on compression,
        3200 atm at 135 K).  NaN above Tc.
    """
    T = np.asarray(T_K, dtype=float)
    disc = 1.0 - 2.0 * P.R * T / P.W
    with np.errstate(invalid='ignore'):
        root = np.sqrt(disc)
    out = []
    for X in (0.5 * (1.0 + root), 0.5 * (1.0 - root)):
        with np.errstate(divide='ignore', invalid='ignore'):
            g = (P.DH0 - T * P.DS0 + P.R * T * np.log(X / (1.0 - X))
                 + P.W * (1.0 - 2.0 * X))
        Pa = P.P0 - g / P.DV0
        out.append(np.where((T > 0) & (disc >= 0.0), Pa / 1e6, np.nan))
    return out[0], out[1]


def compute_batch(T_K, P_MPa, branch='stable'):
    """Properties determined by the model, on flat (T, P) arrays.

    Parameters
    ----------
    T_K, P_MPa : 1-D arrays
    branch : {'stable', 'hdl', 'ldl'}
        Liquid branch; see ``watereos.two_state_branch``.

    Returns
    -------
    dict of 1-D arrays: x, rho, V, alpha, Kt, Cp_ex, V_alpha_ex,
    V_kappa_ex (and ``metastable`` for a named branch).
    """
    branch = normalize_branch(branch)
    T = np.atleast_1d(np.asarray(T_K, dtype=float))
    P_MPa = np.atleast_1d(np.asarray(P_MPa, dtype=float))
    T, P_MPa = np.broadcast_arrays(T, P_MPa)
    T = np.array(T, dtype=float)
    P_Pa = np.array(P_MPa, dtype=float) * 1e6
    RT = P.R * T

    # In the package's normalized form the variable is the low-density
    # fraction x = 1 - X, and the field is (G_LDA - G_HDA) / RT.
    with np.errstate(divide='ignore', invalid='ignore'):
        L = np.where(T > 0, -_reaction_gibbs(T, P_Pa) / RT, np.nan)
        omega = np.where(T > 0, P.W / RT, np.nan)
    x, metastable = select_branch(L, omega, branch)
    missing = ~np.isfinite(x)
    X = np.clip(1.0 - np.where(missing, 0.5, x), _EPS, 1.0 - _EPS)

    with np.errstate(divide='ignore', invalid='ignore'):
        XX = X * (1.0 - X)
        g_xx = RT / XX - 2.0 * P.W                 # d2 dG / dX2
        g_xT = -P.DS0 + P.R * np.log(X / (1.0 - X))   # d2 dG / dX dT
        g_xP = P.DV0                               # d2 dG / dX dP
        X_T = -g_xT / g_xx
        X_P = -g_xP / g_xx
        Cp_ex = T * g_xT ** 2 / g_xx / P.M_H2O     # Eq. 17, J/(kg K)

        crit = critical_point()
        Tc, Pc = crit['T_K'], crit['p_MPa'] * 1e6
        v_b = P.V_C * np.exp(P.ALPHA_BKGR * (T - Tc)
                             - P.KAPPA_BKGR * (P_Pa - Pc))
        dv = P.DV0 / P.M_H2O                      # m3/kg
        V = v_b + (X - 0.5) * dv
        V_alpha_ex = dv * X_T                   # m³/(kg K), Eq. 18
        V_kappa_ex = -dv * X_P * 1e6            # m³/(kg MPa), Eq. 19
        V_T = P.ALPHA_BKGR * v_b + V_alpha_ex
        V_P = -P.KAPPA_BKGR * v_b - V_kappa_ex / 1e6
        alpha = V_T / V
        Kt = -V / V_P / 1e6

    out = {
        'x': 1.0 - X, 'rho': 1.0 / V, 'V': V, 'alpha': alpha, 'Kt': Kt,
        'Cp_ex': Cp_ex, 'V_alpha_ex': V_alpha_ex, 'V_kappa_ex': V_kappa_ex,
    }
    for key in out:
        out[key] = np.where(missing, np.nan, out[key])
    if branch != 'stable':
        out['metastable'] = np.asarray(metastable, dtype=bool)
    return out


def compute_properties(T_K, P_MPa, branch='stable'):
    """Scalar convenience wrapper around :func:`compute_batch`."""
    out = compute_batch(np.array([T_K]), np.array([P_MPa]), branch=branch)
    return {k: (bool(v[0]) if k == 'metastable' else float(v[0]))
            for k, v in out.items()}


def coexistence_volumes(T_K):
    """Specific volumes (m3/kg) of the coexisting liquids at T < Tc.

    Returns
    -------
    v_hdl, v_ldl : arrays, NaN above Tc.
    """
    T = np.atleast_1d(np.asarray(T_K, dtype=float))
    p = coexistence_pressure(T)
    h = compute_batch(T, p, branch='hdl')['V']
    lo = compute_batch(T, p, branch='ldl')['V']
    return h, lo
