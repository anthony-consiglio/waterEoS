"""
Grenke & Elliott (2025) Tait-Tammann EoS: core engine.

Direct empirical correlation for liquid water — no two-state decomposition.
Valid range: 200-300 K, 0.1-400 MPa.

All internal calculations use SI units (Pa, K, m^3/kg, J/(kg*K)).

Reference:
  Grenke & Elliott, J. Phys. Chem. B 129, 1997-2012 (2025)
  Correction: J. Phys. Chem. B 129, 9850-9853 (2025)
"""

import math
import numpy as np
from scipy.special import expi
from . import params as P


# ═══════════════════════════════════════════════════════════════════════════
# 1. Base functions and derivatives
# ═══════════════════════════════════════════════════════════════════════════

def _v0_all(T):
    """
    Specific volume at reference pressure v0(T, P0) and derivatives.
    Eq 20-22.  Returns (v0, v0_T, v0_TT).
    """
    e1 = P.a1 * math.exp(P.a2 * T)
    e2 = P.a3 * math.exp(P.a4 * T)
    v0 = e1 + e2 + P.a5
    v0_T = P.a2 * e1 + P.a4 * e2
    v0_TT = P.a2**2 * e1 + P.a4**2 * e2
    return v0, v0_T, v0_TT


def _B_all(T):
    """
    Tait parameter B(T) and derivatives.
    Eq 23, 25-26.  Output in Pa (includes 1e8 factor).
    Returns (B, B_T, B_TT).
    """
    u = T / P.b2
    ub3 = u ** P.b3                       # u^b3
    base = 1.0 + ub3                      # 1 + u^b3
    basem = base ** (-P.b4)               # (1+u^b3)^(-b4)

    B = P.b1 * basem * 1e8

    # First derivative (eq 25)
    coeff = -P.b1 * P.b3 * P.b4
    B_T = coeff * ub3 * base ** (-P.b4 - 1.0) / T * 1e8

    # Second derivative (eq 26)
    factor = (P.b3 * P.b4 + 1.0) * ub3 - P.b3 + 1.0
    B_TT = P.b1 * P.b3 * P.b4 * ub3 * base ** (-P.b4 - 2.0) * factor / (T * T) * 1e8

    return B, B_T, B_TT


def _C_all(T):
    """
    Tait parameter C(T) and derivatives.
    Eq 24, 27-28.  Dimensionless.
    Returns (C, C_T, C_TT).
    """
    w = T / P.c2
    wc3 = w ** P.c3                       # w^c3
    base = wc3 + 1.0                      # w^c3 + 1
    basem = base ** (-P.c4)               # (w^c3+1)^(-c4)

    C = P.c1 * basem + P.c5

    # First derivative (eq 27)
    coeff = -P.c1 * P.c3 * P.c4
    C_T = coeff * wc3 * base ** (-P.c4 - 1.0) / T

    # Second derivative (eq 28)
    factor = (P.c3 * P.c4 + 1.0) * wc3 - P.c3 + 1.0
    C_TT = P.c1 * P.c3 * P.c4 * wc3 * base ** (-P.c4 - 2.0) * factor / (T * T)

    return C, C_T, C_TT


def _cp0(T):
    """Heat capacity at reference pressure cp0(T, P0).  Eq 29."""
    return P.d1 * math.exp(P.d2 * T) + P.d3


# ═══════════════════════════════════════════════════════════════════════════
# 2. Scalar property computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_properties(T_K, p_MPa):
    """
    Compute all thermodynamic properties at a single (T, P) point.

    Parameters
    ----------
    T_K   : float — temperature in K
    p_MPa : float — pressure in MPa

    Returns
    -------
    dict with keys: rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, Kp, alpha, vel
    """
    T = float(T_K)
    P_Pa = float(p_MPa) * 1e6

    # Step 1 — Base functions
    v0, v0_T, v0_TT = _v0_all(T)
    B, B_T, B_TT = _B_all(T)
    C, C_T, C_TT = _C_all(T)
    cp0 = _cp0(T)

    # Step 2 — Intermediate quantities (Appendix A of the reference)
    # BP = B + P is the shifted pressure variable in the Tait-Tammann equation
    BP = B + P_Pa
    BP0 = B + P.P0
    E = math.log(BP / BP0)                         # A.10: log pressure ratio
    F = B_T / BP                                    # A.25: dB/dT normalised by BP
    G_fn = B_T / BP0                                # A.26: dB/dT normalised by BP0
    H_fn = B_TT / BP                                # A.30: d²B/dT² normalised by BP
    I_fn = B_T**2 / BP**2                           # A.31: (dB/dT)² normalised by BP²
    G_fn_T = B_TT / BP0 - B_T**2 / BP0**2          # A.32: dG_fn/dT

    # Step 3 — Auxiliary integrals for thermodynamic path integration
    # (needed for entropy, enthalpy, and heat capacity; Eqs. A.39-A.45)
    dP = P_Pa - P.P0
    J = BP * (E - 1.0) + BP0                        # A.39: ∫C·(B+P')⁻¹ dP' term
    K = B_T * E                                     # A.40: T-derivative of J
    L = B_TT * E                                    # A.41: second T-derivative
    M = -B_T**2 * (1.0/BP - 1.0/BP0)               # A.42: mixed-derivative term
    R = dP - C * J                                  # A.43: volume integral
    # Q — CORRECTED sign (correction paper, Eq. A.44)
    Q = -C * K + C * G_fn * dP - C_T * J            # A.44: entropy-related integral
    N = (-C * L + C * M + C * G_fn_T * dP
         - 2.0 * C_T * K + 2.0 * C_T * G_fn * dP
         - C_TT * J)                                # A.45: Cp-related integral

    # Step 4 — Physical properties
    v = v0 * (1.0 - C * E)                          # eq 1
    rho = 1.0 / v

    kappa_T = v0 * C / (v * BP)                     # eq 2 (1/Pa)
    Kt = 1.0 / kappa_T / 1e6                        # MPa

    # v_T (eq 7 / A.13)
    v_T = v0 * (-C * F + C * G_fn - C_T * E) + v0_T * (1.0 - C * E)
    alpha = v_T / v                                 # eq 3 (1/K)

    # cp (eq 4 / A.46)
    cp = cp0 - T * (v0 * N + 2.0 * v0_T * Q + v0_TT * R)
    Cv = cp - T * v * alpha**2 / kappa_T            # standard thermo

    kappa_S = kappa_T - T * v * alpha**2 / cp       # eq A.53
    Ks = 1.0 / kappa_S / 1e6                        # MPa
    vel = math.sqrt(v / kappa_S)                    # eq 5 / A.55

    # Step 5 — Thermodynamic potentials via path integration
    # Reference state: (T0, P0) = (273.15 K, 0.1 MPa), matching IAPWS-95.
    # Temperature leg: (T0, P0) -> (T, P0) using cp0(T) = d1·exp(d2·T) + d3
    # expi() is the exponential integral Ei(x) = ∫_{-∞}^x e^t/t dt,
    # arising from ∫ exp(d2·T)/T dT in the entropy integral.
    dh_T = (P.d1 / P.d2) * (math.exp(P.d2 * T) - math.exp(P.d2 * P.T0)) + P.d3 * (T - P.T0)
    ds_T = P.d1 * (expi(P.d2 * T) - expi(P.d2 * P.T0)) + P.d3 * math.log(T / P.T0)

    # Pressure leg: (T, P0) -> (T, P)
    dh_P = v0 * R - T * (v0 * Q + v0_T * R)
    ds_P = -(v0 * Q + v0_T * R)

    S = P.S_OFFSET + ds_T + ds_P
    H = P.H_OFFSET + dh_T + dh_P
    G = H - T * S
    U = H - P_Pa * v
    A = G - P_Pa * v

    return {
        'rho': rho, 'V': v, 'S': S, 'G': G, 'H': H, 'U': U, 'A': A,
        'Cp': cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks, 'Kp': float('nan'),
        'alpha': alpha, 'vel': vel,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Vectorized batch computation
# ═══════════════════════════════════════════════════════════════════════════

def _v0_all_vec(T):
    """Vectorized v0(T), v0_T, v0_TT."""
    e1 = P.a1 * np.exp(P.a2 * T)
    e2 = P.a3 * np.exp(P.a4 * T)
    v0 = e1 + e2 + P.a5
    v0_T = P.a2 * e1 + P.a4 * e2
    v0_TT = P.a2**2 * e1 + P.a4**2 * e2
    return v0, v0_T, v0_TT


def _B_all_vec(T):
    """Vectorized B(T), B_T, B_TT in Pa."""
    u = T / P.b2
    ub3 = np.power(u, P.b3)
    base = 1.0 + ub3
    basem = np.power(base, -P.b4)

    B = P.b1 * basem * 1e8

    coeff = -P.b1 * P.b3 * P.b4
    B_T = coeff * ub3 * np.power(base, -P.b4 - 1.0) / T * 1e8

    factor = (P.b3 * P.b4 + 1.0) * ub3 - P.b3 + 1.0
    B_TT = P.b1 * P.b3 * P.b4 * ub3 * np.power(base, -P.b4 - 2.0) * factor / (T * T) * 1e8

    return B, B_T, B_TT


def _C_all_vec(T):
    """Vectorized C(T), C_T, C_TT."""
    w = T / P.c2
    wc3 = np.power(w, P.c3)
    base = wc3 + 1.0
    basem = np.power(base, -P.c4)

    C = P.c1 * basem + P.c5

    coeff = -P.c1 * P.c3 * P.c4
    C_T = coeff * wc3 * np.power(base, -P.c4 - 1.0) / T

    factor = (P.c3 * P.c4 + 1.0) * wc3 - P.c3 + 1.0
    C_TT = P.c1 * P.c3 * P.c4 * wc3 * np.power(base, -P.c4 - 2.0) * factor / (T * T)

    return C, C_T, C_TT


def compute_batch(T_K, p_MPa):
    """
    Vectorized computation of all thermodynamic properties.

    Parameters
    ----------
    T_K   : 1-D array — temperature in K
    p_MPa : 1-D array — pressure in MPa (same length as T_K)

    Returns
    -------
    dict of 1-D arrays with keys:
        rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, alpha, vel
    (Kp is NOT included — not computed in batch mode.)
    """
    T = np.asarray(T_K, dtype=float)
    P_Pa = np.asarray(p_MPa, dtype=float) * 1e6

    # Step 1 — Base functions
    v0, v0_T, v0_TT = _v0_all_vec(T)
    B, B_T, B_TT = _B_all_vec(T)
    C, C_T, C_TT = _C_all_vec(T)
    cp0 = P.d1 * np.exp(P.d2 * T) + P.d3

    # Step 2 — Placeholders
    BP = B + P_Pa
    BP0 = B + P.P0
    E = np.log(BP / BP0)
    F = B_T / BP
    G_fn = B_T / BP0
    G_fn_T = B_TT / BP0 - B_T**2 / BP0**2

    # Step 3 — Auxiliary integrals
    dP = P_Pa - P.P0
    J = BP * (E - 1.0) + BP0
    K = B_T * E
    L = B_TT * E
    M = -B_T**2 * (1.0 / BP - 1.0 / BP0)
    R = dP - C * J
    Q = -C * K + C * G_fn * dP - C_T * J                   # CORRECTED
    N = (-C * L + C * M + C * G_fn_T * dP
         - 2.0 * C_T * K + 2.0 * C_T * G_fn * dP
         - C_TT * J)

    # Step 4 — Physical properties
    v = v0 * (1.0 - C * E)
    rho = 1.0 / v

    kappa_T = v0 * C / (v * BP)
    Kt = 1.0 / kappa_T / 1e6

    v_T = v0 * (-C * F + C * G_fn - C_T * E) + v0_T * (1.0 - C * E)
    alpha = v_T / v

    cp = cp0 - T * (v0 * N + 2.0 * v0_T * Q + v0_TT * R)
    Cv = cp - T * v * alpha**2 / kappa_T

    kappa_S = kappa_T - T * v * alpha**2 / cp
    Ks = 1.0 / kappa_S / 1e6
    vel = np.sqrt(v / kappa_S)

    # Step 5 — Thermodynamic potentials
    # Temperature leg
    dh_T = (P.d1 / P.d2) * (np.exp(P.d2 * T) - np.exp(P.d2 * P.T0)) + P.d3 * (T - P.T0)
    ds_T = P.d1 * (expi(P.d2 * T) - expi(P.d2 * P.T0)) + P.d3 * np.log(T / P.T0)

    # Pressure leg
    dh_P = v0 * R - T * (v0 * Q + v0_T * R)
    ds_P = -(v0 * Q + v0_T * R)

    S = P.S_OFFSET + ds_T + ds_P
    H = P.H_OFFSET + dh_T + dh_P
    G = H - T * S
    U = H - P_Pa * v
    A = G - P_Pa * v

    return {
        'rho': rho, 'V': v, 'S': S, 'G': G, 'H': H, 'U': U, 'A': A,
        'Cp': cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks,
        'alpha': alpha, 'vel': vel,
    }
