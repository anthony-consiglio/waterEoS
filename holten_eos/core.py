"""
Holten, Sengers & Anisimov (2014) two-state EoS: core engine.

All internal calculations follow the MATLAB reference implementation
(HoltenJPCRD2014.m) exactly. Holten uses per-kg specific quantities
(R = 461.523087 J/(kg·K)), so no M_H2O conversion is needed.

Dual reduced variables:
  Background B: tau = T/Tc, pi = (P - P0)/(rho0*R*Tc)
  Field L, omega: t = (T - Tc)/Tc, p = (P - Pc)/(rho0*R*Tc)

The main entry point is compute_properties(T_K, p_MPa).

Reference: V. Holten, J. V. Sengers, M. A. Anisimov,
           J. Phys. Chem. Ref. Data 43, 014101 (2014).
"""

import math
import numpy as np
from . import params as P


# ═══════════════════════════════════════════════════════════════════════════
# 1. Reduced variables
# ═══════════════════════════════════════════════════════════════════════════

def _reduce(T_K, p_MPa):
    """
    Compute all reduced variables.

    Returns (tau, pi, t, p_red) where:
      tau = T/Tc                          (for background B)
      pi  = (P - P0) / (rho0*R*Tc)       (for background B)
      t   = (T - Tc)/Tc = tau - 1         (for field L, omega)
      p_red = (P - Pc) / (rho0*R*Tc)     (for field L, omega)
    """
    P_Pa = p_MPa * 1e6
    tau = T_K / P.Tc
    pi = (P_Pa - P.P0 * 1e6) / P.P_scale_Pa
    t = tau - 1.0
    p_red = (P_Pa - P.Pc * 1e6) / P.P_scale_Pa
    return tau, pi, t, p_red


# ═══════════════════════════════════════════════════════════════════════════
# 2. Background B(tau, pi) and derivatives (Eq. 12, Table 7)
# ═══════════════════════════════════════════════════════════════════════════

def _B_all(tau, pi):
    """Compute B and all derivatives in a single scalar loop.

    Returns (B, Bp, Bt, Bpp, Btp, Btt).
    Much faster than 6 separate numpy calls for N=20 terms.
    """
    B = Bp = Bt = Bpp = Btp = Btt = 0.0
    inv_pi = 1.0 / pi
    inv_tau = 1.0 / tau
    c_arr = P.c_bg; a_arr = P.a_bg; b_arr = P.b_bg; d_arr = P.d_bg
    for i in range(20):
        ci = c_arr[i]; ai = a_arr[i]; bi = b_arr[i]; di = d_arr[i]
        base = ci * (tau ** ai) * (pi ** bi) * math.exp(-di * pi)
        bdp = bi - di * pi  # b_i - d_i * pi
        B   += base
        Bp  += base * bdp * inv_pi
        Bt  += base * ai * inv_tau
        Bpp += base * (bdp * bdp - bi) * inv_pi * inv_pi
        Btp += base * ai * bdp * inv_tau * inv_pi
        Btt += base * ai * (ai - 1.0) * inv_tau * inv_tau
    return B, Bp, Bt, Bpp, Btp, Btt


# ═══════════════════════════════════════════════════════════════════════════
# 3. Hyperbolic field L and derivatives (Eq. 14)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_L(t, p_red):
    """
    Compute the hyperbolic field L and all derivatives.

    Returns (L, Lt, Lp, Ltt, Ltp, Lpp).
    """
    k0, k1, k2, L0 = P.k0, P.k1, P.k2, P.L0

    arg = p_red - k2 * t
    inner = 1.0 + k0 * k2 + k1 * arg
    K1 = math.sqrt(inner**2 - 4.0 * k0 * k1 * k2 * arg)
    K3 = K1**3
    K2 = math.sqrt(1.0 + k2**2)

    L = L0 * K2 * (1.0 - K1 + k0 * k2 + k1 * (p_red + k2 * t)) / (2.0 * k1 * k2)
    Lt = L0 * 0.5 * K2 * (1.0 + (1.0 - k0 * k2 + k1 * (p_red - k2 * t)) / K1)
    Lp = L0 * K2 * (K1 + k0 * k2 - k1 * p_red + k1 * k2 * t - 1.0) / (2.0 * k2 * K1)
    Ltt = -2.0 * L0 * K2 * k0 * k1 * k2**2 / K3
    Ltp = 2.0 * L0 * K2 * k0 * k1 * k2 / K3
    Lpp = -2.0 * L0 * K2 * k0 * k1 / K3

    return L, Lt, Lp, Ltt, Ltp, Lpp


# ═══════════════════════════════════════════════════════════════════════════
# 4. Equilibrium solver (Eq. 10) with flip trick
# ═══════════════════════════════════════════════════════════════════════════

def _findxe(L, omega):
    """
    Find the equilibrium tetrahedral fraction x_e by solving:
        L + ln(x/(1-x)) + omega*(1-2x) = 0

    Uses the L<0 flip trick from MATLAB: if L<0, negate L, solve
    (guaranteed small x), return 1-x. Bracketing from MATLAB's findxe.
    """
    flip = L < 0
    if flip:
        L = -L

    # Smart bracket selection (from MATLAB)
    if omega < 1.1111111 * (2.944439 - L):
        # xe = 0.05 isoline
        x0 = 0.049
        x1 = 0.5
    elif omega < 1.0204081 * (4.595119 - L):
        # xe = 0.01 isoline
        x0 = 0.0099
        x1 = 0.051
    else:
        x0 = 0.99 * math.exp(-1.0204081 * L - omega)
        x1 = 1.01 * 1.087 * math.exp(-L - omega)
        if x1 > 0.0101:
            x1 = 0.0101

    # Bisection solver
    def _f(x):
        if x <= 0 or x >= 1:
            return float('inf')
        return L + math.log(x / (1.0 - x)) + omega * (1.0 - 2.0 * x)

    # Ensure bracket is valid
    x0 = max(x0, 1e-15)
    x1 = min(x1, 1.0 - 1e-15)

    f0 = _f(x0)
    f1 = _f(x1)

    # If bracket doesn't straddle zero, try wider
    if f0 * f1 > 0:
        x0 = 1e-15
        x1 = 0.5
        f0 = _f(x0)
        f1 = _f(x1)

    if f0 * f1 > 0:
        # Fallback: Newton from 0.05
        xe = _newton_xe(L, omega, 0.05)
    else:
        # Bisection
        lo, hi = x0, x1
        flo = f0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            fm = _f(mid)
            if fm * flo < 0:
                hi = mid
            else:
                lo = mid
                flo = fm
            if hi - lo < 1e-14:
                break
        xe = (lo + hi) / 2.0

    if flip:
        xe = 1.0 - xe

    return xe


def _newton_xe(L, omega, x0, max_iter=200, tol=1e-13):
    """Newton solver for L + ln(x/(1-x)) + omega*(1-2x) = 0."""
    x = x0
    EPS = 1e-15
    for _ in range(max_iter):
        x = max(EPS, min(1.0 - EPS, x))
        lnrat = math.log(x / (1.0 - x))
        F = L + lnrat + omega * (1.0 - 2.0 * x)
        Fx = 1.0 / (x * (1.0 - x)) - 2.0 * omega
        if abs(Fx) < 1e-30:
            break
        dx = -F / Fx
        if x + dx < EPS:
            x = x / 2.0
        elif x + dx > 1.0 - EPS:
            x = (x + 1.0 - EPS) / 2.0
        else:
            x = x + dx
        if abs(F) < tol:
            break
    return x


# ═══════════════════════════════════════════════════════════════════════════
# 5. Property formulation using phi/chi (MATLAB lines 104–127)
# ═══════════════════════════════════════════════════════════════════════════

def _physical_props_holten(tau, t, p_red, x,
                           L, Lt, Lp, Ltt, Ltp, Lpp,
                           Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val,
                           T_K, B_val=0.0):
    """
    Compute physical properties using the phi/chi formulation.

    Uses order parameter f = 2x - 1 and susceptibility chi = 1/[2/(1-f^2) - omega].

    Returns dict with: rho, V, S, Cp, Cv, Kt, Ks, alpha, vel, G
    """
    omega = 2.0 + P.omega0 * p_red

    # Order parameter and susceptibility
    f = 2.0 * x - 1.0
    f2 = f * f
    if abs(1.0 - f2) > 1e-30:
        chi = 1.0 / (2.0 / (1.0 - f2) - omega)
    else:
        chi = 0.0

    # Dimensionless properties (MATLAB lines 109–117)
    EPS = 1e-300
    if x > EPS and x < 1.0 - EPS:
        g0 = x * L + x * math.log(x) + (1.0 - x) * math.log(1.0 - x) + omega * x * (1.0 - x)
    elif x <= EPS:
        g0 = 0.0
    else:
        g0 = L + omega * 0.0  # x=1 limit

    s = -0.5 * (f + 1.0) * Lt * tau - g0 - Bt_val
    v = 0.5 * tau * (P.omega0 / 2.0 * (1.0 - f2) + Lp * (f + 1.0)) + Bp_val
    kap = (1.0 / v) * (tau / 2.0 * (chi * (Lp - P.omega0 * f)**2
                                     - (f + 1.0) * Lpp) - Bpp_val)
    alp = (1.0 / v) * (Ltp / 2.0 * tau * (f + 1.0)
                        + (P.omega0 / 2.0 * (1.0 - f2) + Lp * (f + 1.0)) / 2.0
                        - tau * Lt / 2.0 * chi * (Lp - P.omega0 * f) + Btp_val)
    cp = tau * (-Lt * (f + 1.0) + tau * (Lt**2 * chi - Ltt * (f + 1.0)) / 2.0
                - Btt_val)

    # SI units (MATLAB lines 119–127)
    S_val = P.R * s                              # J/(kg·K)
    rho = P.rho0 / v                             # kg/m³
    Kap = kap / (P.rho0 * P.R * P.Tc)           # 1/Pa (isothermal compressibility)
    Alp = alp / P.Tc                             # 1/K (expansion coefficient)
    Cp_val = P.R * cp                            # J/(kg·K)
    Cv_val = Cp_val - T_K * Alp**2 / (rho * Kap)  # J/(kg·K)

    # Speed of sound: U = 1/sqrt(rho * kapS - T*alpP^2/Cp) [MATLAB line 127]
    # kapS = kapT - T*V*alpP^2/Cp  where V = 1/rho
    kap_S = Kap - T_K * (1.0 / rho) * Alp**2 / Cp_val  # 1/Pa
    vel = 1.0 / math.sqrt(rho * kap_S) if kap_S > 0 else float('nan')

    # Kt in MPa (bulk modulus = 1/kapT)
    Kt = 1.0 / Kap / 1e6 if Kap > 0 else float('inf')
    Ks = 1.0 / kap_S / 1e6 if kap_S > 0 else float('inf')
    V_spec = 1.0 / rho

    # Gibbs energy: g = B + tau*g0 (dimensionless), G = R*Tc*g [J/kg]
    g_red = B_val + tau * g0
    G_val = P.R * P.Tc * g_red

    return {
        'rho': rho, 'V': V_spec, 'S': S_val,
        'Cp': Cp_val, 'Cv': Cv_val, 'Kt': Kt, 'Ks': Ks,
        'alpha': Alp, 'vel': vel, 'G': G_val,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. State A and State B properties
# ═══════════════════════════════════════════════════════════════════════════

def _physical_props_stateA(tau, T_K, B_val, Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val):
    """
    State A properties (x=0, background only).
    f = -1, chi-terms vanish, L terms with (f+1) vanish.
    Accepts precomputed B values to avoid redundant computation.
    """
    v = Bp_val
    s = -Bt_val
    kap_dimless = -Bpp_val / v
    alp_dimless = Btp_val / v
    cp_dimless = -tau * Btt_val

    rho = P.rho0 / v
    V_spec = 1.0 / rho
    S_val = P.R * s
    Kap = kap_dimless / (P.rho0 * P.R * P.Tc)
    Alp = alp_dimless / P.Tc
    Cp_val = P.R * cp_dimless
    Cv_val = Cp_val - T_K * Alp**2 / (rho * Kap) if Kap > 0 else Cp_val

    kap_S = Kap - T_K * V_spec * Alp**2 / Cp_val if Cp_val > 0 else Kap
    vel = 1.0 / math.sqrt(rho * kap_S) if kap_S > 0 else float('nan')
    Kt = 1.0 / Kap / 1e6 if Kap > 0 else float('inf')
    Ks = 1.0 / kap_S / 1e6 if kap_S > 0 else float('inf')

    G_val = P.R * P.Tc * B_val

    return {
        'rho': rho, 'V': V_spec, 'S': S_val,
        'Cp': Cp_val, 'Cv': Cv_val, 'Kt': Kt, 'Ks': Ks,
        'alpha': Alp, 'vel': vel, 'G': G_val,
    }


def _physical_props_stateB(tau, T_K, B_val, Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val,
                            L, Lt_val, Lp_val, Ltt_val, Ltp_val, Lpp_val, p_red):
    """
    State B properties (x=1).
    f = +1, f+1 = 2, 1-f^2 = 0.  Chi terms vanish.
    Accepts precomputed B and L values to avoid redundant computation.
    """
    s = -Lt_val * tau - L - Bt_val
    v = tau * Lp_val + Bp_val
    kap_dimless = (1.0 / v) * (-tau * Lpp_val - Bpp_val)
    alp_dimless = (1.0 / v) * (Ltp_val * tau + Lp_val + Btp_val)
    cp = tau * (-2.0 * Lt_val - tau * Ltt_val - Btt_val)

    rho = P.rho0 / v
    V_spec = 1.0 / rho
    S_val = P.R * s
    Kap = kap_dimless / (P.rho0 * P.R * P.Tc)
    Alp = alp_dimless / P.Tc
    Cp_val = P.R * cp

    Cv_val = Cp_val - T_K * Alp**2 / (rho * Kap) if Kap > 0 else Cp_val
    kap_S = Kap - T_K * V_spec * Alp**2 / Cp_val if Cp_val > 0 else Kap
    vel = 1.0 / math.sqrt(rho * kap_S) if kap_S > 0 else float('nan')
    Kt = 1.0 / Kap / 1e6 if Kap > 0 else float('inf')
    Ks = 1.0 / kap_S / 1e6 if kap_S > 0 else float('inf')

    G_val = P.R * P.Tc * (B_val + tau * L)

    return {
        'rho': rho, 'V': V_spec, 'S': S_val,
        'Cp': Cp_val, 'Cv': Cv_val, 'Kt': Kt, 'Ks': Ks,
        'alpha': Alp, 'vel': vel, 'G': G_val,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def compute_properties(T_K, p_MPa, _compute_Kp=False):
    """
    Compute all thermodynamic properties at a single (T, p) point.

    Returns dict with:
      rho, V, S, G, Cp, Cv, Kt, Ks, alpha, vel, x, H, U, A, Kp
      rho_A, V_A, ..., vel_A, G_A, H_A, U_A, A_A, Kp_A  (state A)
      rho_B, V_B, ..., vel_B, G_B, H_B, U_B, A_B, Kp_B  (state B)
    """
    tau, pi, t, p_red = _reduce(T_K, p_MPa)

    # Background value and derivatives (single-pass scalar loop)
    B_val, Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val = _B_all(tau, pi)

    # Field L and derivatives
    L, Lt, Lp, Ltt, Ltp, Lpp = _compute_L(t, p_red)

    # Interaction parameter
    omega = 2.0 + P.omega0 * p_red

    # Equilibrium x
    x = _findxe(L, omega)

    # Mixture properties
    mix = _physical_props_holten(tau, t, p_red, x,
                                 L, Lt, Lp, Ltt, Ltp, Lpp,
                                 Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val,
                                 T_K, B_val)
    mix['x'] = x

    # State A (x=0) — reuse precomputed B values
    stateA = _physical_props_stateA(tau, T_K,
                                     B_val, Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val)

    # State B (x=1) — reuse precomputed B and L values
    stateB = _physical_props_stateB(tau, T_K,
                                     B_val, Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val,
                                     L, Lt, Lp, Ltt, Ltp, Lpp, p_red)

    # Assemble output
    result = {}
    for key, val in mix.items():
        result[key] = val
    for key, val in stateA.items():
        result[key + '_A'] = val
    for key, val in stateB.items():
        result[key + '_B'] = val

    # ── IAPWS-95 reference state alignment ────────────────────────────────
    for suffix in ['', '_A', '_B']:
        result['S' + suffix] += P.S_OFFSET
        result['G' + suffix] += P.H_OFFSET - T_K * P.S_OFFSET

    # ── Derived thermodynamic potentials (H, U, A) ─────────────────────
    p_Pa = p_MPa * 1e6
    for suffix in ['', '_A', '_B']:
        G = result['G' + suffix]
        S = result['S' + suffix]
        V = result['V' + suffix]
        result['H' + suffix] = G + T_K * S
        result['U' + suffix] = G + T_K * S - p_Pa * V
        result['A' + suffix] = G - p_Pa * V

    # Kp = dKt/dp via central difference (opt-in, expensive)
    if _compute_Kp:
        dp = 0.001  # MPa
        props_plus = compute_properties(T_K, p_MPa + dp)
        props_minus = compute_properties(T_K, p_MPa - dp)
        for suffix in ['', '_A', '_B']:
            result['Kp' + suffix] = (
                (props_plus['Kt' + suffix] - props_minus['Kt' + suffix])
                / (2.0 * dp)
            )

    return result


def compute_properties_at_x(T_K, p_MPa, x, _compute_Kp=False):
    """
    Compute properties at given (T, p) with forced x (not equilibrium).
    Useful for phase diagram routines.

    Returns dict with: rho, V, S, G, Cp, Cv, Kt, Ks, alpha, vel, x,
                       H, U, A, Kp
    """
    tau, pi, t, p_red = _reduce(T_K, p_MPa)

    B_val, Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val = _B_all(tau, pi)

    L, Lt, Lp, Ltt, Ltp, Lpp = _compute_L(t, p_red)

    props = _physical_props_holten(tau, t, p_red, x,
                                   L, Lt, Lp, Ltt, Ltp, Lpp,
                                   Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val,
                                   T_K, B_val)
    props['x'] = x

    # ── IAPWS-95 reference state alignment ────────────────────────────────
    props['S'] += P.S_OFFSET
    props['G'] += P.H_OFFSET - T_K * P.S_OFFSET

    # ── Derived thermodynamic potentials ────────────────────────────────
    p_Pa = p_MPa * 1e6
    G = props['G']
    S = props['S']
    V = props['V']
    props['H'] = G + T_K * S
    props['U'] = G + T_K * S - p_Pa * V
    props['A'] = G - p_Pa * V

    if _compute_Kp:
        dp = 0.001  # MPa
        pp = compute_properties_at_x(T_K, p_MPa + dp, x)
        pm = compute_properties_at_x(T_K, p_MPa - dp, x)
        props['Kp'] = (pp['Kt'] - pm['Kt']) / (2.0 * dp)

    return props


# ═══════════════════════════════════════════════════════════════════════════
# 8. Vectorized batch computation
# ═══════════════════════════════════════════════════════════════════════════

def _B_all_vec(tau, pi):
    """Vectorized background B and all derivatives over 20 terms.

    Parameters
    ----------
    tau : 1-D array — T/Tc
    pi  : 1-D array — (P - P0) / P_scale

    Returns
    -------
    (B, Bp, Bt, Bpp, Btp, Btt) — each a 1-D array
    """
    B = np.zeros_like(tau)
    Bp = np.zeros_like(tau)
    Bt = np.zeros_like(tau)
    Bpp = np.zeros_like(tau)
    Btp = np.zeros_like(tau)
    Btt = np.zeros_like(tau)
    inv_pi = 1.0 / pi
    inv_tau = 1.0 / tau
    for i in range(20):
        ci = P.c_bg[i]; ai = P.a_bg[i]; bi = P.b_bg[i]; di = P.d_bg[i]
        base = ci * (tau ** ai) * (pi ** bi) * np.exp(-di * pi)
        bdp = bi - di * pi
        B   += base
        Bp  += base * bdp * inv_pi
        Bt  += base * ai * inv_tau
        Bpp += base * (bdp * bdp - bi) * inv_pi * inv_pi
        Btp += base * ai * bdp * inv_tau * inv_pi
        Btt += base * ai * (ai - 1.0) * inv_tau * inv_tau
    return B, Bp, Bt, Bpp, Btp, Btt


def _compute_L_vec(t, p_red):
    """Vectorized hyperbolic field L and all derivatives.

    Parameters
    ----------
    t     : 1-D array — (T - Tc)/Tc
    p_red : 1-D array — (P - Pc) / P_scale

    Returns
    -------
    (L, Lt, Lp, Ltt, Ltp, Lpp) — each a 1-D array
    """
    k0, k1, k2, L0 = P.k0, P.k1, P.k2, P.L0

    arg = p_red - k2 * t
    inner = 1.0 + k0 * k2 + k1 * arg
    K1 = np.sqrt(inner**2 - 4.0 * k0 * k1 * k2 * arg)
    K3 = K1**3
    K2 = math.sqrt(1.0 + k2**2)  # scalar

    L   = L0 * K2 * (1.0 - K1 + k0 * k2 + k1 * (p_red + k2 * t)) / (2.0 * k1 * k2)
    Lt  = L0 * 0.5 * K2 * (1.0 + (1.0 - k0 * k2 + k1 * (p_red - k2 * t)) / K1)
    Lp  = L0 * K2 * (K1 + k0 * k2 - k1 * p_red + k1 * k2 * t - 1.0) / (2.0 * k2 * K1)
    Ltt = -2.0 * L0 * K2 * k0 * k1 * k2**2 / K3
    Ltp =  2.0 * L0 * K2 * k0 * k1 * k2 / K3
    Lpp = -2.0 * L0 * K2 * k0 * k1 / K3

    return L, Lt, Lp, Ltt, Ltp, Lpp


def _solve_xe_vec(L, omega):
    """Vectorized equilibrium x solver with flip trick.

    Solves L + ln(x/(1-x)) + omega*(1-2x) = 0 for each element.
    Uses the L<0 flip trick: work with |L| (small-x root), then flip back.

    Parameters
    ----------
    L     : 1-D array — field value
    omega : 1-D array — interaction parameter

    Returns
    -------
    x : 1-D array — equilibrium tetrahedral fraction
    """
    EPS = 1e-15
    n = len(L)
    flip = L < 0.0
    L_work = np.abs(L)

    def _newton_vec(x0_val):
        x = np.full(n, x0_val)
        for _ in range(100):
            x = np.clip(x, EPS, 1.0 - EPS)
            lnrat = np.log(x / (1.0 - x))
            F = L_work + lnrat + omega * (1.0 - 2.0 * x)
            Fx = 1.0 / (x * (1.0 - x)) - 2.0 * omega
            Fx_safe = np.where(np.abs(Fx) < 1e-30, 1e-30, Fx)
            dx = -F / Fx_safe
            x_new = x + dx
            x = np.where(x_new < EPS, x / 2.0,
                    np.where(x_new > 1.0 - EPS, (x + 1.0 - EPS) / 2.0,
                             x_new))
        return np.clip(x, EPS, 1.0 - EPS)

    x_lo = _newton_vec(0.05)
    x_hi = _newton_vec(0.5)

    def _g_vec(x):
        xc = np.clip(x, EPS, 1.0 - EPS)
        me = xc * np.log(xc) + (1.0 - xc) * np.log(1.0 - xc)
        return xc * L_work + me + omega * xc * (1.0 - xc)

    x = np.where(_g_vec(x_lo) <= _g_vec(x_hi), x_lo, x_hi)
    x = np.where(flip, 1.0 - x, x)
    return x


def compute_batch(T_K, p_MPa):
    """
    Vectorized computation of all thermodynamic properties.

    Parameters
    ----------
    T_K   : 1-D array — temperature in K
    p_MPa : 1-D array — pressure in MPa  (same length as T_K)

    Returns
    -------
    dict of 1-D arrays with keys:
        rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, alpha, vel, x,
        rho_A, V_A, ..., vel_A, G_A, H_A, U_A, A_A,
        rho_B, V_B, ..., vel_B, G_B, H_B, U_B, A_B
    (Kp, Kp_A, Kp_B are NOT included — too expensive for batch.)
    """
    T_K = np.asarray(T_K, dtype=float)
    p_MPa = np.asarray(p_MPa, dtype=float)

    # ── Reduced variables ────────────────────────────────────────────
    P_Pa = p_MPa * 1e6
    tau = T_K / P.Tc
    pi = (P_Pa - P.P0 * 1e6) / P.P_scale_Pa
    t = tau - 1.0
    p_red = (P_Pa - P.Pc * 1e6) / P.P_scale_Pa

    # ── Background B and derivatives ─────────────────────────────────
    B_val, Bp_val, Bt_val, Bpp_val, Btp_val, Btt_val = _B_all_vec(tau, pi)

    # ── Field L and derivatives ──────────────────────────────────────
    L, Lt, Lp, Ltt, Ltp, Lpp = _compute_L_vec(t, p_red)

    # ── Interaction parameter ────────────────────────────────────────
    omega = 2.0 + P.omega0 * p_red

    # ── Equilibrium x ────────────────────────────────────────────────
    x = _solve_xe_vec(L, omega)

    # ── Mixture properties (phi/chi formulation) ─────────────────────
    f = 2.0 * x - 1.0
    f2 = f * f
    chi = np.where(np.abs(1.0 - f2) > 1e-30,
                   1.0 / (2.0 / (1.0 - f2) - omega), 0.0)

    EPS_log = 1e-300
    x_c = np.clip(x, EPS_log, 1.0 - EPS_log)
    g0 = x_c * L + x_c * np.log(x_c) + (1.0 - x_c) * np.log(1.0 - x_c) + omega * x_c * (1.0 - x_c)

    s_mix = -0.5 * (f + 1.0) * Lt * tau - g0 - Bt_val
    v_mix = (0.5 * tau * (P.omega0 / 2.0 * (1.0 - f2) + Lp * (f + 1.0))
             + Bp_val)
    kap_mix = ((1.0 / v_mix)
               * (tau / 2.0 * (chi * (Lp - P.omega0 * f)**2
                               - (f + 1.0) * Lpp)
                  - Bpp_val))
    alp_mix = ((1.0 / v_mix)
               * (Ltp / 2.0 * tau * (f + 1.0)
                  + (P.omega0 / 2.0 * (1.0 - f2) + Lp * (f + 1.0)) / 2.0
                  - tau * Lt / 2.0 * chi * (Lp - P.omega0 * f)
                  + Btp_val))
    cp_mix = tau * (-Lt * (f + 1.0) + tau * (Lt**2 * chi - Ltt * (f + 1.0)) / 2.0
                    - Btt_val)

    # Mixture: reduced → SI
    S = P.R * s_mix
    rho = P.rho0 / v_mix
    V = 1.0 / rho
    Kap = kap_mix / (P.rho0 * P.R * P.Tc)
    Alp = alp_mix / P.Tc
    Cp = P.R * cp_mix
    Cv = np.where((Kap > 0) & np.isfinite(Kap),
                  Cp - T_K * Alp**2 / (rho * Kap), Cp)
    kap_S = np.where((Cp > 0) & np.isfinite(Cp),
                     Kap - T_K * V * Alp**2 / Cp, Kap)
    vel = np.where((rho > 0) & (kap_S > 0),
                   np.sqrt(np.maximum(1.0 / (rho * kap_S), 0.0)), np.nan)
    Kt = np.where((Kap > 0) & np.isfinite(Kap), 1.0 / Kap / 1e6, np.inf)
    Ks = np.where((kap_S > 0) & np.isfinite(kap_S), 1.0 / kap_S / 1e6, np.inf)
    g_red = B_val + tau * g0
    G = P.R * P.Tc * g_red

    # ── State A (x=0): background only ───────────────────────────────
    v_A = Bp_val
    s_A = -Bt_val
    kap_A = -Bpp_val / v_A
    alp_A = Btp_val / v_A
    cp_A = -tau * Btt_val

    rho_A = P.rho0 / v_A
    V_A = 1.0 / rho_A
    S_A = P.R * s_A
    Kap_A = kap_A / (P.rho0 * P.R * P.Tc)
    Alp_A = alp_A / P.Tc
    Cp_A = P.R * cp_A
    Cv_A = np.where(Kap_A > 0,
                    Cp_A - T_K * Alp_A**2 / (rho_A * Kap_A), Cp_A)
    kap_S_A = np.where(Cp_A > 0,
                       Kap_A - T_K * V_A * Alp_A**2 / Cp_A, Kap_A)
    vel_A = np.where((rho_A > 0) & (kap_S_A > 0),
                     np.sqrt(np.maximum(1.0 / (rho_A * kap_S_A), 0.0)), np.nan)
    Kt_A = np.where((Kap_A > 0) & np.isfinite(Kap_A), 1.0 / Kap_A / 1e6, np.inf)
    Ks_A = np.where((kap_S_A > 0) & np.isfinite(kap_S_A), 1.0 / kap_S_A / 1e6, np.inf)
    G_A = P.R * P.Tc * B_val

    # ── State B (x=1): f=+1, f+1=2, 1-f^2=0 ────────────────────────
    s_B = -Lt * tau - L - Bt_val
    v_B = tau * Lp + Bp_val
    kap_B = (1.0 / v_B) * (-tau * Lpp - Bpp_val)
    alp_B = (1.0 / v_B) * (Ltp * tau + Lp + Btp_val)
    cp_B = tau * (-2.0 * Lt - tau * Ltt - Btt_val)

    rho_B = P.rho0 / v_B
    V_B = 1.0 / rho_B
    S_B = P.R * s_B
    Kap_B = kap_B / (P.rho0 * P.R * P.Tc)
    Alp_B = alp_B / P.Tc
    Cp_B = P.R * cp_B
    Cv_B = np.where(Kap_B > 0,
                    Cp_B - T_K * Alp_B**2 / (rho_B * Kap_B), Cp_B)
    kap_S_B = np.where(Cp_B > 0,
                       Kap_B - T_K * V_B * Alp_B**2 / Cp_B, Kap_B)
    vel_B = np.where((rho_B > 0) & (kap_S_B > 0),
                     np.sqrt(np.maximum(1.0 / (rho_B * kap_S_B), 0.0)), np.nan)
    Kt_B = np.where((Kap_B > 0) & np.isfinite(Kap_B), 1.0 / Kap_B / 1e6, np.inf)
    Ks_B = np.where((kap_S_B > 0) & np.isfinite(kap_S_B), 1.0 / kap_S_B / 1e6, np.inf)
    G_B = P.R * P.Tc * (B_val + tau * L)

    # ── IAPWS-95 reference state alignment ───────────────────────────
    for S_arr, G_arr in [(S, G), (S_A, G_A), (S_B, G_B)]:
        S_arr += P.S_OFFSET
        G_arr += P.H_OFFSET - T_K * P.S_OFFSET

    # ── Derived thermodynamic potentials ─────────────────────────────
    p_Pa = p_MPa * 1e6
    H   = G   + T_K * S;    U   = H   - p_Pa * V;    A_pot   = G   - p_Pa * V
    H_A = G_A + T_K * S_A;  U_A = H_A - p_Pa * V_A;  A_pot_A = G_A - p_Pa * V_A
    H_B = G_B + T_K * S_B;  U_B = H_B - p_Pa * V_B;  A_pot_B = G_B - p_Pa * V_B

    # ── Assemble output dict ─────────────────────────────────────────
    result = {
        'rho': rho, 'V': V, 'S': S, 'G': G, 'H': H, 'U': U, 'A': A_pot,
        'Cp': Cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks, 'alpha': Alp, 'vel': vel,
        'x': x,
        'rho_A': rho_A, 'V_A': V_A, 'S_A': S_A, 'G_A': G_A, 'H_A': H_A,
        'U_A': U_A, 'A_A': A_pot_A, 'Cp_A': Cp_A, 'Cv_A': Cv_A,
        'Kt_A': Kt_A, 'Ks_A': Ks_A, 'alpha_A': Alp_A, 'vel_A': vel_A,
        'rho_B': rho_B, 'V_B': V_B, 'S_B': S_B, 'G_B': G_B, 'H_B': H_B,
        'U_B': U_B, 'A_B': A_pot_B, 'Cp_B': Cp_B, 'Cv_B': Cv_B,
        'Kt_B': Kt_B, 'Ks_B': Ks_B, 'alpha_B': Alp_B, 'vel_B': vel_B,
    }
    return result
