"""
EOS-VaT core engine: scalar evaluation of all thermodynamic properties.

All internal calculations use reduced variables (That, phat).
The main entry point is compute_properties(T_K, p_MPa) which returns
a dict of physical-unit properties for the mixture and both pure states.

Reference: M. Duška, "Water above the spinodal",
J. Chem. Phys. 152, 174501 (2020).
"""

import math
import numpy as np
from . import params as P


# ═══════════════════════════════════════════════════════════════════════════
# 1. Spinodal polynomials  (Eqs. 6a-c, 11-13)
# ═══════════════════════════════════════════════════════════════════════════

def spinodal_props(Th):
    """Evaluate spinodal pressure, volume, phi, and their T-derivatives."""
    dTh = Th - 1.0

    # Pressure at spinodal: pS = 1 + d1*(Th-1) + d2*(Th-1)^2 + d3*(Th-1)^3
    pS = 1.0 + P.d[0]*dTh + P.d[1]*dTh**2 + P.d[2]*dTh**3
    dpS = P.d[0] + 2*P.d[1]*dTh + 3*P.d[2]*dTh**2

    # Volume at spinodal: VS = c0 + c1*Th + c2*Th^2 + c3*Th^3 + c4*Th^4
    VS = P.c[0] + P.c[1]*Th + P.c[2]*Th**2 + P.c[3]*Th**3 + P.c[4]*Th**4
    dVS = P.c[1] + 2*P.c[2]*Th + 3*P.c[3]*Th**2 + 4*P.c[4]*Th**3

    # Second derivative of pressure wrt density at spinodal
    phi = P.b[0] + P.b[1]*Th + P.b[2]*Th**2 + P.b[3]*Th**3 + P.b[4]*Th**4
    dphi = P.b[1] + 2*P.b[2]*Th + 3*P.b[3]*Th**2 + 4*P.b[4]*Th**3

    return pS, VS, phi, dpS, dVS, dphi


# ═══════════════════════════════════════════════════════════════════════════
# 2. Auxiliary parameter B  (Eqs. 14-16)
# ═══════════════════════════════════════════════════════════════════════════

def compute_B(pS, VS, phi, dpS, dVS, dphi):
    """Compute B and dB/dTh from spinodal properties.

    B = sqrt(-phi / (2 * pS * VS^2))

    With corrected d coefficients, pS < 0 at moderate temperatures,
    so -phi/(2*pS*VS^2) > 0 and the sqrt is real.
    """
    arg = -phi / (2.0 * pS * VS**2)
    B = math.sqrt(abs(arg))  # abs for safety near sign transitions

    # dB/dTh via chain rule on -phi/(2*pS*VS^2)
    num = (-dphi * pS * VS**2
           + phi * dpS * VS**2
           + 2.0 * phi * pS * VS * dVS)
    denom = 2.0 * pS**2 * VS**4
    darg = num / denom
    dB = darg / (2.0 * B) if B > 0 else 0.0

    return B, dB


# ═══════════════════════════════════════════════════════════════════════════
# 3. Volume of state A  (Eqs. 17-18)
# ═══════════════════════════════════════════════════════════════════════════

def compute_VA(ph, pS, VS, B):
    """Compute reduced volume of state A and its pressure derivative."""
    u = math.sqrt(max(1.0 - ph / pS, 0.0))
    denom = u + B

    VA = VS * B / denom

    # dVA/dph
    if u > 1e-30:
        dVA_dp = VS * B / (2.0 * pS * denom**2 * u)
    else:
        dVA_dp = float('inf')

    return VA, dVA_dp, u


# ═══════════════════════════════════════════════════════════════════════════
# 4. Entropy of state A  (Eqs. 19-26)
# ═══════════════════════════════════════════════════════════════════════════

def compute_SS(Th):
    """Entropy at the spinodal (Eq. 25): integral of SS'(Th)."""
    return (P.s[0] * math.log(Th)
            + P.s[1] * Th
            + P.s[2] / 2.0 * Th**2
            + P.s[3] / 3.0 * Th**3)


def compute_SA(VA, VS, Th, dpS, B, dB, pS, dVS, dphi):
    """Compute reduced entropy of state A (Eq. 20)."""
    rhoS = 1.0 / VS

    # Auxiliary coefficients A and C (Eqs. 21-22, corrected: use pS not rhoS)
    A_coeff = dpS * B**2 + 2.0 * pS * B * dB
    C_coeff = 2.0 * pS * B**2 * dVS

    SS = compute_SS(Th)

    r = VA / VS  # ratio > 1 for stable liquid
    ln_r = math.log(r) if r > 0 else 0.0

    SA = (dpS * (VA - VS)
          + A_coeff * (VS**2 / VA + 2.0 * VS * ln_r - VA)
          + C_coeff * (VS / VA + ln_r - 1.0)
          + SS)
    return SA


# ═══════════════════════════════════════════════════════════════════════════
# 5. Gibbs free energy difference G^B - G^A  (Eqs. 27-32)
# ═══════════════════════════════════════════════════════════════════════════

def compute_DeltaG(ph, Th):
    """Compute DeltaG and all its partial derivatives."""
    DG = (P.a[0] + P.a[1]*ph*Th + P.a[2]*ph + P.a[3]*Th
          + P.a[4]*Th**2 + P.a[5]*ph**2 + P.a[6]*ph**3)

    dDG_dp = P.a[1]*Th + P.a[2] + 2*P.a[5]*ph + 3*P.a[6]*ph**2
    dDG_dT = P.a[1]*ph + P.a[3] + 2*P.a[4]*Th

    d2DG_dp2 = 2*P.a[5] + 6*P.a[6]*ph
    d2DG_dT2 = 2*P.a[4]
    d2DG_dpT = P.a[1]

    return DG, dDG_dp, dDG_dT, d2DG_dp2, d2DG_dT2, d2DG_dpT


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cooperativity  (Eqs. 33-35)
# ═══════════════════════════════════════════════════════════════════════════

def compute_omega(ph, Th):
    """Compute omega and its partial derivatives."""
    om = P.w[0] * (1.0 + P.w[1]*ph + P.w[2]*Th + P.w[3]*Th*ph)
    dom_dp = P.w[0] * (P.w[1] + P.w[3]*Th)
    dom_dT = P.w[0] * (P.w[2] + P.w[3]*ph)
    return om, dom_dp, dom_dT


# ═══════════════════════════════════════════════════════════════════════════
# 7. Equilibrium solver  (Eqs. 7-9)
# ═══════════════════════════════════════════════════════════════════════════

def solve_x(DG, Th, om, max_iter=200, tol=1e-12):
    """Solve for equilibrium fraction x via Newton-Raphson."""
    x = 0.1  # initial guess

    EPS = 1e-15
    for _ in range(max_iter):
        x = max(EPS, min(1.0 - EPS, x))
        lnrat = math.log(x / (1.0 - x))

        F = DG + Th * lnrat + om * (1.0 - 2.0 * x)
        Fx = Th / (x * (1.0 - x)) - 2.0 * om

        if abs(Fx) < 1e-30:
            break
        dx = -F / Fx
        # Damped step to stay in bounds
        if x + dx < EPS:
            x = x / 2.0
        elif x + dx > 1.0 - EPS:
            x = (x + 1.0 - EPS) / 2.0
        else:
            x = x + dx

        if abs(F) < tol:
            break

    return x


def _solve_x_stable(DG, Th, om, ph):
    """
    Solve for the globally stable equilibrium x.

    In the 3-root region (below the upper spinodal), there are three
    equilibrium compositions. This function finds all three and returns
    the one with the lowest Gibbs free energy of mixing.

    Uses both bracket search (via inflection points) and Newton from
    multiple starting points to robustly find all roots, then picks
    the one with lowest g.
    """
    EPS = 1e-12

    def _F(x):
        return DG + Th * math.log(x / (1.0 - x)) + om * (1.0 - 2.0 * x)

    def _g(x):
        mix_ent = x * math.log(x) + (1.0 - x) * math.log(1.0 - x) if EPS < x < 1.0 - EPS else 0.0
        return x * DG + Th * mix_ent + om * x * (1.0 - x)

    def _newton(x0, max_iter=200, tol=1e-12):
        x = x0
        for _ in range(max_iter):
            x = max(EPS, min(1.0 - EPS, x))
            lnrat = math.log(x / (1.0 - x))
            F = DG + Th * lnrat + om * (1.0 - 2.0 * x)
            Fx = Th / (x * (1.0 - x)) - 2.0 * om
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

    # Collect candidate roots from multiple approaches
    candidates = []

    # 1. Newton from low-x and high-x starting points
    for x0 in (0.05, 0.5, 0.95):
        xr = _newton(x0)
        if EPS < xr < 1.0 - EPS and abs(_F(xr)) < 1e-8:
            candidates.append(xr)

    # 2. Bracket search via inflection points (if they exist)
    if om > 0:
        disc = 1.0 - 2.0 * Th / om
        if disc > 0:
            sqrt_disc = math.sqrt(disc)
            x_infl_lo = (1.0 - sqrt_disc) / 2.0
            x_infl_hi = (1.0 + sqrt_disc) / 2.0
            intervals = [(EPS, x_infl_lo), (x_infl_lo, x_infl_hi),
                         (x_infl_hi, 1.0 - EPS)]
            for a, b in intervals:
                try:
                    fa, fb = _F(a), _F(b)
                except (ValueError, ZeroDivisionError):
                    continue
                if fa * fb < 0:
                    lo, hi = a, b
                    flo = fa
                    for _ in range(80):
                        mid = (lo + hi) / 2.0
                        fm = _F(mid)
                        if fm * flo < 0:
                            hi = mid
                        else:
                            lo = mid
                            flo = fm
                    candidates.append((lo + hi) / 2.0)

    if not candidates:
        return solve_x(DG, Th, om)

    # Deduplicate (roots within 1e-6 of each other)
    unique = [candidates[0]]
    for c in candidates[1:]:
        if all(abs(c - u) > 1e-6 for u in unique):
            unique.append(c)

    # Pick the root with the lowest g(x)
    best_x = unique[0]
    best_g = _g(unique[0])
    for r in unique[1:]:
        gr = _g(r)
        if gr < best_g:
            best_g = gr
            best_x = r
    return best_x


# ═══════════════════════════════════════════════════════════════════════════
# 8. Finite-difference helpers for temperature derivatives of state A
# ═══════════════════════════════════════════════════════════════════════════

def _state_A_at(ph, Th):
    """Evaluate VA and SA at a given (ph, Th). Returns (VA, SA)."""
    pS, VS, phi, dpS, dVS, dphi = spinodal_props(Th)
    B, dB = compute_B(pS, VS, phi, dpS, dVS, dphi)
    VA, _, _ = compute_VA(ph, pS, VS, B)
    SA = compute_SA(VA, VS, Th, dpS, B, dB, pS, dVS, dphi)
    return VA, SA


def _dVA_dTh(ph, Th, delta=1e-7):
    """Central finite difference for dVA/dTh at constant ph."""
    VA_p, _ = _state_A_at(ph, Th + delta)
    VA_m, _ = _state_A_at(ph, Th - delta)
    return (VA_p - VA_m) / (2.0 * delta)


def _dSA_dTh(ph, Th, delta=1e-7):
    """Central finite difference for dSA/dTh at constant ph."""
    _, SA_p = _state_A_at(ph, Th + delta)
    _, SA_m = _state_A_at(ph, Th - delta)
    return (SA_p - SA_m) / (2.0 * delta)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Reduced Gibbs energy of state A via numerical integration
# ═══════════════════════════════════════════════════════════════════════════

# 16-point Gauss-Legendre nodes and weights (precomputed)
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _compute_g_A_integral(ph, Th):
    """
    Compute the reduced Gibbs energy of state A via 2D path integral.

    The path goes from (ph=0, Th=1) [the VLCP] to (ph, Th) in two legs:
      Leg 1 (T-leg): (0, 1) -> (0, Th), integrating -SA(0, Th') dTh'
      Leg 2 (P-leg): (0, Th) -> (ph, Th), integrating VA(ph', Th) dph'

    g_A(ph, Th) = integral_1^{Th} (-SA(0, Th')) dTh'
                + integral_0^{ph} VA(ph', Th) dph'

    The reference g_A(0, 1) = 0 (arbitrary constant at the VLCP).
    Uses 16-point Gauss-Legendre quadrature for both legs.
    """
    # -- Leg 1: T-integral from Th=1 to Th at ph=0 -------------------------
    g_T_leg = 0.0
    dTh_val = Th - 1.0
    if abs(dTh_val) > 1e-30:
        half_dTh = dTh_val / 2.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
            Th_i = half_dTh * (node + 1.0) + 1.0
            pS_i, VS_i, phi_i, dpS_i, dVS_i, dphi_i = spinodal_props(Th_i)
            B_i, dB_i = compute_B(pS_i, VS_i, phi_i, dpS_i, dVS_i, dphi_i)
            VA_i, _, _ = compute_VA(0.0, pS_i, VS_i, B_i)
            SA_i = compute_SA(VA_i, VS_i, Th_i, dpS_i, B_i, dB_i,
                              pS_i, dVS_i, dphi_i)
            g_T_leg += weight * (-SA_i)
        g_T_leg *= half_dTh

    # -- Leg 2: P-integral from ph=0 to ph at fixed Th ---------------------
    g_P_leg = 0.0
    if abs(ph) > 1e-30:
        pS, VS, phi, dpS, dVS, dphi = spinodal_props(Th)
        B, dB = compute_B(pS, VS, phi, dpS, dVS, dphi)
        half_ph = ph / 2.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
            ph_i = half_ph * (node + 1.0)
            VA_i, _, _ = compute_VA(ph_i, pS, VS, B)
            g_P_leg += weight * VA_i
        g_P_leg *= half_ph

    return g_T_leg + g_P_leg


# ═══════════════════════════════════════════════════════════════════════════
# 10. Pure-state property calculator
# ═══════════════════════════════════════════════════════════════════════════

def _pure_state_props(Vh, Sh, dVh_dp, dVh_dTh, dSh_dTh, Th, ph, g_red=0.0):
    """
    From reduced V, S and their derivatives, compute physical properties
    for a single pure state.

    g_red : reduced Gibbs energy (dimensionless)

    Returns dict with: rho, V, S, G, Cp, Cv, Kt, Ks, alpha, vel
    """
    T = Th * P.T_VLCP
    p = ph * P.p_VLCP  # MPa

    V_phys = Vh * P.R_specific * P.T_VLCP / (P.p_VLCP * 1e6)  # m³/kg
    rho = 1.0 / V_phys if V_phys > 0 else float('inf')
    S_phys = Sh * P.R_specific  # J/(kg·K)

    # Cp = R * Th * (dSh/dTh)_ph  (reduced -> physical)
    Cp = P.R_specific * Th * dSh_dTh

    # kappa_T = -(1/Vh)*(dVh/dph) / p_VLCP   [Pa^-1]
    # But we want Kt = 1/kappa_T in MPa
    dVh_dp_reduced = dVh_dp  # this is dVh/dph (reduced)
    if abs(dVh_dp_reduced) > 1e-30:
        kappa_T_reduced = -(1.0 / Vh) * dVh_dp_reduced  # dimensionless
        kappa_T = kappa_T_reduced / (P.p_VLCP * 1e6)     # Pa^-1
        Kt = 1.0 / kappa_T / 1e6                         # MPa
    else:
        kappa_T = float('inf')
        Kt = 0.0

    # alpha = (1/Vh)*(dVh/dTh) / T_VLCP   [K^-1]
    dVh_dTh_reduced = dVh_dTh  # dVh/dTh (reduced)
    alpha = (1.0 / Vh) * dVh_dTh_reduced / P.T_VLCP if Vh > 0 else 0.0

    # Cv = Cp - T * V * alpha^2 / kappa_T
    if kappa_T > 0 and kappa_T != float('inf'):
        Cv = Cp - T * V_phys * alpha**2 / kappa_T
    else:
        Cv = Cp

    # Speed of sound: w = sqrt(1 / (rho * kappa_S))
    # kappa_S = kappa_T - T * V * alpha^2 / Cp
    if Cp > 0:
        kappa_S = kappa_T - T * V_phys * alpha**2 / Cp
    else:
        kappa_S = kappa_T

    Ks = 1.0 / kappa_S / 1e6 if kappa_S > 0 else float('inf')  # MPa

    if rho > 0 and kappa_S > 0:
        vel = math.sqrt(1.0 / (rho * kappa_S))
    else:
        vel = float('nan')

    # Gibbs energy: G = R_specific * T_VLCP * g_red [J/kg]
    G_val = P.R_specific * P.T_VLCP * g_red

    return {
        'rho': rho,
        'V': V_phys,
        'S': S_phys,
        'G': G_val,
        'Cp': Cp,
        'Cv': Cv,
        'Kt': Kt,
        'Ks': Ks,
        'alpha': alpha,
        'vel': vel,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 11. Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def compute_properties(T_K, p_MPa, _compute_Kp=False):
    """
    Compute all thermodynamic properties at a single (T, p) point.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.
    p_MPa : float
        Pressure in MPa.

    Returns
    -------
    dict : thermodynamic properties in physical units for the mixture
           and for pure states A and B.
    """
    # ── Reduce inputs ────────────────────────────────────────────────────
    Th = T_K / P.T_VLCP
    ph = p_MPa / P.p_VLCP

    # ── Spinodal polynomials ─────────────────────────────────────────────
    pS, VS, phi, dpS, dVS, dphi = spinodal_props(Th)

    # ── B parameter ──────────────────────────────────────────────────────
    B, dB = compute_B(pS, VS, phi, dpS, dVS, dphi)

    # ── State A volume ───────────────────────────────────────────────────
    VA, dVA_dp, u = compute_VA(ph, pS, VS, B)

    # ── State A entropy ──────────────────────────────────────────────────
    SA = compute_SA(VA, VS, Th, dpS, B, dB, pS, dVS, dphi)

    # ── DeltaG and omega ─────────────────────────────────────────────────
    DG, dDG_dp, dDG_dT, d2DG_dp2, d2DG_dT2, d2DG_dpT = compute_DeltaG(ph, Th)
    om, dom_dp, dom_dT = compute_omega(ph, Th)

    # ── Solve for x (globally stable equilibrium) ───────────────────────
    x = _solve_x_stable(DG, Th, om, ph)

    # ── Total reduced volume (Eq. 36) ────────────────────────────────────
    DeltaV = dDG_dp
    Vh = VA + x * DeltaV + dom_dp * x * (1.0 - x)

    # ── Total reduced entropy (Eq. 37) ───────────────────────────────────
    mix_entropy = x * math.log(x) + (1.0 - x) * math.log(1.0 - x) if 0 < x < 1 else 0.0
    Sh = SA - x * dDG_dT - mix_entropy - dom_dT * x * (1.0 - x)

    # ── dx/dTh and dx/dph (Eqs. 38-40) ──────────────────────────────────
    Fx = Th / (x * (1.0 - x)) - 2.0 * om
    lnrat = math.log(x / (1.0 - x)) if 0 < x < 1 else 0.0

    F_Th = dDG_dT + lnrat + dom_dT * (1.0 - 2.0 * x)
    F_ph = DeltaV + dom_dp * (1.0 - 2.0 * x)

    dx_dTh = -F_Th / Fx if abs(Fx) > 1e-30 else 0.0
    dx_dph = -F_ph / Fx if abs(Fx) > 1e-30 else 0.0

    # ── dVh/dph at fixed x (Eq. 44) ─────────────────────────────────────
    dVh_dp_x = dVA_dp + x * d2DG_dp2
    dVhdx = DeltaV + dom_dp * (1.0 - 2.0 * x)
    dVh_dp = dVh_dp_x + dVhdx * dx_dph

    # ── dVA/dTh analytical (eliminates finite-difference calls) ──────────
    denom = u + B
    if u > 1e-30:
        du_dTh = ph * dpS / (2.0 * pS**2 * u)
        dN_dTh = dVS * B + VS * dB     # d(VS*B)/dTh
        dD_dTh = du_dTh + dB            # d(u+B)/dTh
        dVA_dTh_val = (dN_dTh * denom - VS * B * dD_dTh) / denom**2
    else:
        dVA_dTh_val = 0.0

    # ── dVh/dTh at fixed x (Eq. 47) ─────────────────────────────────────
    dVh_dT_x = dVA_dTh_val + x * d2DG_dpT + P.w[0] * P.w[3] * x * (1.0 - x)
    dVh_dT = dVh_dT_x + dVhdx * dx_dTh

    # ── dSA/dTh via finite difference (computed once, reused below) ──────
    delta = 1e-7
    _, SA_p = _state_A_at(ph, Th + delta)
    _, SA_m = _state_A_at(ph, Th - delta)
    dSA_dTh_val = (SA_p - SA_m) / (2.0 * delta)

    # ── dSh/dTh at fixed x (Eq. 42) ─────────────────────────────────────
    dSh_dT_x = dSA_dTh_val - x * d2DG_dT2  # d2omega/dT2 = 0
    dShdx = -dDG_dT + math.log((1.0 - x) / x) - dom_dT * (1.0 - 2.0 * x) if 0 < x < 1 else 0.0
    dSh_dT = dSh_dT_x + dShdx * dx_dTh

    # ── Reduced Gibbs energies for G computation ─────────────────────────
    g_A = _compute_g_A_integral(ph, Th)
    g_mix = g_A + x * DG + Th * mix_entropy + om * x * (1.0 - x)
    g_B = g_A + DG

    # ── Mixture physical properties ──────────────────────────────────────
    mix = _pure_state_props(Vh, Sh, dVh_dp, dVh_dT, dSh_dT, Th, ph, g_mix)
    mix['x'] = x

    # ── State A properties (reuse dSA_dTh_val — same (ph, Th)) ──────────
    stateA = _pure_state_props(VA, SA, dVA_dp, dVA_dTh_val, dSA_dTh_val, Th, ph, g_A)

    # ── State B properties (analytical temperature derivatives) ──────────
    VB = VA + DeltaV
    SB = SA - dDG_dT
    dVB_dp = dVA_dp + d2DG_dp2
    dVB_dTh = dVA_dTh_val + d2DG_dpT
    dSB_dTh = dSA_dTh_val - d2DG_dT2

    stateB = _pure_state_props(VB, SB, dVB_dp, dVB_dTh, dSB_dTh, Th, ph, g_B)

    # ── Assemble output ──────────────────────────────────────────────────
    result = {}

    # Mixture
    for key, val in mix.items():
        result[key] = val

    # State A (suffix _A)
    for key, val in stateA.items():
        result[key + '_A'] = val

    # State B (suffix _B)
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
    Compute thermodynamic properties at a given (T, p) with a forced x value
    (instead of solving for equilibrium x). Useful for evaluating properties
    along binodal/spinodal branches at known coexisting compositions.

    Returns dict with: rho, V, S, G, Cp, Cv, Kt, Ks, alpha, vel, x,
                       H, U, A, Kp
    """
    Th = T_K / P.T_VLCP
    ph = p_MPa / P.p_VLCP

    pS, VS, phi, dpS, dVS, dphi = spinodal_props(Th)
    B, dB = compute_B(pS, VS, phi, dpS, dVS, dphi)
    VA, dVA_dp, u = compute_VA(ph, pS, VS, B)
    SA = compute_SA(VA, VS, Th, dpS, B, dB, pS, dVS, dphi)

    DG, dDG_dp, dDG_dT, d2DG_dp2, d2DG_dT2, d2DG_dpT = compute_DeltaG(ph, Th)
    om, dom_dp, dom_dT = compute_omega(ph, Th)

    # Total reduced volume and entropy at forced x
    Vh = VA + x * dDG_dp + dom_dp * x * (1.0 - x)
    mix_entropy = x * math.log(x) + (1.0 - x) * math.log(1.0 - x) if 0 < x < 1 else 0.0
    Sh = SA - x * dDG_dT - mix_entropy - dom_dT * x * (1.0 - x)

    # dx/dTh and dx/dph (same formulas, but using the forced x)
    Fx = Th / (x * (1.0 - x)) - 2.0 * om if 0 < x < 1 else 1e30
    lnrat = math.log(x / (1.0 - x)) if 0 < x < 1 else 0.0
    F_Th = dDG_dT + lnrat + dom_dT * (1.0 - 2.0 * x)
    F_ph = dDG_dp + dom_dp * (1.0 - 2.0 * x)
    dx_dTh = -F_Th / Fx if abs(Fx) > 1e-30 else 0.0
    dx_dph = -F_ph / Fx if abs(Fx) > 1e-30 else 0.0

    # dVh/dph
    dVh_dp_x = dVA_dp + x * d2DG_dp2
    dVhdx = dDG_dp + dom_dp * (1.0 - 2.0 * x)
    dVh_dp = dVh_dp_x + dVhdx * dx_dph

    # dVA/dTh analytical
    denom = u + B
    if u > 1e-30:
        du_dTh = ph * dpS / (2.0 * pS**2 * u)
        dN_dTh = dVS * B + VS * dB
        dD_dTh = du_dTh + dB
        dVA_dTh_val = (dN_dTh * denom - VS * B * dD_dTh) / denom**2
    else:
        dVA_dTh_val = 0.0

    # dVh/dTh
    dVh_dT_x = dVA_dTh_val + x * d2DG_dpT + P.w[0] * P.w[3] * x * (1.0 - x)
    dVh_dT = dVh_dT_x + dVhdx * dx_dTh

    # dSA/dTh via finite difference
    delta = 1e-7
    _, SA_p = _state_A_at(ph, Th + delta)
    _, SA_m = _state_A_at(ph, Th - delta)
    dSA_dTh_val = (SA_p - SA_m) / (2.0 * delta)

    # dSh/dTh
    dSh_dT_x = dSA_dTh_val - x * d2DG_dT2
    dShdx = -dDG_dT + math.log((1.0 - x) / x) - dom_dT * (1.0 - 2.0 * x) if 0 < x < 1 else 0.0
    dSh_dT = dSh_dT_x + dShdx * dx_dTh

    # Reduced Gibbs energy at forced x
    g_A = _compute_g_A_integral(ph, Th)
    g_mix = g_A + x * DG + Th * mix_entropy + om * x * (1.0 - x)

    props = _pure_state_props(Vh, Sh, dVh_dp, dVh_dT, dSh_dT, Th, ph, g_mix)
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
# 12. Vectorized batch computation
# ═══════════════════════════════════════════════════════════════════════════

def _solve_x_vec_duska(DG, Th, om):
    """
    Vectorized Newton solver for the Duska equilibrium x.

    Solves F(x) = DG + Th * ln(x/(1-x)) + om * (1 - 2x) = 0
    from two starting points (0.05 and 0.95) and picks the root
    with lower Gibbs energy of mixing for each point.

    Parameters
    ----------
    DG : 1-D array — G^B - G^A Gibbs difference (reduced)
    Th : 1-D array — T_hat = T / T_VLCP
    om : 1-D array — cooperativity parameter omega

    Returns
    -------
    x : 1-D array — equilibrium tetrahedral fraction
    """
    EPS = 1e-15
    n = len(DG)

    def _newton_vec(x0_val):
        x = np.full(n, x0_val)
        for _ in range(50):
            x = np.clip(x, EPS, 1.0 - EPS)
            lnrat = np.log(x / (1.0 - x))
            F = DG + Th * lnrat + om * (1.0 - 2.0 * x)
            Fx = Th / (x * (1.0 - x)) - 2.0 * om
            Fx_safe = np.where(np.abs(Fx) < 1e-30, 1e-30, Fx)
            dx = -F / Fx_safe
            x_new = x + dx
            x = np.where(x_new < EPS, x / 2.0,
                    np.where(x_new > 1.0 - EPS, (x + 1.0 - EPS) / 2.0,
                             x_new))
        return np.clip(x, EPS, 1.0 - EPS)

    x_lo = _newton_vec(0.05)
    x_hi = _newton_vec(0.95)

    def _g_vec(x):
        xc = np.clip(x, EPS, 1.0 - EPS)
        me = xc * np.log(xc) + (1.0 - xc) * np.log(1.0 - xc)
        return xc * DG + Th * me + om * xc * (1.0 - xc)

    return np.where(_g_vec(x_lo) <= _g_vec(x_hi), x_lo, x_hi)


def _SA_vec(ph, Th):
    """
    Vectorized evaluation of state-A entropy SA at arbitrary (ph, Th) arrays.

    Used for finite-difference dSA/dTh. Returns 1-D array of SA values.
    """
    dTh = Th - 1.0

    # Spinodal polynomials
    pS = 1.0 + P.d[0]*dTh + P.d[1]*dTh**2 + P.d[2]*dTh**3
    dpS = P.d[0] + 2*P.d[1]*dTh + 3*P.d[2]*dTh**2
    VS = P.c[0] + P.c[1]*Th + P.c[2]*Th**2 + P.c[3]*Th**3 + P.c[4]*Th**4
    dVS = P.c[1] + 2*P.c[2]*Th + 3*P.c[3]*Th**2 + 4*P.c[4]*Th**3
    phi = P.b[0] + P.b[1]*Th + P.b[2]*Th**2 + P.b[3]*Th**3 + P.b[4]*Th**4
    dphi = P.b[1] + 2*P.b[2]*Th + 3*P.b[3]*Th**2 + 4*P.b[4]*Th**3

    # B parameter
    arg = -phi / (2.0 * pS * VS**2)
    B = np.sqrt(np.abs(arg))
    num_B = (-dphi * pS * VS**2 + phi * dpS * VS**2
             + 2.0 * phi * pS * VS * dVS)
    denom_B = 2.0 * pS**2 * VS**4
    darg = num_B / denom_B
    dB = np.where(B > 0, darg / (2.0 * B), 0.0)

    # State A volume
    u = np.sqrt(np.maximum(1.0 - ph / pS, 0.0))
    denom_VA = u + B
    VA = VS * B / denom_VA

    # Entropy coefficients
    A_coeff = dpS * B**2 + 2.0 * pS * B * dB
    C_coeff = 2.0 * pS * B**2 * dVS
    SS = (P.s[0] * np.log(Th) + P.s[1]*Th
          + P.s[2]/2.0*Th**2 + P.s[3]/3.0*Th**3)
    r = VA / VS
    ln_r = np.log(np.maximum(r, 1e-300))
    SA = (dpS * (VA - VS)
          + A_coeff * (VS**2/VA + 2.0*VS*ln_r - VA)
          + C_coeff * (VS/VA + ln_r - 1.0) + SS)
    return SA


def compute_batch(T_K, p_MPa):
    """
    Vectorized computation of all thermodynamic properties.

    Parameters
    ----------
    T_K   : 1-D array — temperature in K
    p_MPa : 1-D array — pressure in MPa

    Returns
    -------
    dict of 1-D arrays with keys:
        rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, alpha, vel, x,
        rho_A, V_A, S_A, G_A, H_A, U_A, A_A, Cp_A, Cv_A, Kt_A, Ks_A, alpha_A, vel_A,
        rho_B, V_B, S_B, G_B, H_B, U_B, A_B, Cp_B, Cv_B, Kt_B, Ks_B, alpha_B, vel_B
    """
    T_K = np.asarray(T_K, dtype=float)
    p_MPa = np.asarray(p_MPa, dtype=float)
    n = len(T_K)
    EPS = 1e-15

    # ── Reduced variables ───────────────────────────────────────────────
    Th = T_K / P.T_VLCP
    ph = p_MPa / P.p_VLCP
    dTh = Th - 1.0

    # ── (a) Spinodal polynomials ────────────────────────────────────────
    pS = 1.0 + P.d[0]*dTh + P.d[1]*dTh**2 + P.d[2]*dTh**3
    dpS = P.d[0] + 2*P.d[1]*dTh + 3*P.d[2]*dTh**2
    VS = P.c[0] + P.c[1]*Th + P.c[2]*Th**2 + P.c[3]*Th**3 + P.c[4]*Th**4
    dVS = P.c[1] + 2*P.c[2]*Th + 3*P.c[3]*Th**2 + 4*P.c[4]*Th**3
    phi = P.b[0] + P.b[1]*Th + P.b[2]*Th**2 + P.b[3]*Th**3 + P.b[4]*Th**4
    dphi = P.b[1] + 2*P.b[2]*Th + 3*P.b[3]*Th**2 + 4*P.b[4]*Th**3

    # ── (b) B parameter ────────────────────────────────────────────────
    arg = -phi / (2.0 * pS * VS**2)
    B = np.sqrt(np.abs(arg))
    num_B = (-dphi * pS * VS**2 + phi * dpS * VS**2
             + 2.0 * phi * pS * VS * dVS)
    denom_B = 2.0 * pS**2 * VS**4
    darg = num_B / denom_B
    dB = np.where(B > 0, darg / (2.0 * B), 0.0)

    # ── (c) State A volume ──────────────────────────────────────────────
    u = np.sqrt(np.maximum(1.0 - ph / pS, 0.0))
    denom_VA = u + B
    VA = VS * B / denom_VA
    dVA_dp = np.where(u > 1e-30,
                      VS * B / (2.0 * pS * denom_VA**2 * u), np.inf)

    # ── (d) State A entropy ─────────────────────────────────────────────
    A_coeff = dpS * B**2 + 2.0 * pS * B * dB
    C_coeff = 2.0 * pS * B**2 * dVS
    SS = (P.s[0] * np.log(Th) + P.s[1]*Th
          + P.s[2]/2.0*Th**2 + P.s[3]/3.0*Th**3)
    r = VA / VS
    ln_r = np.log(np.maximum(r, 1e-300))
    SA = (dpS * (VA - VS)
          + A_coeff * (VS**2/VA + 2.0*VS*ln_r - VA)
          + C_coeff * (VS/VA + ln_r - 1.0) + SS)

    # ── (e) DeltaG and omega ────────────────────────────────────────────
    DG = (P.a[0] + P.a[1]*ph*Th + P.a[2]*ph + P.a[3]*Th
          + P.a[4]*Th**2 + P.a[5]*ph**2 + P.a[6]*ph**3)
    dDG_dp = P.a[1]*Th + P.a[2] + 2*P.a[5]*ph + 3*P.a[6]*ph**2
    dDG_dT = P.a[1]*ph + P.a[3] + 2*P.a[4]*Th
    d2DG_dp2 = 2*P.a[5] + 6*P.a[6]*ph
    d2DG_dT2 = np.full(n, 2*P.a[4])
    d2DG_dpT = np.full(n, P.a[1])

    om = P.w[0] * (1.0 + P.w[1]*ph + P.w[2]*Th + P.w[3]*Th*ph)
    dom_dp = P.w[0] * (P.w[1] + P.w[3]*Th)
    dom_dT = P.w[0] * (P.w[2] + P.w[3]*ph)

    # ── (f) Equilibrium solver ──────────────────────────────────────────
    x = _solve_x_vec_duska(DG, Th, om)

    # ── (g) Mixture volume and entropy (Eqs. 36-37) ────────────────────
    DeltaV = dDG_dp
    Vh = VA + x * DeltaV + dom_dp * x * (1.0 - x)

    x_c = np.clip(x, EPS, 1.0 - EPS)
    mix_entropy = x_c * np.log(x_c) + (1.0 - x_c) * np.log(1.0 - x_c)
    Sh = SA - x * dDG_dT - mix_entropy - dom_dT * x * (1.0 - x)

    # ── (h) dx/dTh and dx/dph (Eqs. 38-40) ─────────────────────────────
    Fx = Th / (x_c * (1.0 - x_c)) - 2.0 * om
    lnrat = np.log(x_c / (1.0 - x_c))
    F_Th = dDG_dT + lnrat + dom_dT * (1.0 - 2.0 * x)
    F_ph = DeltaV + dom_dp * (1.0 - 2.0 * x)
    Fx_safe = np.where(np.abs(Fx) > 1e-30, Fx, 1e30)
    dx_dTh = -F_Th / Fx_safe
    dx_dph = -F_ph / Fx_safe

    # ── (i) dVh/dph ─────────────────────────────────────────────────────
    dVh_dp_x = dVA_dp + x * d2DG_dp2
    dVhdx = DeltaV + dom_dp * (1.0 - 2.0 * x)
    dVh_dp = dVh_dp_x + dVhdx * dx_dph

    # ── (j) Analytical dVA/dTh ──────────────────────────────────────────
    du_dTh = np.where(u > 1e-30, ph * dpS / (2.0 * pS**2 * u), 0.0)
    dN_dTh = dVS * B + VS * dB
    dD_dTh = du_dTh + dB
    dVA_dTh = np.where(u > 1e-30,
        (dN_dTh * denom_VA - VS * B * dD_dTh) / denom_VA**2, 0.0)

    # ── (k) dVh/dTh ─────────────────────────────────────────────────────
    dVh_dT_x = dVA_dTh + x * d2DG_dpT + P.w[0] * P.w[3] * x * (1.0 - x)
    dVh_dT = dVh_dT_x + dVhdx * dx_dTh

    # ── (l) dSA/dTh via finite difference ───────────────────────────────
    delta = 1e-7
    SA_p = _SA_vec(ph, Th + delta)
    SA_m = _SA_vec(ph, Th - delta)
    dSA_dTh = (SA_p - SA_m) / (2.0 * delta)

    # ── (m) dSh/dTh ─────────────────────────────────────────────────────
    dSh_dT_x = dSA_dTh - x * d2DG_dT2
    dShdx = np.where(
        (x > EPS) & (x < 1.0 - EPS),
        -dDG_dT + np.log((1.0 - x_c) / x_c) - dom_dT * (1.0 - 2.0 * x),
        0.0)
    dSh_dT = dSh_dT_x + dShdx * dx_dTh

    # ── (n) Gibbs energy via Gauss-Legendre integral ────────────────────
    nodes, weights = _GL_NODES, _GL_WEIGHTS

    # Leg 1: T-integral from Th=1 to Th at ph=0
    g_T_leg = np.zeros(n)
    dTh_val = Th - 1.0
    half_dTh = dTh_val / 2.0
    for node, weight in zip(nodes, weights):
        Th_i = half_dTh * (node + 1.0) + 1.0
        SA_i = _SA_vec(np.zeros(n), Th_i)
        g_T_leg += weight * (-SA_i)
    g_T_leg *= half_dTh

    # Leg 2: P-integral from ph=0 to ph at fixed Th
    # pS, VS, B at fixed Th are already computed
    g_P_leg = np.zeros(n)
    half_ph = ph / 2.0
    for node, weight in zip(nodes, weights):
        ph_i = half_ph * (node + 1.0)
        u_i = np.sqrt(np.maximum(1.0 - ph_i / pS, 0.0))
        VA_i = VS * B / (u_i + B)
        g_P_leg += weight * VA_i
    g_P_leg *= half_ph

    g_A = g_T_leg + g_P_leg

    # ── (o) Reduced Gibbs energies ──────────────────────────────────────
    g_mix = g_A + x * DG + Th * mix_entropy + om * x * (1.0 - x)
    g_B = g_A + DG

    # ── (p) Property conversion: reduced -> physical (vectorized) ───────
    def _phys_vec(Vh_v, Sh_v, dVh_dp_v, dVh_dT_v, dSh_dT_v, Th_v, ph_v,
                  g_red_v):
        T_v = Th_v * P.T_VLCP
        V_phys = Vh_v * P.R_specific * P.T_VLCP / (P.p_VLCP * 1e6)
        rho = np.where(V_phys > 0, 1.0 / V_phys, np.inf)
        S_phys = Sh_v * P.R_specific
        Cp = P.R_specific * Th_v * dSh_dT_v

        kappa_T_red = np.where(np.abs(dVh_dp_v) > 1e-30,
                               -(1.0 / Vh_v) * dVh_dp_v, np.inf)
        kappa_T = kappa_T_red / (P.p_VLCP * 1e6)
        Kt = np.where((np.abs(kappa_T) > 1e-30) & np.isfinite(kappa_T),
                      1.0 / kappa_T / 1e6, 0.0)

        alpha = np.where(Vh_v > 0,
                         (1.0 / Vh_v) * dVh_dT_v / P.T_VLCP, 0.0)

        Cv = np.where((kappa_T > 0) & np.isfinite(kappa_T),
                      Cp - T_v * V_phys * alpha**2 / kappa_T, Cp)

        kappa_S = np.where(Cp > 0,
                           kappa_T - T_v * V_phys * alpha**2 / Cp,
                           kappa_T)
        Ks = np.where(kappa_S > 0, 1.0 / kappa_S / 1e6, np.inf)
        rho_kS = np.maximum(rho * kappa_S, 1e-300)  # avoid sqrt of negative
        vel = np.where((rho > 0) & (kappa_S > 0),
                       np.sqrt(1.0 / rho_kS), np.nan)

        G_val = P.R_specific * P.T_VLCP * g_red_v
        return rho, V_phys, S_phys, G_val, Cp, Cv, Kt, Ks, alpha, vel

    # ── Mixture properties ──────────────────────────────────────────────
    (rho, V, S, G, Cp, Cv, Kt, Ks, alpha, vel) = _phys_vec(
        Vh, Sh, dVh_dp, dVh_dT, dSh_dT, Th, ph, g_mix)

    # ── State A properties ──────────────────────────────────────────────
    (rho_A, V_A, S_A, G_A, Cp_A, Cv_A, Kt_A, Ks_A, alpha_A, vel_A) = \
        _phys_vec(VA, SA, dVA_dp, dVA_dTh, dSA_dTh, Th, ph, g_A)

    # ── (r) State B properties ──────────────────────────────────────────
    VB = VA + DeltaV
    SB = SA - dDG_dT
    dVB_dp = dVA_dp + d2DG_dp2
    dVB_dTh = dVA_dTh + d2DG_dpT
    dSB_dTh = dSA_dTh - d2DG_dT2
    (rho_B, V_B, S_B, G_B, Cp_B, Cv_B, Kt_B, Ks_B, alpha_B, vel_B) = \
        _phys_vec(VB, SB, dVB_dp, dVB_dTh, dSB_dTh, Th, ph, g_B)

    # ── (q) IAPWS-95 reference state alignment ─────────────────────────
    for S_arr, G_arr in [(S, G), (S_A, G_A), (S_B, G_B)]:
        S_arr += P.S_OFFSET
        G_arr += P.H_OFFSET - T_K * P.S_OFFSET

    # ── Derived thermodynamic potentials ────────────────────────────────
    p_Pa = p_MPa * 1e6
    H   = G   + T_K * S;    U   = H   - p_Pa * V;    A_pot   = G   - p_Pa * V
    H_A = G_A + T_K * S_A;  U_A = H_A - p_Pa * V_A;  A_pot_A = G_A - p_Pa * V_A
    H_B = G_B + T_K * S_B;  U_B = H_B - p_Pa * V_B;  A_pot_B = G_B - p_Pa * V_B

    # ── Assemble output dict ────────────────────────────────────────────
    result = {
        'rho': rho, 'V': V, 'S': S, 'G': G, 'H': H, 'U': U, 'A': A_pot,
        'Cp': Cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks, 'alpha': alpha, 'vel': vel,
        'x': x,
        'rho_A': rho_A, 'V_A': V_A, 'S_A': S_A, 'G_A': G_A, 'H_A': H_A,
        'U_A': U_A, 'A_A': A_pot_A, 'Cp_A': Cp_A, 'Cv_A': Cv_A,
        'Kt_A': Kt_A, 'Ks_A': Ks_A, 'alpha_A': alpha_A, 'vel_A': vel_A,
        'rho_B': rho_B, 'V_B': V_B, 'S_B': S_B, 'G_B': G_B, 'H_B': H_B,
        'U_B': U_B, 'A_B': A_pot_B, 'Cp_B': Cp_B, 'Cv_B': Cv_B,
        'Kt_B': Kt_B, 'Ks_B': Ks_B, 'alpha_B': alpha_B, 'vel_B': vel_B,
    }
    return result
