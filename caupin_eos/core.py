"""
Caupin & Anisimov (2019) two-state EoS: core engine.

All internal calculations use reduced variables:
  ΔT̂ = (T - Tc)/Tc,  ΔP̂ = (P - Pc)·Vc/(RTc)
  Ĝ = G/(RTc)

The main entry point is compute_properties(T_K, p_MPa).

Reference: F. Caupin and M. A. Anisimov, J. Chem. Phys. 151, 034503 (2019).
"""

import math
from . import params as P


# ═══════════════════════════════════════════════════════════════════════════
# 1. Liquid-vapor spinodal (Eq. 2)
# ═══════════════════════════════════════════════════════════════════════════

def spinodal_pressure(T_K):
    """
    Liquid-vapor spinodal pressure Ps(T) from TIP4P/2005 (Eq. 2).

    Returns Ps [MPa], dPs/dT [MPa/K], d²Ps/dT² [MPa/K²].
    """
    dT = T_K - P.ps_T0
    Ps = P.ps_a + P.ps_b * dT + P.ps_c * dT**2
    dPs_dT = P.ps_b + 2.0 * P.ps_c * dT
    d2Ps_dT2 = 2.0 * P.ps_c
    return Ps, dPs_dT, d2Ps_dT2


# ═══════════════════════════════════════════════════════════════════════════
# 2. Spinodal Gibbs contribution Ĝ^σ (Eqs. 1, 8)
# ═══════════════════════════════════════════════════════════════════════════

def compute_Gsigma(dTh, dPh, T_K):
    """
    Compute Ĝ^σ = Â(T)·[P̂ - P̂s(T)]^{3/2} and all derivatives
    w.r.t. ΔT̂ and ΔP̂ up to second order.

    Parameters
    ----------
    dTh : float  — ΔT̂ = (T - Tc)/Tc
    dPh : float  — ΔP̂ = (P - Pc)·Vc/(RTc)
    T_K : float  — temperature in K (needed for spinodal)

    Returns
    -------
    Gs, dGs_dP, dGs_dT, d2Gs_dP2, d2Gs_dT2, d2Gs_dPdT
    """
    # Â(T) = Â₀ + Â₁·ΔT̂
    Ah = P.A0 + P.A1 * dTh
    dAh_dT = P.A1  # ∂Â/∂ΔT̂

    # Spinodal in reduced pressure: P̂s = Ps·Vc/(RTc) = Ps/P_scale
    Ps, dPs_dT, d2Ps_dT2 = spinodal_pressure(T_K)
    Phs = Ps / P_scale_MPa_local()

    # u = P̂ - P̂s = (ΔP̂ + P̂c) - P̂s = ΔP̂ + (Pc - Ps)/P_scale
    # where P̂ = P/P_scale = ΔP̂ + Pc/P_scale
    dPhs = (P.Pc - Ps) / P_scale_MPa_local()  # P̂c - P̂s
    u = dPh + dPhs  # = P̂ - P̂s

    # Derivatives of dPhs w.r.t. ΔT̂ (chain rule: ∂/∂ΔT̂ = Tc·∂/∂T)
    # ∂P̂s/∂ΔT̂ = (dPs/dT)·Tc / P_scale
    dPhs_dTh = dPs_dT * P.Tc / P_scale_MPa_local()
    d2Phs_dTh2 = d2Ps_dT2 * P.Tc**2 / P_scale_MPa_local()

    # ∂u/∂ΔT̂ = -∂P̂s/∂ΔT̂ = -dPhs_dTh
    # (since ΔP̂ doesn't depend on ΔT̂, and dPhs = Phc - Phs)
    # Actually u = dPh + (Pc - Ps)/P_scale, so ∂u/∂ΔT̂ = -(∂Ps/∂ΔT̂)/P_scale = -dPhs_dTh
    # Wait: dPhs = (Pc - Ps)/P_scale, so ∂dPhs/∂ΔT̂ = -(dPs/dT)·Tc/P_scale = -dPhs_dTh
    # No: dPhs_dTh = dPs_dT·Tc/P_scale (positive if dPs_dT > 0)
    # u = dPh + dPhs, ∂u/∂ΔT̂ = ∂dPhs/∂ΔT̂ = -(dPs/dT·Tc)/P_scale = -dPhs_dTh
    du_dTh = -dPhs_dTh
    d2u_dTh2 = -d2Phs_dTh2

    # Safety: clamp u to small positive value near spinodal
    u_safe = max(u, 1e-30)
    sqrt_u = math.sqrt(u_safe)
    u32 = u_safe * sqrt_u        # u^{3/2}
    u12 = sqrt_u                  # u^{1/2}
    u_m12 = 1.0 / sqrt_u if u_safe > 1e-30 else 0.0  # u^{-1/2}

    # Ĝ^σ = Â · u^{3/2}
    Gs = Ah * u32

    # ∂Ĝ^σ/∂ΔP̂ = (3/2)·Â·u^{1/2}  (since ∂u/∂ΔP̂ = 1)
    dGs_dP = 1.5 * Ah * u12

    # ∂²Ĝ^σ/∂ΔP̂² = (3/4)·Â·u^{-1/2}
    d2Gs_dP2 = 0.75 * Ah * u_m12

    # ∂Ĝ^σ/∂ΔT̂ = Â₁·u^{3/2} + (3/2)·Â·u^{1/2}·(∂u/∂ΔT̂)
    dGs_dT = dAh_dT * u32 + 1.5 * Ah * u12 * du_dTh

    # ∂²Ĝ^σ/∂ΔP̂∂ΔT̂ = (3/2)·Â₁·u^{1/2} + (3/4)·Â·u^{-1/2}·(∂u/∂ΔT̂)
    d2Gs_dPdT = 1.5 * dAh_dT * u12 + 0.75 * Ah * u_m12 * du_dTh

    # ∂²Ĝ^σ/∂ΔT̂² = 3·Â₁·u^{1/2}·(∂u/∂ΔT̂)
    #              + (3/4)·Â·u^{-1/2}·(∂u/∂ΔT̂)²
    #              + (3/2)·Â·u^{1/2}·(∂²u/∂ΔT̂²)
    d2Gs_dT2 = (3.0 * dAh_dT * u12 * du_dTh
                + 0.75 * Ah * u_m12 * du_dTh**2
                + 1.5 * Ah * u12 * d2u_dTh2)

    return Gs, dGs_dP, dGs_dT, d2Gs_dP2, d2Gs_dT2, d2Gs_dPdT


def P_scale_MPa_local():
    """Pressure scale in MPa (cached-like helper)."""
    return P.P_scale_MPa


# ═══════════════════════════════════════════════════════════════════════════
# 3. Polynomial part of Ĝ^A (Eq. 6)
# ═══════════════════════════════════════════════════════════════════════════

def compute_GA_poly(dTh, dPh):
    """
    Polynomial contribution: Σ c_mn · ΔT̂^m · ΔP̂^n
    Returns value and all first/second derivatives w.r.t. ΔT̂, ΔP̂.
    """
    # Shorthand
    T, Q = dTh, dPh
    T2, T3, T4 = T**2, T**3, T**4
    Q2, Q3, Q4 = Q**2, Q**3, Q**4

    # Value
    val = (P.c01*Q + P.c02*Q2 + P.c11*T*Q + P.c20*T2
           + P.c03*Q3 + P.c12*T*Q2 + P.c21*T2*Q + P.c30*T3
           + P.c04*Q4 + P.c13*T*Q3 + P.c22*T2*Q2 + P.c40*T4
           + P.c14*T*Q4)

    # ∂/∂ΔP̂
    dval_dP = (P.c01 + 2*P.c02*Q + P.c11*T
               + 3*P.c03*Q2 + 2*P.c12*T*Q + P.c21*T2
               + 4*P.c04*Q3 + 3*P.c13*T*Q2 + 2*P.c22*T2*Q
               + 4*P.c14*T*Q3)

    # ∂/∂ΔT̂
    dval_dT = (P.c11*Q + 2*P.c20*T
               + P.c12*Q2 + 2*P.c21*T*Q + 3*P.c30*T2
               + P.c13*Q3 + 2*P.c22*T*Q2 + 4*P.c40*T3
               + P.c14*Q4)

    # ∂²/∂ΔP̂²
    d2val_dP2 = (2*P.c02 + 6*P.c03*Q + 2*P.c12*T
                 + 12*P.c04*Q2 + 6*P.c13*T*Q + 2*P.c22*T2
                 + 12*P.c14*T*Q2)

    # ∂²/∂ΔT̂²
    d2val_dT2 = (2*P.c20 + 2*P.c21*Q + 6*P.c30*T
                 + 2*P.c22*Q2 + 12*P.c40*T2)

    # ∂²/∂ΔP̂∂ΔT̂
    d2val_dPdT = (P.c11 + 2*P.c12*Q + 2*P.c21*T
                  + 3*P.c13*Q2 + 4*P.c22*T*Q
                  + 4*P.c14*Q3)

    return val, dval_dP, dval_dT, d2val_dP2, d2val_dT2, d2val_dPdT


def compute_GA(dTh, dPh, T_K):
    """
    Full Ĝ^A = Ĝ^σ + Σ c_mn ΔT̂^m ΔP̂^n and all derivatives.
    """
    Gs, dGs_dP, dGs_dT, d2Gs_dP2, d2Gs_dT2, d2Gs_dPdT = \
        compute_Gsigma(dTh, dPh, T_K)
    Gp, dGp_dP, dGp_dT, d2Gp_dP2, d2Gp_dT2, d2Gp_dPdT = \
        compute_GA_poly(dTh, dPh)

    return (Gs + Gp,
            dGs_dP + dGp_dP,
            dGs_dT + dGp_dT,
            d2Gs_dP2 + d2Gp_dP2,
            d2Gs_dT2 + d2Gp_dT2,
            d2Gs_dPdT + d2Gp_dPdT)


# ═══════════════════════════════════════════════════════════════════════════
# 4. State B−A Gibbs difference Ĝ^BA (Eq. 7)
# ═══════════════════════════════════════════════════════════════════════════

def compute_GBA(dTh, dPh):
    """
    Ĝ^BA = λ(ΔT̂ + a·ΔP̂ + b·ΔT̂·ΔP̂ + d·ΔP̂² + f·ΔT̂²)
    Returns value and all first/second derivatives.
    """
    GBA = P.lam * (dTh + P.a*dPh + P.b*dTh*dPh + P.d*dPh**2 + P.f*dTh**2)

    dGBA_dP = P.lam * (P.a + P.b*dTh + 2*P.d*dPh)
    dGBA_dT = P.lam * (1.0 + P.b*dPh + 2*P.f*dTh)

    d2GBA_dP2 = 2*P.lam*P.d
    d2GBA_dT2 = 2*P.lam*P.f
    d2GBA_dPdT = P.lam*P.b

    return GBA, dGBA_dP, dGBA_dT, d2GBA_dP2, d2GBA_dT2, d2GBA_dPdT


# ═══════════════════════════════════════════════════════════════════════════
# 5. Interaction parameter ω̂ (Eq. 5)
# ═══════════════════════════════════════════════════════════════════════════

def compute_omega(dTh, dPh):
    """
    ω̂ = (2 + ω₀·ΔP̂) / T̂   where T̂ = 1 + ΔT̂
    Returns ω̂ and all first/second derivatives.
    """
    Th = 1.0 + dTh
    num = 2.0 + P.omega0 * dPh
    om = num / Th

    dom_dP = P.omega0 / Th
    dom_dT = -num / Th**2     # = -ω̂/T̂

    d2om_dP2 = 0.0
    d2om_dT2 = 2.0 * num / Th**3    # = 2ω̂/T̂²
    d2om_dPdT = -P.omega0 / Th**2

    return om, dom_dP, dom_dT, d2om_dP2, d2om_dT2, d2om_dPdT


# ═══════════════════════════════════════════════════════════════════════════
# 6. Equilibrium solver
# ═══════════════════════════════════════════════════════════════════════════

def solve_x(GBA, Th, om, x0=0.1, max_iter=200, tol=1e-12):
    """Solve F(x) = GBA + T̂·[ln(x/(1-x)) + ω̂·(1-2x)] = 0 via Newton."""
    x = x0
    EPS = 1e-15
    for _ in range(max_iter):
        x = max(EPS, min(1.0 - EPS, x))
        lnrat = math.log(x / (1.0 - x))
        F = GBA + Th * (lnrat + om * (1.0 - 2.0 * x))
        Fx = Th * (1.0 / (x * (1.0 - x)) - 2.0 * om)
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


def _solve_x_stable(GBA, Th, om):
    """
    Find the globally stable equilibrium x (lowest Gibbs energy of mixing).
    Uses both Newton from multiple starting points and bracket search
    via inflection points, then picks the root with minimum g.
    """
    EPS = 1e-12

    def _F(x):
        return GBA + Th * (math.log(x / (1.0 - x)) + om * (1.0 - 2.0 * x))

    def _g(x):
        if EPS < x < 1.0 - EPS:
            mix_ent = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
        else:
            mix_ent = 0.0
        return x * GBA + Th * (mix_ent + om * x * (1.0 - x))

    def _newton(x0):
        x = x0
        for _ in range(200):
            x = max(EPS, min(1.0 - EPS, x))
            lnrat = math.log(x / (1.0 - x))
            F = GBA + Th * (lnrat + om * (1.0 - 2.0 * x))
            Fx = Th * (1.0 / (x * (1.0 - x)) - 2.0 * om)
            if abs(Fx) < 1e-30:
                break
            dx = -F / Fx
            if x + dx < EPS:
                x = x / 2.0
            elif x + dx > 1.0 - EPS:
                x = (x + 1.0 - EPS) / 2.0
            else:
                x = x + dx
            if abs(F) < 1e-12:
                break
        return x

    candidates = []

    # Newton from multiple starting points
    for x0 in (0.05, 0.5, 0.95):
        xr = _newton(x0)
        if EPS < xr < 1.0 - EPS and abs(_F(xr)) < 1e-8:
            candidates.append(xr)

    # Bracket search via inflection points
    if om > 0:
        disc = 1.0 - 2.0 / om
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
        return solve_x(GBA, Th, om)

    # Deduplicate
    unique = [candidates[0]]
    for c in candidates[1:]:
        if all(abs(c - u) > 1e-6 for u in unique):
            unique.append(c)

    # Pick minimum g
    best_x = unique[0]
    best_g = _g(unique[0])
    for r in unique[1:]:
        gr = _g(r)
        if gr < best_g:
            best_g = gr
            best_x = r
    return best_x


# ═══════════════════════════════════════════════════════════════════════════
# 7. Property conversion: reduced → physical
# ═══════════════════════════════════════════════════════════════════════════

def _physical_props(Vh, Sh_red, d2G_dP2_total, d2G_dT2_total,
                    d2G_dPdT_total, T_K, G_hat_red=0.0):
    """
    Convert reduced Gibbs derivatives to physical properties.

    Vh = ∂Ĝ/∂ΔP̂   (= V_molar / Vc, dimensionless)
    Sh_red = -∂Ĝ/∂ΔT̂  (= S_molar / R, dimensionless)
    G_hat_red = Ĝ (reduced Gibbs energy, dimensionless)

    d2G_dP2_total  = d²Ĝ/dΔP̂²  (total, including dx terms)
    d2G_dT2_total  = d²Ĝ/dΔT̂²
    d2G_dPdT_total = d²Ĝ/dΔP̂dΔT̂

    Returns dict with: rho, V, S, Cp, Cv, Kt, Ks, alpha, vel, G
    """
    Th = T_K / P.Tc

    # Molar volume and specific volume
    V_molar = P.Vc * Vh                     # m³/mol
    V_spec = V_molar / P.M_H2O              # m³/kg
    rho = 1.0 / V_spec if V_spec > 0 else float('inf')

    # Molar entropy and specific entropy
    S_molar = P.R * Sh_red                   # J/(mol·K)
    S_spec = S_molar / P.M_H2O              # J/(kg·K)

    # Cp_molar = -R·T̂·(d²Ĝ/dΔT̂²)_total
    Cp_molar = -P.R * Th * d2G_dT2_total    # J/(mol·K)
    Cp = Cp_molar / P.M_H2O                 # J/(kg·K)

    # κT = -(Vc/(RTc)) · (d²Ĝ/dΔP̂²)_total / (∂Ĝ/∂ΔP̂)  [Pa⁻¹]
    if abs(Vh) > 1e-30:
        kappa_T = -(P.Vc / (P.R * P.Tc)) * d2G_dP2_total / Vh  # Pa⁻¹
    else:
        kappa_T = float('inf')

    # Kt = 1/κT  [Pa → MPa]
    if abs(kappa_T) > 1e-30 and kappa_T != float('inf'):
        Kt = 1.0 / kappa_T / 1e6            # MPa
    else:
        Kt = 0.0

    # αP = (1/Tc) · (d²Ĝ/dΔP̂dΔT̂)_total / (∂Ĝ/∂ΔP̂)  [K⁻¹]
    if abs(Vh) > 1e-30:
        alpha = (1.0 / P.Tc) * d2G_dPdT_total / Vh
    else:
        alpha = 0.0

    # Cv = Cp - T·V·α²/κT
    if kappa_T > 0 and kappa_T != float('inf'):
        Cv = Cp - T_K * V_spec * alpha**2 / kappa_T
    else:
        Cv = Cp

    # κS = κT - T·V·α²/Cp  (adiabatic compressibility)
    if Cp > 0:
        kappa_S = kappa_T - T_K * V_spec * alpha**2 / Cp
    else:
        kappa_S = kappa_T

    Ks = 1.0 / kappa_S / 1e6 if kappa_S > 0 else float('inf')

    # Speed of sound: w = sqrt(1/(ρ·κS))
    if rho > 0 and kappa_S > 0:
        vel = math.sqrt(1.0 / (rho * kappa_S))
    else:
        vel = float('nan')

    # Gibbs energy: G = R*Tc*G_hat / M_H2O [J/kg]
    G_val = P.R * P.Tc * G_hat_red / P.M_H2O

    return {
        'rho': rho, 'V': V_spec, 'S': S_spec,
        'Cp': Cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks,
        'alpha': alpha, 'vel': vel, 'G': G_val,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def compute_properties(T_K, p_MPa, _compute_Kp=False):
    """
    Compute all thermodynamic properties at a single (T, p) point.

    Parameters
    ----------
    T_K : float   — temperature in K
    p_MPa : float — pressure in MPa

    Returns
    -------
    dict : properties for mixture, state A (suffix _A), state B (suffix _B)
    """
    # ── Reduced variables ──────────────────────────────────────────────
    dTh = (T_K - P.Tc) / P.Tc
    dPh = (p_MPa - P.Pc) / P.P_scale_MPa
    Th = 1.0 + dTh  # = T/Tc

    # ── State A Gibbs energy and derivatives ───────────────────────────
    GA, dGA_dP, dGA_dT, d2GA_dP2, d2GA_dT2, d2GA_dPdT = \
        compute_GA(dTh, dPh, T_K)

    # ── State B−A Gibbs difference ─────────────────────────────────────
    GBA, dGBA_dP, dGBA_dT, d2GBA_dP2, d2GBA_dT2, d2GBA_dPdT = \
        compute_GBA(dTh, dPh)

    # ── Interaction parameter ──────────────────────────────────────────
    om, dom_dP, dom_dT, d2om_dP2, d2om_dT2, d2om_dPdT = \
        compute_omega(dTh, dPh)

    # ── Solve for equilibrium x ────────────────────────────────────────
    x = _solve_x_stable(GBA, Th, om)

    # ── Ω = T̂·ω̂ = 2 + ω₀·ΔP̂ (independent of T) ──────────────────────
    # Using Ω simplifies all mixing contributions since ∂Ω/∂ΔT̂ = 0.
    Om = Th * om                # = 2 + omega0 * dPh
    dOm_dP = P.omega0           # only P-derivative is nonzero

    # ── Total reduced volume V̂ = ∂Ĝ/∂ΔP̂ ──────────────────────────────
    # Ĝ = ĜA + x·ĜBA + T̂·idmix + Ω·x(1-x)
    Vh = dGA_dP + x * dGBA_dP + dOm_dP * x * (1.0 - x)

    # ── Total reduced entropy Ŝ = -∂Ĝ/∂ΔT̂ ────────────────────────────
    # ∂(T̂·idmix)/∂ΔT̂ = idmix;  ∂(Ω·x(1-x))/∂ΔT̂ = 0
    if 0 < x < 1:
        mix_ent = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    else:
        mix_ent = 0.0
    Sh_red = -(dGA_dT + x * dGBA_dT + mix_ent)

    # ── dx/dΔT̂ and dx/dΔP̂ via implicit differentiation ────────────────
    # F(x) = ∂Ĝ/∂x = GBA + T̂·[ln(x/(1-x)) + ω̂·(1-2x)]
    #       = GBA + T̂·ln(x/(1-x)) + Ω·(1-2x)
    Fx = Th * (1.0 / (x * (1.0 - x)) - 2.0 * om) if 0 < x < 1 else 1e30
    lnrat = math.log(x / (1.0 - x)) if 0 < x < 1 else 0.0

    # ∂F/∂ΔT̂ = ∂GBA/∂ΔT̂ + ln(x/(1-x))  [Ω term vanishes since ∂Ω/∂ΔT̂=0]
    F_dT = dGBA_dT + lnrat
    # ∂F/∂ΔP̂ = ∂GBA/∂ΔP̂ + ω₀·(1-2x)
    F_dP = dGBA_dP + dOm_dP * (1.0 - 2.0 * x)

    dx_dT = -F_dT / Fx if abs(Fx) > 1e-30 else 0.0
    dx_dP = -F_dP / Fx if abs(Fx) > 1e-30 else 0.0

    # ── Total second derivatives of Ĝ (including dx contributions) ─────
    # At fixed x, all mixing second derivatives vanish:
    #   ∂²(Ω·x(1-x))/∂ΔP̂² = 0, ∂²(T̂·idmix)/∂ΔT̂² = 0, cross = 0
    d2G_dP2_x = d2GA_dP2 + x * d2GBA_dP2
    d2G_dT2_x = d2GA_dT2 + x * d2GBA_dT2
    d2G_dPdT_x = d2GA_dPdT + x * d2GBA_dPdT

    # Total: d²Ĝ/dΔP̂² = (∂²Ĝ/∂ΔP̂²)_x + F_dP · dx/dΔP̂
    d2G_dP2_total = d2G_dP2_x + F_dP * dx_dP
    d2G_dT2_total = d2G_dT2_x + F_dT * dx_dT
    d2G_dPdT_total = d2G_dPdT_x + F_dP * dx_dT

    # ── Reduced Gibbs energy values for G computation ──────────────────
    # Mixture: Ĝ = ĜA + x·ĜBA + T̂·(x ln x + (1-x)ln(1-x)) + Ω·x(1-x)
    G_hat_mix = GA + x * GBA + Th * mix_ent + Om * x * (1.0 - x)
    G_hat_A = GA              # state A: x=0, all mixing terms vanish
    G_hat_B = GA + GBA        # state B: x=1, mixing terms vanish

    # ── Mixture physical properties ────────────────────────────────────
    mix = _physical_props(Vh, Sh_red, d2G_dP2_total, d2G_dT2_total,
                          d2G_dPdT_total, T_K, G_hat_mix)
    mix['x'] = x

    # ── State A properties (x = 0, no mixing terms) ───────────────────
    stateA = _physical_props(dGA_dP, -dGA_dT,
                             d2GA_dP2, d2GA_dT2, d2GA_dPdT, T_K, G_hat_A)

    # ── State B properties ─────────────────────────────────────────────
    # Ĝ^B = Ĝ^A + Ĝ^BA → derivatives add
    VhB = dGA_dP + dGBA_dP
    ShB_red = -(dGA_dT + dGBA_dT)
    d2GB_dP2 = d2GA_dP2 + d2GBA_dP2
    d2GB_dT2 = d2GA_dT2 + d2GBA_dT2
    d2GB_dPdT = d2GA_dPdT + d2GBA_dPdT
    stateB = _physical_props(VhB, ShB_red,
                             d2GB_dP2, d2GB_dT2, d2GB_dPdT, T_K, G_hat_B)

    # ── Assemble output ────────────────────────────────────────────────
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
    Compute thermodynamic properties at a given (T, p) with a forced x value
    (instead of solving for equilibrium x). Useful for evaluating properties
    along binodal/spinodal branches.

    Returns dict with: rho, V, S, G, Cp, Cv, Kt, Ks, alpha, vel, x,
                       H, U, A, Kp
    """
    dTh = (T_K - P.Tc) / P.Tc
    dPh = (p_MPa - P.Pc) / P.P_scale_MPa
    Th = 1.0 + dTh

    GA, dGA_dP, dGA_dT, d2GA_dP2, d2GA_dT2, d2GA_dPdT = \
        compute_GA(dTh, dPh, T_K)
    GBA, dGBA_dP, dGBA_dT, d2GBA_dP2, d2GBA_dT2, d2GBA_dPdT = \
        compute_GBA(dTh, dPh)
    om, dom_dP, dom_dT, d2om_dP2, d2om_dT2, d2om_dPdT = \
        compute_omega(dTh, dPh)

    # Ω = T̂·ω̂ = 2 + ω₀·ΔP̂
    Om = Th * om
    dOm_dP = P.omega0

    # Total V̂ and Ŝ at forced x
    Vh = dGA_dP + x * dGBA_dP + dOm_dP * x * (1.0 - x)

    if 0 < x < 1:
        mix_ent = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    else:
        mix_ent = 0.0
    Sh_red = -(dGA_dT + x * dGBA_dT + mix_ent)

    # dx/dΔT̂ and dx/dΔP̂
    Fx = Th * (1.0 / (x * (1.0 - x)) - 2.0 * om) if 0 < x < 1 else 1e30
    lnrat = math.log(x / (1.0 - x)) if 0 < x < 1 else 0.0
    F_dT = dGBA_dT + lnrat
    F_dP = dGBA_dP + dOm_dP * (1.0 - 2.0 * x)
    dx_dT = -F_dT / Fx if abs(Fx) > 1e-30 else 0.0
    dx_dP = -F_dP / Fx if abs(Fx) > 1e-30 else 0.0

    # Total second derivatives (mixing second derivatives vanish at fixed x)
    d2G_dP2_x = d2GA_dP2 + x * d2GBA_dP2
    d2G_dT2_x = d2GA_dT2 + x * d2GBA_dT2
    d2G_dPdT_x = d2GA_dPdT + x * d2GBA_dPdT

    d2G_dP2_total = d2G_dP2_x + F_dP * dx_dP
    d2G_dT2_total = d2G_dT2_x + F_dT * dx_dT
    d2G_dPdT_total = d2G_dPdT_x + F_dP * dx_dT

    # Reduced Gibbs energy at forced x
    G_hat = GA + x * GBA + Th * mix_ent + Om * x * (1.0 - x)

    props = _physical_props(Vh, Sh_red, d2G_dP2_total, d2G_dT2_total,
                            d2G_dPdT_total, T_K, G_hat)
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
# 9. Vectorized batch computation
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np


def _solve_x_vec_caupin(GBA, Th, om):
    """
    Vectorized Newton solver for the equilibrium x.

    Solves F(x) = GBA + T̂·[ln(x/(1-x)) + ω̂·(1-2x)] = 0
    from two starting points (0.05 and 0.95) and picks the root
    with lower Gibbs energy of mixing for each point.

    Parameters
    ----------
    GBA : 1-D array — state B−A Gibbs difference (reduced)
    Th  : 1-D array — T̂ = T/Tc
    om  : 1-D array — ω̂ = (2 + ω₀·ΔP̂) / T̂

    Returns
    -------
    x : 1-D array — equilibrium tetrahedral fraction
    """
    EPS = 1e-15
    n = len(GBA)

    def _newton_vec(x0_val):
        x = np.full(n, x0_val)
        for _ in range(50):
            x = np.clip(x, EPS, 1.0 - EPS)
            lnrat = np.log(x / (1.0 - x))
            F = GBA + Th * (lnrat + om * (1.0 - 2.0 * x))
            Fx = Th * (1.0 / (x * (1.0 - x)) - 2.0 * om)
            # Avoid division by zero in Fx
            Fx_safe = np.where(np.abs(Fx) < 1e-30, 1e-30, Fx)
            dx = -F / Fx_safe
            # Damped step
            x_new = x + dx
            x = np.where(x_new < EPS, x / 2.0,
                    np.where(x_new > 1.0 - EPS, (x + 1.0 - EPS) / 2.0,
                             x_new))
        x = np.clip(x, EPS, 1.0 - EPS)
        return x

    # Solve from two starting points
    x_lo = _newton_vec(0.05)
    x_hi = _newton_vec(0.95)

    # Gibbs energy of mixing: g(x) = x*GBA + T̂*(x ln x + (1-x)ln(1-x) + ω̂*x*(1-x))
    def _g_vec(x):
        x_c = np.clip(x, EPS, 1.0 - EPS)
        mix_ent = x_c * np.log(x_c) + (1.0 - x_c) * np.log(1.0 - x_c)
        return x_c * GBA + Th * (mix_ent + om * x_c * (1.0 - x_c))

    g_lo = _g_vec(x_lo)
    g_hi = _g_vec(x_hi)

    # Pick root with lower g for each point
    x = np.where(g_lo <= g_hi, x_lo, x_hi)
    return x


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

    # ── Reduced variables ──────────────────────────────────────────────
    dTh = (T_K - P.Tc) / P.Tc
    dPh = (p_MPa - P.Pc) / P.P_scale_MPa
    Th = 1.0 + dTh  # = T/Tc

    # ── Spinodal pressure (Eq. 2) ─────────────────────────────────────
    dT_sp = T_K - P.ps_T0
    Ps = P.ps_a + P.ps_b * dT_sp + P.ps_c * dT_sp**2
    dPs_dT = P.ps_b + 2.0 * P.ps_c * dT_sp
    d2Ps_dT2 = 2.0 * P.ps_c

    # ── Spinodal Gibbs contribution Ĝ^σ ───────────────────────────────
    Ah = P.A0 + P.A1 * dTh
    dAh_dT = P.A1

    Pscale = P.P_scale_MPa

    # u = ΔP̂ + (Pc - Ps)/P_scale
    dPhs = (P.Pc - Ps) / Pscale
    u = dPh + dPhs

    # Derivatives of u w.r.t. ΔT̂
    dPhs_dTh = dPs_dT * P.Tc / Pscale
    d2Phs_dTh2 = d2Ps_dT2 * P.Tc**2 / Pscale
    du_dTh = -dPhs_dTh
    d2u_dTh2 = -d2Phs_dTh2

    # Clamp u
    u_safe = np.maximum(u, 1e-30)
    sqrt_u = np.sqrt(u_safe)
    u32 = u_safe * sqrt_u
    u12 = sqrt_u
    u_m12 = np.where(u_safe > 1e-30, 1.0 / sqrt_u, 0.0)

    # Ĝ^σ and derivatives
    Gs = Ah * u32
    dGs_dP = 1.5 * Ah * u12
    d2Gs_dP2 = 0.75 * Ah * u_m12
    dGs_dT = dAh_dT * u32 + 1.5 * Ah * u12 * du_dTh
    d2Gs_dPdT = 1.5 * dAh_dT * u12 + 0.75 * Ah * u_m12 * du_dTh
    d2Gs_dT2 = (3.0 * dAh_dT * u12 * du_dTh
                + 0.75 * Ah * u_m12 * du_dTh**2
                + 1.5 * Ah * u12 * d2u_dTh2)

    # ── Polynomial part of Ĝ^A (Eq. 6) ────────────────────────────────
    T = dTh
    Q = dPh
    T2 = T**2;  T3 = T**3;  T4 = T**4
    Q2 = Q**2;  Q3 = Q**3;  Q4 = Q**4

    Gp = (P.c01*Q + P.c02*Q2 + P.c11*T*Q + P.c20*T2
          + P.c03*Q3 + P.c12*T*Q2 + P.c21*T2*Q + P.c30*T3
          + P.c04*Q4 + P.c13*T*Q3 + P.c22*T2*Q2 + P.c40*T4
          + P.c14*T*Q4)

    dGp_dP = (P.c01 + 2*P.c02*Q + P.c11*T
              + 3*P.c03*Q2 + 2*P.c12*T*Q + P.c21*T2
              + 4*P.c04*Q3 + 3*P.c13*T*Q2 + 2*P.c22*T2*Q
              + 4*P.c14*T*Q3)

    dGp_dT = (P.c11*Q + 2*P.c20*T
              + P.c12*Q2 + 2*P.c21*T*Q + 3*P.c30*T2
              + P.c13*Q3 + 2*P.c22*T*Q2 + 4*P.c40*T3
              + P.c14*Q4)

    d2Gp_dP2 = (2*P.c02 + 6*P.c03*Q + 2*P.c12*T
                + 12*P.c04*Q2 + 6*P.c13*T*Q + 2*P.c22*T2
                + 12*P.c14*T*Q2)

    d2Gp_dT2 = (2*P.c20 + 2*P.c21*Q + 6*P.c30*T
                + 2*P.c22*Q2 + 12*P.c40*T2)

    d2Gp_dPdT = (P.c11 + 2*P.c12*Q + 2*P.c21*T
                 + 3*P.c13*Q2 + 4*P.c22*T*Q
                 + 4*P.c14*Q3)

    # ── Full Ĝ^A = Ĝ^σ + polynomial ──────────────────────────────────
    GA       = Gs + Gp
    dGA_dP   = dGs_dP + dGp_dP
    dGA_dT   = dGs_dT + dGp_dT
    d2GA_dP2 = d2Gs_dP2 + d2Gp_dP2
    d2GA_dT2 = d2Gs_dT2 + d2Gp_dT2
    d2GA_dPdT = d2Gs_dPdT + d2Gp_dPdT

    # ── State B−A Gibbs difference Ĝ^BA (Eq. 7) ───────────────────────
    GBA = P.lam * (dTh + P.a*dPh + P.b*dTh*dPh + P.d*dPh**2 + P.f*dTh**2)
    dGBA_dP = P.lam * (P.a + P.b*dTh + 2*P.d*dPh)
    dGBA_dT = P.lam * (1.0 + P.b*dPh + 2*P.f*dTh)
    d2GBA_dP2 = np.full(n, 2*P.lam*P.d)
    d2GBA_dT2 = np.full(n, 2*P.lam*P.f)
    d2GBA_dPdT = np.full(n, P.lam*P.b)

    # ── Interaction parameter ω̂ ────────────────────────────────────────
    num = 2.0 + P.omega0 * dPh
    om = num / Th

    # ── Solve for equilibrium x ────────────────────────────────────────
    x = _solve_x_vec_caupin(GBA, Th, om)

    # ── Ω = T̂·ω̂ = 2 + ω₀·ΔP̂ ─────────────────────────────────────────
    Om = Th * om                # = 2 + omega0 * dPh
    dOm_dP = P.omega0

    # ── Total reduced volume V̂ ─────────────────────────────────────────
    Vh = dGA_dP + x * dGBA_dP + dOm_dP * x * (1.0 - x)

    # ── Total reduced entropy Ŝ = -∂Ĝ/∂ΔT̂ ────────────────────────────
    EPS = 1e-15
    x_c = np.clip(x, EPS, 1.0 - EPS)
    mix_ent = x_c * np.log(x_c) + (1.0 - x_c) * np.log(1.0 - x_c)
    Sh_red = -(dGA_dT + x * dGBA_dT + mix_ent)

    # ── dx/dΔT̂ and dx/dΔP̂ via implicit differentiation ────────────────
    lnrat = np.log(x_c / (1.0 - x_c))
    Fx = Th * (1.0 / (x_c * (1.0 - x_c)) - 2.0 * om)
    Fx_safe = np.where(np.abs(Fx) > 1e-30, Fx, 1e30)

    F_dT = dGBA_dT + lnrat
    F_dP = dGBA_dP + dOm_dP * (1.0 - 2.0 * x)

    dx_dT = -F_dT / Fx_safe
    dx_dP = -F_dP / Fx_safe

    # ── Total second derivatives ───────────────────────────────────────
    d2G_dP2_x = d2GA_dP2 + x * d2GBA_dP2
    d2G_dT2_x = d2GA_dT2 + x * d2GBA_dT2
    d2G_dPdT_x = d2GA_dPdT + x * d2GBA_dPdT

    d2G_dP2_total = d2G_dP2_x + F_dP * dx_dP
    d2G_dT2_total = d2G_dT2_x + F_dT * dx_dT
    d2G_dPdT_total = d2G_dPdT_x + F_dP * dx_dT

    # ── Reduced Gibbs energy ───────────────────────────────────────────
    G_hat_mix = GA + x * GBA + Th * mix_ent + Om * x * (1.0 - x)
    G_hat_A = GA
    G_hat_B = GA + GBA

    # ── Helper: reduced → physical (vectorized) ────────────────────────
    def _phys_vec(Vh_v, Sh_red_v, d2P2, d2T2, d2PT, G_hat_v):
        """Convert reduced Gibbs derivatives to physical properties (arrays)."""
        Th_v = T_K / P.Tc

        V_molar = P.Vc * Vh_v
        V_spec = V_molar / P.M_H2O
        rho = np.where(V_spec > 0, 1.0 / V_spec, np.inf)

        S_molar = P.R * Sh_red_v
        S_spec = S_molar / P.M_H2O

        Cp_molar = -P.R * Th_v * d2T2
        Cp_v = Cp_molar / P.M_H2O

        kappa_T = np.where(np.abs(Vh_v) > 1e-30,
                           -(P.Vc / (P.R * P.Tc)) * d2P2 / Vh_v,
                           np.inf)
        Kt_v = np.where((np.abs(kappa_T) > 1e-30) & np.isfinite(kappa_T),
                        1.0 / kappa_T / 1e6, 0.0)

        alpha_v = np.where(np.abs(Vh_v) > 1e-30,
                           (1.0 / P.Tc) * d2PT / Vh_v, 0.0)

        Cv_v = np.where((kappa_T > 0) & np.isfinite(kappa_T),
                        Cp_v - T_K * V_spec * alpha_v**2 / kappa_T,
                        Cp_v)

        kappa_S = np.where(Cp_v > 0,
                           kappa_T - T_K * V_spec * alpha_v**2 / Cp_v,
                           kappa_T)
        Ks_v = np.where(kappa_S > 0, 1.0 / kappa_S / 1e6, np.inf)

        vel_v = np.where((rho > 0) & (kappa_S > 0),
                         np.sqrt(1.0 / (rho * kappa_S)), np.nan)

        G_val = P.R * P.Tc * G_hat_v / P.M_H2O

        return rho, V_spec, S_spec, G_val, Cp_v, Cv_v, Kt_v, Ks_v, alpha_v, vel_v

    # ── Mixture properties ─────────────────────────────────────────────
    (rho, V, S, G, Cp, Cv, Kt, Ks, alpha, vel) = _phys_vec(
        Vh, Sh_red, d2G_dP2_total, d2G_dT2_total, d2G_dPdT_total, G_hat_mix)

    # ── State A (x=0, no mixing) ──────────────────────────────────────
    (rho_A, V_A, S_A, G_A, Cp_A, Cv_A, Kt_A, Ks_A, alpha_A, vel_A) = _phys_vec(
        dGA_dP, -dGA_dT, d2GA_dP2, d2GA_dT2, d2GA_dPdT, G_hat_A)

    # ── State B (x=1, derivatives = GA + GBA) ─────────────────────────
    VhB = dGA_dP + dGBA_dP
    ShB_red = -(dGA_dT + dGBA_dT)
    d2GB_dP2 = d2GA_dP2 + d2GBA_dP2
    d2GB_dT2 = d2GA_dT2 + d2GBA_dT2
    d2GB_dPdT = d2GA_dPdT + d2GBA_dPdT
    (rho_B, V_B, S_B, G_B, Cp_B, Cv_B, Kt_B, Ks_B, alpha_B, vel_B) = _phys_vec(
        VhB, ShB_red, d2GB_dP2, d2GB_dT2, d2GB_dPdT, G_hat_B)

    # ── IAPWS-95 reference state alignment ─────────────────────────────
    for S_arr, G_arr in [(S, G), (S_A, G_A), (S_B, G_B)]:
        S_arr += P.S_OFFSET           # in-place
        G_arr += P.H_OFFSET - T_K * P.S_OFFSET

    # ── Derived thermodynamic potentials ───────────────────────────────
    p_Pa = p_MPa * 1e6
    H   = G   + T_K * S;    U   = H   - p_Pa * V;    A_pot   = G   - p_Pa * V
    H_A = G_A + T_K * S_A;  U_A = H_A - p_Pa * V_A;  A_pot_A = G_A - p_Pa * V_A
    H_B = G_B + T_K * S_B;  U_B = H_B - p_Pa * V_B;  A_pot_B = G_B - p_Pa * V_B

    # ── Assemble output dict ───────────────────────────────────────────
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
