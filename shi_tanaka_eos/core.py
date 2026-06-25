"""
Shi & Tanaka (2020) hierarchical two-state EoS — core engine.

Computes thermodynamic properties of liquid water from the hierarchical
two-state free energy (Eq. S1 of the supporting information):

    Ĝ(T̂, P̂) = Ĝ_ρ(T̂, P̂)  +  s · ΔĜ(T̂, P̂)
              + T̂ · [s ln s + (1-s) ln(1-s)]

with negligible cooperativity (J ≈ 0), so the LFTS fraction is analytic:

    s = 1 / (1 + exp(ΔĜ / T̂))                         (Eq. S3)

where ΔĜ = ΔÊ - T̂ Δσ̂ + P̂ ΔV̂ (Eq. S2) and Ĝ_ρ is a constrained
polynomial in (ΔT̂, ΔP̂) with a c_1·T̂·(ln T̂ - 1) term that supplies a
physically reasonable heat-capacity background for the DNLS state.

The thermodynamic identities the model is fitted to (Eq. S5-S9):

    V̂           = ∂Ĝ_ρ/∂P̂  +  s ΔV̂
    ρ̂           = n / V̂
    κ̂_T         = -(1/V̂) [∂²Ĝ_ρ/∂P̂²  -  s(1-s) ΔV̂² / T̂]
    α̂_P         =  (1/V̂) [∂²Ĝ_ρ/∂P̂∂T̂ + s(1-s) ΔV̂ (ΔÊ + P̂ ΔV̂) / T̂²]
    Ĉ_P         = -(T̂/n) [∂²Ĝ_ρ/∂T̂²  -  s(1-s) (ΔÊ + P̂ ΔV̂)² / T̂³]

All single-derivative quantities use the envelope theorem: at equilibrium,
the contribution from ∂s/∂(T̂ or P̂) cancels.

Reference: R. Shi and H. Tanaka, PNAS 117, 26591 (2020).
"""

import math

import numpy as np

from . import params as P


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Background polynomial G_ρ(T̂, P̂)  +  c_1 · T̂·(ln T̂ - 1)
# ═══════════════════════════════════════════════════════════════════════════
# Terms present in Table S2 (m, n) ordered pairs.  c_22, c_10, c_00 are
# explicitly absent.  ΔT̂ = T̂ - 1, ΔP̂ = P̂ - 1.

def _g_rho_scalar(dT, dP, T_hat):
    """Background G_ρ and all first/second derivatives — scalar form.

    Returns ``(G, dG/dP̂, dG/dT̂, d²G/dP̂², d²G/dT̂², d²G/dP̂dT̂)``.
    """
    dT2 = dT * dT;   dT3 = dT2 * dT
    dP2 = dP * dP;   dP3 = dP2 * dP

    log_T = math.log(T_hat)

    val = (
        P.c01*dP + P.c02*dP2 + P.c03*dP3
        + P.c11*dT*dP + P.c12*dT*dP2 + P.c13*dT*dP3
        + P.c20*dT2 + P.c21*dT2*dP + P.c23*dT2*dP3
        + P.c30*dT3 + P.c31*dT3*dP
        + P.c1 * T_hat * (log_T - 1.0)
    )

    dval_dP = (
        P.c01 + 2.0*P.c02*dP + 3.0*P.c03*dP2
        + P.c11*dT + 2.0*P.c12*dT*dP + 3.0*P.c13*dT*dP2
        + P.c21*dT2 + 3.0*P.c23*dT2*dP2
        + P.c31*dT3
    )

    # d/dT̂  (note dΔT̂/dT̂ = 1; the c_1 term gives c_1·ln T̂)
    dval_dT = (
        P.c11*dP + P.c12*dP2 + P.c13*dP3
        + 2.0*P.c20*dT + 2.0*P.c21*dT*dP + 2.0*P.c23*dT*dP3
        + 3.0*P.c30*dT2 + 3.0*P.c31*dT2*dP
        + P.c1 * log_T
    )

    d2val_dP2 = (
        2.0*P.c02 + 6.0*P.c03*dP
        + 2.0*P.c12*dT + 6.0*P.c13*dT*dP
        + 6.0*P.c23*dT2*dP
    )

    d2val_dT2 = (
        2.0*P.c20 + 2.0*P.c21*dP + 2.0*P.c23*dP3
        + 6.0*P.c30*dT + 6.0*P.c31*dT*dP
        + P.c1 / T_hat
    )

    d2val_dPdT = (
        P.c11 + 2.0*P.c12*dP + 3.0*P.c13*dP2
        + 2.0*P.c21*dT + 6.0*P.c23*dT*dP2
        + 3.0*P.c31*dT2
    )

    return val, dval_dP, dval_dT, d2val_dP2, d2val_dT2, d2val_dPdT


def _g_rho_vec(dT, dP, T_hat):
    """Vectorized G_ρ — accepts numpy arrays."""
    dT2 = dT * dT;   dT3 = dT2 * dT
    dP2 = dP * dP;   dP3 = dP2 * dP

    log_T = np.log(T_hat)

    val = (
        P.c01*dP + P.c02*dP2 + P.c03*dP3
        + P.c11*dT*dP + P.c12*dT*dP2 + P.c13*dT*dP3
        + P.c20*dT2 + P.c21*dT2*dP + P.c23*dT2*dP3
        + P.c30*dT3 + P.c31*dT3*dP
        + P.c1 * T_hat * (log_T - 1.0)
    )
    dval_dP = (
        P.c01 + 2.0*P.c02*dP + 3.0*P.c03*dP2
        + P.c11*dT + 2.0*P.c12*dT*dP + 3.0*P.c13*dT*dP2
        + P.c21*dT2 + 3.0*P.c23*dT2*dP2
        + P.c31*dT3
    )
    dval_dT = (
        P.c11*dP + P.c12*dP2 + P.c13*dP3
        + 2.0*P.c20*dT + 2.0*P.c21*dT*dP + 2.0*P.c23*dT*dP3
        + 3.0*P.c30*dT2 + 3.0*P.c31*dT2*dP
        + P.c1 * log_T
    )
    d2val_dP2 = (
        2.0*P.c02 + 6.0*P.c03*dP
        + 2.0*P.c12*dT + 6.0*P.c13*dT*dP
        + 6.0*P.c23*dT2*dP
    )
    d2val_dT2 = (
        2.0*P.c20 + 2.0*P.c21*dP + 2.0*P.c23*dP3
        + 6.0*P.c30*dT + 6.0*P.c31*dT*dP
        + P.c1 / T_hat
    )
    d2val_dPdT = (
        P.c11 + 2.0*P.c12*dP + 3.0*P.c13*dP2
        + 2.0*P.c21*dT + 6.0*P.c23*dT*dP2
        + 3.0*P.c31*dT2
    )
    return val, dval_dP, dval_dT, d2val_dP2, d2val_dT2, d2val_dPdT


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Reduced → physical conversion (vectorized)
# ═══════════════════════════════════════════════════════════════════════════

def _physical_vec(Vh, Sh, d2G_dP2, d2G_dT2, d2G_dPdT, G_hat, T_K):
    """Map reduced second derivatives to SI properties.

    ``Vh, Sh, d²G…`` are per-structural-unit, dimensionless.
    The factor of ``1/n`` here converts to per-molecule before SI scaling.
    """
    # Volume: V̂ is per structural unit (m³ per unit / Vr).
    V_unit = P.Vr * Vh                       # m³ per structural unit
    # Specific volume: V per kg = (V_unit / n_molecules_per_unit) * NA / M
    V_spec = V_unit * P.NA / (P.N_UNIT * P.M_H2O)         # m³/kg
    rho = np.where(V_spec > 0, 1.0 / V_spec, np.inf)      # kg/m³

    # Entropy per kg
    # Sh = -dG/dT̂ (per unit, dimensionless).  Per molecule: × kB / n.
    # Per kg: × R / (n·M)
    S_spec = Sh * P.R / (P.N_UNIT * P.M_H2O)              # J/(kg·K)

    # Heat capacity Cp per kg
    # Ĉ_P = -T̂/n · d²Ĝ/dT̂²   (per molecule, in units of kB)
    # Cp_per_kg = Ĉ_P · R / M
    T_hat = T_K / P.Tr
    Cp = -(T_hat / P.N_UNIT) * d2G_dT2 * P.R / P.M_H2O    # J/(kg·K)

    # Isothermal compressibility
    # κ̂_T = -(1/V̂)·d²Ĝ_total/dP̂²   [per structural unit; dimensionless]
    # κ_T [Pa⁻¹] = κ̂_T · Vr / (kB Tr) = κ̂_T / Pr_Pa
    kappa_T_hat = np.where(np.abs(Vh) > 1e-30, -d2G_dP2 / Vh, np.inf)
    kappa_T = kappa_T_hat / P.Pr_Pa                       # Pa⁻¹
    Kt = np.where(np.isfinite(kappa_T) & (np.abs(kappa_T) > 1e-30),
                  1.0 / kappa_T / 1e6, 0.0)               # MPa

    # Thermal expansion
    # α̂_P = (1/V̂)·d²Ĝ_total/dP̂dT̂   [per unit, dimensionless wrt T̂]
    # α_P [K⁻¹] = α̂_P / Tr
    alpha = np.where(np.abs(Vh) > 1e-30, d2G_dPdT / Vh, 0.0) / P.Tr

    # Cv from thermodynamic identity
    Cv = np.where((kappa_T > 0) & np.isfinite(kappa_T),
                  Cp - T_K * V_spec * alpha**2 / kappa_T,
                  Cp)

    # Adiabatic compressibility κ_S and speed of sound
    kappa_S = np.where(Cp > 0,
                       kappa_T - T_K * V_spec * alpha**2 / Cp,
                       kappa_T)
    Ks = np.where(kappa_S > 0, 1.0 / kappa_S / 1e6, np.inf)
    vel = np.where((rho > 0) & (kappa_S > 0),
                   np.sqrt(1.0 / (rho * kappa_S)), np.nan)

    # Gibbs energy per kg
    G_val = G_hat * P.R * P.Tr / (P.N_UNIT * P.M_H2O)

    return rho, V_spec, S_spec, G_val, Cp, Cv, Kt, Ks, alpha, vel


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Vectorized batch entry point
# ═══════════════════════════════════════════════════════════════════════════

def compute_batch(T_K, P_MPa):
    """Compute all thermodynamic properties on a flat (T, P) batch.

    Parameters
    ----------
    T_K : 1-D array
        Temperatures in K.
    P_MPa : 1-D array
        Pressures in MPa (same length as ``T_K``).

    Returns
    -------
    dict of 1-D arrays
        Keys: ``rho``, ``V``, ``S``, ``G``, ``H``, ``U``, ``A``, ``Cp``,
        ``Cv``, ``Kt``, ``Ks``, ``alpha``, ``vel``, ``x`` (= LFTS fraction
        s) plus ``_A`` (DNLS) and ``_B`` (LFTS) variants of every
        thermodynamic key.
    """
    T_K = np.ascontiguousarray(np.asarray(T_K, dtype=float))
    P_MPa = np.ascontiguousarray(np.asarray(P_MPa, dtype=float))

    # Reduced variables.  T̂ must be > 0 for log T̂ to make sense.
    T_hat = T_K / P.Tr
    P_hat = P_MPa / P.Pr
    dT = T_hat - 1.0
    dP = P_hat - 1.0

    # Safeguard against T ≤ 0 — propagate NaN through the log term.
    T_hat_safe = np.where(T_hat > 0, T_hat, np.nan)

    # ── Background G_ρ ─────────────────────────────────────────────────
    Grho, dGrho_dP, dGrho_dT, d2Grho_dP2, d2Grho_dT2, d2Grho_dPdT = (
        _g_rho_vec(dT, dP, T_hat_safe)
    )

    # ── Two-state piece ────────────────────────────────────────────────
    dE  = P.DELTA_E_HAT
    dS_ = P.DELTA_SIGMA_HAT
    dV  = P.DELTA_V_HAT

    deltaG = dE - T_hat * dS_ + P_hat * dV
    # s = 1 / (1 + exp(deltaG / T̂))    (numerically stable form)
    arg = np.clip(deltaG / T_hat_safe, -700.0, 700.0)
    s = 1.0 / (1.0 + np.exp(arg))

    # ── First derivatives (envelope theorem cancels ∂s/∂… terms) ───────
    Vh = dGrho_dP + s * dV
    EPS = 1e-15
    s_c = np.clip(s, EPS, 1.0 - EPS)
    mix_ent = s_c * np.log(s_c) + (1.0 - s_c) * np.log(1.0 - s_c)
    Sh = -(dGrho_dT - s * dS_ + mix_ent)

    # ── Second derivatives with fluctuation contributions ──────────────
    delta_H = dE + P_hat * dV           # ΔH̃ = ΔÊ + P̂·ΔV̂ (enthalpy-like)
    sfluc = s * (1.0 - s)
    d2G_dP2 = d2Grho_dP2 - sfluc * dV**2 / T_hat_safe
    d2G_dPdT = d2Grho_dPdT + sfluc * dV * delta_H / T_hat_safe**2
    d2G_dT2 = d2Grho_dT2 - sfluc * delta_H**2 / T_hat_safe**3

    # ── Total reduced Gibbs energy ─────────────────────────────────────
    G_hat_mix = Grho + s * deltaG + T_hat * mix_ent
    G_hat_A = Grho                                # s = 0
    G_hat_B = Grho + (dE - T_hat * dS_ + P_hat * dV)   # s = 1, no mix

    # ── Mixture properties ────────────────────────────────────────────
    rho, V, S, G, Cp, Cv, Kt, Ks, alpha, vel = _physical_vec(
        Vh, Sh, d2G_dP2, d2G_dT2, d2G_dPdT, G_hat_mix, T_K)

    # ── State A = DNLS  (s = 0) ───────────────────────────────────────
    Vh_A = dGrho_dP
    Sh_A = -dGrho_dT
    rho_A, V_A, S_A, G_A, Cp_A, Cv_A, Kt_A, Ks_A, alpha_A, vel_A = _physical_vec(
        Vh_A, Sh_A, d2Grho_dP2, d2Grho_dT2, d2Grho_dPdT, G_hat_A, T_K)

    # ── State B = LFTS  (s = 1).  All s(1-s) fluctuation terms vanish; the
    #     ΔĜ contribution is linear in T̂, so it adds to first derivatives
    #     only.
    Vh_B = dGrho_dP + dV
    Sh_B = -(dGrho_dT - dS_)
    rho_B, V_B, S_B, G_B, Cp_B, Cv_B, Kt_B, Ks_B, alpha_B, vel_B = _physical_vec(
        Vh_B, Sh_B, d2Grho_dP2, d2Grho_dT2, d2Grho_dPdT, G_hat_B, T_K)

    # ── IAPWS-95 reference state alignment (no-op until calibrated) ────
    for S_arr, G_arr in [(S, G), (S_A, G_A), (S_B, G_B)]:
        S_arr += P.S_OFFSET
        G_arr += P.H_OFFSET - T_K * P.S_OFFSET

    # ── Derived potentials ─────────────────────────────────────────────
    p_Pa = P_MPa * 1e6
    H   = G   + T_K * S;     U   = H   - p_Pa * V;     A_pot   = G   - p_Pa * V
    H_A = G_A + T_K * S_A;   U_A = H_A - p_Pa * V_A;   A_pot_A = G_A - p_Pa * V_A
    H_B = G_B + T_K * S_B;   U_B = H_B - p_Pa * V_B;   A_pot_B = G_B - p_Pa * V_B

    return {
        'rho': rho, 'V': V, 'S': S, 'G': G, 'H': H, 'U': U, 'A': A_pot,
        'Cp': Cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks, 'alpha': alpha, 'vel': vel,
        'x': s,
        'rho_A': rho_A, 'V_A': V_A, 'S_A': S_A, 'G_A': G_A, 'H_A': H_A,
        'U_A': U_A, 'A_A': A_pot_A, 'Cp_A': Cp_A, 'Cv_A': Cv_A,
        'Kt_A': Kt_A, 'Ks_A': Ks_A, 'alpha_A': alpha_A, 'vel_A': vel_A,
        'rho_B': rho_B, 'V_B': V_B, 'S_B': S_B, 'G_B': G_B, 'H_B': H_B,
        'U_B': U_B, 'A_B': A_pot_B, 'Cp_B': Cp_B, 'Cv_B': Cv_B,
        'Kt_B': Kt_B, 'Ks_B': Ks_B, 'alpha_B': alpha_B, 'vel_B': vel_B,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Scalar entry point  (convenience wrapper around compute_batch)
# ═══════════════════════════════════════════════════════════════════════════

def compute_properties(T_K, P_MPa):
    """Compute all properties at a single (T, P) point.

    Returns a dict with the same keys as :func:`compute_batch` but with
    scalar values instead of 1-D arrays.
    """
    batch = compute_batch(np.array([T_K]), np.array([P_MPa]))
    return {k: float(v[0]) for k, v in batch.items()}
