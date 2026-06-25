"""
Shi & Tanaka (2020) hierarchical transport properties — core engine.

Computes viscosity η, self-diffusion coefficient D, and rotational
relaxation time τ_R from a generalized Arrhenius law (Eq. 5):

    X = x_0 (T/T_r)^λ · exp[(E_a^ρ + s^D · ΔE_a) / (k_B T)]

where E_a^ρ = E_a^0 + P · ΔV_a is the pressure-dependent DNLS activation
energy and s^D is the dynamic order parameter from the hierarchical
extension (Eq. 6):

    s^D = 1 / (1 + exp[(ΔE^D - T Δσ^D + P ΔV^D + P² b) / (k_B T)])

The thermodynamic state (the static LFTS fraction s = ``x`` in waterEoS
parlance) is obtained from the Shi-Tanaka thermodynamic EoS — all
thermodynamic outputs are passed through unchanged so the user gets a
single SeaFreeze-compatible result containing both transport and
thermodynamic properties from a single internally consistent model.
"""

import numpy as np

from . import params as P


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic order parameter  s^D(T, P)
# ═══════════════════════════════════════════════════════════════════════════

def _dynamic_fraction(T_K, P_MPa):
    """Vectorized s^D from Eq. 6 (numerically stable form)."""
    arg = (
        P.DELTA_E_D_K
        - T_K * P.DELTA_SIGMA_D
        + P_MPa * P.DELTA_V_D_MK
        + P_MPa**2 * P.B_M2K
    ) / T_K
    arg = np.clip(arg, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(arg))


# ═══════════════════════════════════════════════════════════════════════════
# Generalized Arrhenius transport property
# ═══════════════════════════════════════════════════════════════════════════

def _arrhenius(T_K, P_MPa, s_D, x0, E_a0_K, dV_a, dE_a_K, lam, sign):
    """One transport property from Eq. 5 — vectorized.

    ``sign`` is +1 for η and τ_R (Arrhenius increases on cooling) and -1 for
    D (inverse-Arrhenius slows down on cooling, so D ∝ exp(-E / k_B T)).
    """
    E_a_rho = E_a0_K + P_MPa * dV_a            # K
    E_total = E_a_rho + s_D * dE_a_K           # K
    arg = sign * E_total / T_K
    arg = np.clip(arg, -700.0, 700.0)
    return x0 * (T_K / P.T_REF) ** lam * np.exp(arg)


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized batch entry point — returns transport + thermo
# ═══════════════════════════════════════════════════════════════════════════

def compute_batch(T_K, P_MPa):
    """Compute transport properties + Shi-Tanaka thermodynamics on a batch.

    Returns a dict containing:

      Transport (this module):
        eta    — dynamic viscosity (Pa·s)
        D      — self-diffusion coefficient (m²/s)
        tau_r  — rotational relaxation time (s)
        f      — dynamic LFTS-cluster fraction s^D  (Eq. 6)

      Thermodynamics (passed through from shi_tanaka_eos.core):
        rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, alpha, vel, x
        plus _A (DNLS) and _B (LFTS) variants of every thermodynamic key.
    """
    from shi_tanaka_eos.core import compute_batch as thermo_batch

    T_K = np.asarray(T_K, dtype=float)
    P_MPa = np.asarray(P_MPa, dtype=float)

    thermo = thermo_batch(T_K, P_MPa)

    s_D = _dynamic_fraction(T_K, P_MPa)

    eta = _arrhenius(T_K, P_MPa, s_D,
                     P.X0_ETA_PAS, P.E_A0_ETA_K, P.DV_A_ETA_MK,
                     P.DE_A_ETA_K, P.LAMBDA_ETA, +1.0)
    D = _arrhenius(T_K, P_MPa, s_D,
                   P.D0_M2_S, P.E_A0_D_K, P.DV_A_D_MK,
                   P.DE_A_D_K, P.LAMBDA_D, -1.0)
    tau_r = _arrhenius(T_K, P_MPa, s_D,
                       P.TAU0_S, P.E_A0_TAU_K, P.DV_A_TAU_MK,
                       P.DE_A_TAU_K, P.LAMBDA_TAU, +1.0)

    result = dict(thermo)
    result['eta'] = eta
    result['D'] = D
    result['tau_r'] = tau_r
    result['f'] = s_D
    return result


def compute_properties(T_K, P_MPa):
    """Scalar entry point — returns a dict with scalar values."""
    batch = compute_batch(np.array([T_K]), np.array([P_MPa]))
    return {k: float(v[0]) for k, v in batch.items()}
