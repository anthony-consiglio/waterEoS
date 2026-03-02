"""
Caupin & Anisimov (2019) two-state EoS: JAX automatic differentiation engine.

Defines G(T, P, x) once as a JAX-differentiable function. All thermodynamic
properties are computed via jax.grad. Batch computation uses jax.vmap.

Falls back to hand-coded core.py when JAX is not installed.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import params as P
from watereos_ad.equilibrium import make_equilibrium_solver
from watereos_ad.properties import compute_properties_ad


# ═══════════════════════════════════════════════════════════════════════════
# 1. Gibbs energy sub-functions (JAX, value only — no hand-coded derivatives)
# ═══════════════════════════════════════════════════════════════════════════

def _spinodal_pressure(T_K):
    """Spinodal pressure Ps(T) in MPa (Eq. 2)."""
    dT = T_K - P.ps_T0
    return P.ps_a + P.ps_b * dT + P.ps_c * dT**2


def _G_sigma(dTh, dPh, T_K):
    """
    Spinodal Gibbs contribution: Ĝ^σ = Â(T)·u^{3/2} (Eqs. 1, 8).
    u = P̂ - P̂_s, clamped to a small positive value near the spinodal.
    """
    Ah = P.A0 + P.A1 * dTh

    Ps = _spinodal_pressure(T_K)
    dPhs = (P.Pc - Ps) / P.P_scale_MPa
    u = dPh + dPhs

    # Soft clamp: use jnp.maximum for differentiability
    u_safe = jnp.maximum(u, 1e-30)
    return Ah * u_safe ** 1.5


def _G_A_poly(dTh, dPh):
    """Polynomial part of Ĝ^A: Σ c_mn · ΔT̂^m · ΔP̂^n (Eq. 6)."""
    T, Q = dTh, dPh
    T2 = T**2; T3 = T**3; T4 = T**4
    Q2 = Q**2; Q3 = Q**3; Q4 = Q**4

    return (P.c01*Q + P.c02*Q2 + P.c11*T*Q + P.c20*T2
            + P.c03*Q3 + P.c12*T*Q2 + P.c21*T2*Q + P.c30*T3
            + P.c04*Q4 + P.c13*T*Q3 + P.c22*T2*Q2 + P.c40*T4
            + P.c14*T*Q4)


def _G_A(dTh, dPh, T_K):
    """Full state-A Gibbs energy: Ĝ^A = Ĝ^σ + polynomial."""
    return _G_sigma(dTh, dPh, T_K) + _G_A_poly(dTh, dPh)


def _G_BA(dTh, dPh):
    """State B−A Gibbs difference: Ĝ^BA (Eq. 7)."""
    return P.lam * (dTh + P.a*dPh + P.b*dTh*dPh + P.d*dPh**2 + P.f*dTh**2)


def _omega_hat(dTh, dPh):
    """Interaction parameter: ω̂ = (2 + ω₀·ΔP̂) / T̂ (Eq. 5)."""
    Th = 1.0 + dTh
    return (2.0 + P.omega0 * dPh) / Th


def _g_mix(x, dTh, dPh, T_K):
    """
    Total reduced Gibbs energy of mixing at composition x.

    Ĝ_mix = Ĝ^A + x·Ĝ^BA + T̂·[x ln x + (1-x) ln(1-x)] + Ω·x·(1-x)

    where Ω = T̂·ω̂ = 2 + ω₀·ΔP̂.
    """
    Th = 1.0 + dTh
    GA = _G_A(dTh, dPh, T_K)
    GBA = _G_BA(dTh, dPh)
    om = _omega_hat(dTh, dPh)
    Om = Th * om  # = 2 + omega0 * dPh

    # Entropy of mixing with safe log
    x_safe = jnp.clip(x, 1e-15, 1.0 - 1e-15)
    mix_ent = x_safe * jnp.log(x_safe) + (1.0 - x_safe) * jnp.log(1.0 - x_safe)

    return GA + x * GBA + Th * mix_ent + Om * x * (1.0 - x)


def g_mix_physical(x, T_K, P_MPa):
    """
    g_mix in physical coordinates for the phase diagram adapter.

    Parameters
    ----------
    x : scalar — composition (fraction of state B)
    T_K : scalar — temperature in K
    P_MPa : scalar — pressure in MPa

    Returns
    -------
    scalar — reduced Gibbs mixing energy
    """
    dTh = (T_K - P.Tc) / P.Tc
    dPh = (P_MPa - P.Pc) / P.P_scale_MPa
    return _g_mix(x, dTh, dPh, T_K)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Equilibrium solver
# ═══════════════════════════════════════════════════════════════════════════

_solve_x_eq = make_equilibrium_solver(_g_mix)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Total Gibbs functions for property computation
# ═══════════════════════════════════════════════════════════════════════════

def _G_hat_mix(dTh, dPh, T_K):
    """Total reduced Gibbs energy at equilibrium x."""
    x = _solve_x_eq(dTh, dPh, T_K)
    return _g_mix(x, dTh, dPh, T_K)


def _G_hat_A(dTh, dPh, T_K):
    """State A Gibbs energy (x=0, mixing terms vanish)."""
    return _G_A(dTh, dPh, T_K)


def _G_hat_B(dTh, dPh, T_K):
    """State B Gibbs energy (x=1, mixing terms vanish)."""
    return _G_A(dTh, dPh, T_K) + _G_BA(dTh, dPh)


def _g_mix_3arg(x, dTh, dPh):
    """g_mix with T_K derived from dTh (3-argument version for AD).

    This ensures that jax.grad w.r.t. dTh captures the T_K = Tc*(1+dTh)
    dependency through the spinodal pressure.
    """
    T_K = P.Tc * (1.0 + dTh)
    return _g_mix(x, dTh, dPh, T_K)


def _make_forced_x_solver_3arg(x_forced):
    """
    Build a fake equilibrium solver for the 3-arg g_mix(x, dTh, dPh).

    Returns x_forced in the forward pass. JVP uses the implicit function
    theorem: dx/dTh = -(d²g/dxdTh) / (d²g/dx²), dx/dPh = -(d²g/dxdPh) / (d²g/dx²).
    """
    dg_dx = jax.grad(_g_mix_3arg, argnums=0)
    d2g_dx2 = jax.grad(dg_dx, argnums=0)
    d2g_dxdTh = jax.grad(dg_dx, argnums=1)
    d2g_dxdPh = jax.grad(dg_dx, argnums=2)

    @jax.custom_jvp
    def solve_x_forced(dTh, dPh):
        return x_forced

    @solve_x_forced.defjvp
    def solve_x_forced_jvp(primals, tangents):
        dTh, dPh = primals
        dTh_dot, dPh_dot = tangents

        gxx = d2g_dx2(x_forced, dTh, dPh)
        gxx_safe = jnp.where(jnp.abs(gxx) > 1e-30, gxx, jnp.sign(gxx + 1e-40) * 1e-30)
        gxT = d2g_dxdTh(x_forced, dTh, dPh)
        gxP = d2g_dxdPh(x_forced, dTh, dPh)

        dx = -(gxT * dTh_dot + gxP * dPh_dot) / gxx_safe
        return x_forced, dx

    return solve_x_forced


# ═══════════════════════════════════════════════════════════════════════════
# 4. Property computation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_G_hat_2arg(G_hat_3arg, T_K):
    """
    Create a 2-argument G_hat(dTh, dPh) from a 3-argument G_hat(dTh, dPh, T_K).

    T_K is derived from dTh as T_K = Tc*(1+dTh) so that AD correctly captures
    the T_K dependence (e.g. spinodal pressure) when differentiating w.r.t. dTh.
    """
    def G_hat(dTh, dPh):
        T_K_derived = P.Tc * (1.0 + dTh)
        return G_hat_3arg(dTh, dPh, T_K_derived)
    return G_hat


def _compute_state_props(G_hat_3arg, dTh, dPh, T_K):
    """Compute properties for a state (mix, A, or B) using AD."""
    G_hat_2arg = _make_G_hat_2arg(G_hat_3arg, T_K)
    return compute_properties_ad(G_hat_2arg, dTh, dPh, T_K, P)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main entry points
# ═══════════════════════════════════════════════════════════════════════════

def _compute_properties_scalar(T_K, p_MPa):
    """Compute all properties at a single (T, P) point (JAX scalars)."""
    dTh = (T_K - P.Tc) / P.Tc
    dPh = (p_MPa - P.Pc) / P.P_scale_MPa

    # Mixture (equilibrium x)
    mix = _compute_state_props(_G_hat_mix, dTh, dPh, T_K)
    x_eq = _solve_x_eq(dTh, dPh, T_K)
    mix['x'] = x_eq

    # State A (x=0)
    stateA = _compute_state_props(_G_hat_A, dTh, dPh, T_K)

    # State B (x=1)
    stateB = _compute_state_props(_G_hat_B, dTh, dPh, T_K)

    # Assemble result
    result = {}
    for key, val in mix.items():
        result[key] = val
    for key, val in stateA.items():
        result[key + '_A'] = val
    for key, val in stateB.items():
        result[key + '_B'] = val

    # IAPWS-95 reference state alignment
    for suffix in ['', '_A', '_B']:
        result['S' + suffix] = result['S' + suffix] + P.S_OFFSET
        result['G' + suffix] = result['G' + suffix] + P.H_OFFSET - T_K * P.S_OFFSET

    # Derived thermodynamic potentials (H, U, A)
    p_Pa = p_MPa * 1e6
    for suffix in ['', '_A', '_B']:
        G = result['G' + suffix]
        S = result['S' + suffix]
        V = result['V' + suffix]
        result['H' + suffix] = G + T_K * S
        result['U' + suffix] = G + T_K * S - p_Pa * V
        result['A' + suffix] = G - p_Pa * V

    return result


def compute_properties(T_K, p_MPa, _compute_Kp=False):
    """
    Compute all thermodynamic properties at a single (T, p) point.

    Parameters
    ----------
    T_K : float — temperature in K
    p_MPa : float — pressure in MPa

    Returns
    -------
    dict : properties for mixture, state A (suffix _A), state B (suffix _B)
    """
    T_jax = jnp.float64(T_K)
    p_jax = jnp.float64(p_MPa)

    result_jax = _compute_scalar_jit(T_jax, p_jax)

    # Convert JAX arrays to Python floats
    result = {k: float(v) for k, v in result_jax.items()}

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
    Compute thermodynamic properties at a given (T, p) with a forced x value.

    Matches core.py behavior:
    - First derivatives (V, S) are partial derivatives at constant x
    - Second derivatives (Cp, Kt, etc.) are total derivatives of the partial
      first derivatives, including implicit dx/dT, dx/dP from the equilibrium
      condition (physically: response functions reflect x adjusting to T,P changes)
    """
    T_jax = jnp.float64(T_K)
    p_jax = jnp.float64(p_MPa)
    x_jax = jnp.float64(x)

    dTh = (T_jax - P.Tc) / P.Tc
    dPh = (p_jax - P.Pc) / P.P_scale_MPa

    # Partial derivatives of the 3-arg g_mix (T_K derived from dTh)
    _dg_dPh = jax.grad(_g_mix_3arg, argnums=2)   # ∂g/∂dPh at fixed (x, dTh)
    _dg_dTh = jax.grad(_g_mix_3arg, argnums=1)   # dg/dTh including T_K chain rule

    # Build forced solver for 3-arg g_mix
    _solve_forced = _make_forced_x_solver_3arg(x_jax)

    # "Total" first-derivative functions: partials composed with solver.
    # Function values = partial derivatives at fixed x.
    # Their gradients = second derivatives including dx/dTh, dx/dPh.
    def _Vh_total(dTh_v, dPh_v):
        x_v = _solve_forced(dTh_v, dPh_v)
        return _dg_dPh(x_v, dTh_v, dPh_v)

    def _dG_dTh_total(dTh_v, dPh_v):
        x_v = _solve_forced(dTh_v, dPh_v)
        return _dg_dTh(x_v, dTh_v, dPh_v)

    # Value: g_mix at forced x
    G_val = _g_mix_3arg(x_jax, dTh, dPh)

    # First derivatives: partial at fixed x
    Vh = _Vh_total(dTh, dPh)
    Sh_red = -_dG_dTh_total(dTh, dPh)

    # Second derivatives: differentiate the "total" first-derivative functions
    # d²G/dP² = d(Vh_total)/dPh
    d2G_dP2 = jax.grad(_Vh_total, 1)(dTh, dPh)
    # d²G/dT² = d(dG_dTh_total)/dTh
    d2G_dT2 = jax.grad(_dG_dTh_total, 0)(dTh, dPh)
    # d²G/dPdT = d(Vh_total)/dTh
    d2G_dPdT = jax.grad(_Vh_total, 0)(dTh, dPh)

    # Convert to physical properties
    Th = T_jax / P.Tc
    V_molar = P.Vc * Vh
    V_spec = V_molar / P.M_H2O
    rho = jnp.where(V_spec > 0, 1.0 / V_spec, jnp.inf)

    S_spec = P.R * Sh_red / P.M_H2O
    Cp = -P.R * Th * d2G_dT2 / P.M_H2O

    kappa_T = jnp.where(jnp.abs(Vh) > 1e-30,
                         -(P.Vc / (P.R * P.Tc)) * d2G_dP2 / Vh, jnp.inf)
    Kt = jnp.where((jnp.abs(kappa_T) > 1e-30) & jnp.isfinite(kappa_T),
                    1.0 / kappa_T / 1e6, 0.0)

    alpha = jnp.where(jnp.abs(Vh) > 1e-30,
                       (1.0 / P.Tc) * d2G_dPdT / Vh, 0.0)

    Cv = jnp.where((kappa_T > 0) & jnp.isfinite(kappa_T),
                    Cp - T_jax * V_spec * alpha**2 / kappa_T, Cp)

    kappa_S = jnp.where(Cp > 0,
                         kappa_T - T_jax * V_spec * alpha**2 / Cp, kappa_T)
    Ks = jnp.where(kappa_S > 0, 1.0 / kappa_S / 1e6, jnp.inf)
    vel = jnp.where((rho > 0) & (kappa_S > 0),
                     jnp.sqrt(1.0 / (rho * kappa_S)), jnp.nan)

    G_phys = P.R * P.Tc * G_val / P.M_H2O

    props = {
        'rho': float(rho), 'V': float(V_spec), 'S': float(S_spec),
        'Cp': float(Cp), 'Cv': float(Cv), 'Kt': float(Kt), 'Ks': float(Ks),
        'alpha': float(alpha), 'vel': float(vel), 'G': float(G_phys),
    }
    props['x'] = float(x)

    # IAPWS-95 alignment
    props['S'] += P.S_OFFSET
    props['G'] += P.H_OFFSET - T_K * P.S_OFFSET

    # Derived potentials
    p_Pa = p_MPa * 1e6
    G = props['G']
    S = props['S']
    V = props['V']
    props['H'] = G + T_K * S
    props['U'] = G + T_K * S - p_Pa * V
    props['A'] = G - p_Pa * V

    if _compute_Kp:
        dp = 0.001
        pp = compute_properties_at_x(T_K, p_MPa + dp, x)
        pm = compute_properties_at_x(T_K, p_MPa - dp, x)
        props['Kp'] = (pp['Kt'] - pm['Kt']) / (2.0 * dp)

    return props


# ═══════════════════════════════════════════════════════════════════════════
# 6. Vectorized batch computation via vmap
# ═══════════════════════════════════════════════════════════════════════════

_compute_scalar_jit = jax.jit(_compute_properties_scalar)
_compute_batch_vmap = jax.jit(jax.vmap(_compute_properties_scalar, in_axes=(0, 0)))


def compute_batch(T_K, p_MPa):
    """
    Vectorized computation of all thermodynamic properties.

    Parameters
    ----------
    T_K   : 1-D array — temperature in K
    p_MPa : 1-D array — pressure in MPa

    Returns
    -------
    dict of 1-D arrays matching core.py compute_batch output format.
    """
    T_arr = jnp.asarray(T_K, dtype=jnp.float64)
    p_arr = jnp.asarray(p_MPa, dtype=jnp.float64)

    result_jax = _compute_batch_vmap(T_arr, p_arr)

    # Convert to numpy arrays
    result = {k: np.asarray(v) for k, v in result_jax.items()}
    return result
