"""
Holten, Sengers & Anisimov (2014) two-state EoS: JAX automatic differentiation engine.

Defines G(T, P, x) once as a JAX-differentiable function. All thermodynamic
properties are computed via jax.grad. Batch computation uses jax.vmap.

Falls back to hand-coded core.py when JAX is not installed.

The total reduced Gibbs energy is:
    g_red = B(tau, pi) + tau * g0(x, L, omega)
where g0 = x*L + x*ln(x) + (1-x)*ln(1-x) + omega*x*(1-x)

Reduced variables:
  Background B: tau = T/Tc, pi = (P_Pa - P0_Pa) / P_scale_Pa
  Field L, omega: t = tau - 1, p_red = (P_Pa - Pc_Pa) / P_scale_Pa
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import params as P
from watereos_ad.equilibrium import make_equilibrium_solver


# ═══════════════════════════════════════════════════════════════════════════
# 1. Gibbs energy sub-functions (JAX, value only)
# ═══════════════════════════════════════════════════════════════════════════

# Pre-convert params to JAX arrays for use in JIT-compiled functions
_c_bg = jnp.array(P.c_bg)
_a_bg = jnp.array(P.a_bg)
_b_bg = jnp.array(P.b_bg)
_d_bg = jnp.array(P.d_bg)


def _B(tau, pi):
    """Background function B(tau, pi) — Eq. 12, Table 7.

    B = sum_i c_i * tau^a_i * pi^b_i * exp(-d_i * pi)
    """
    def term(i):
        return _c_bg[i] * tau**_a_bg[i] * pi**_b_bg[i] * jnp.exp(-_d_bg[i] * pi)
    # Use fori_loop for JAX compatibility
    def body(i, acc):
        return acc + _c_bg[i] * tau**_a_bg[i] * pi**_b_bg[i] * jnp.exp(-_d_bg[i] * pi)
    return jax.lax.fori_loop(0, 20, body, 0.0)


def _L(t, p_red):
    """Hyperbolic field L(t, p_red) — Eq. 14."""
    k0, k1, k2, L0 = P.k0, P.k1, P.k2, P.L0

    arg = p_red - k2 * t
    inner = 1.0 + k0 * k2 + k1 * arg
    K1 = jnp.sqrt(inner**2 - 4.0 * k0 * k1 * k2 * arg)
    K2 = jnp.sqrt(1.0 + k2**2)

    return L0 * K2 * (1.0 - K1 + k0 * k2 + k1 * (p_red + k2 * t)) / (2.0 * k1 * k2)


def _g_mix(x, tau, pi, _unused):
    """
    Total reduced Gibbs energy at composition x.

    g_red = B(tau, pi) + tau * g0
    g0 = x*L + x*ln(x) + (1-x)*ln(1-x) + omega*x*(1-x)

    The 4th argument (_unused) matches the equilibrium solver signature
    (x, arg1, arg2, arg3) but is not used. We derive all variables from
    tau and pi.
    """
    # Reduced variables for field L and omega
    t = tau - 1.0
    # pi = (P_Pa - P0_Pa)/P_scale_Pa, p_red = (P_Pa - Pc_Pa)/P_scale_Pa
    # P_Pa = pi * P_scale_Pa + P0_Pa
    # p_red = (pi * P_scale_Pa + P0_Pa - Pc_Pa) / P_scale_Pa
    #       = pi + (P0_Pa - Pc_Pa) / P_scale_Pa
    p_red = pi + (P.P0 * 1e6 - P.Pc * 1e6) / P.P_scale_Pa

    # Background
    B_val = _B(tau, pi)

    # Field and interaction
    L_val = _L(t, p_red)
    omega = 2.0 + P.omega0 * p_red

    # Mixing part g0
    x_safe = jnp.clip(x, 1e-15, 1.0 - 1e-15)
    mix_ent = x_safe * jnp.log(x_safe) + (1.0 - x_safe) * jnp.log(1.0 - x_safe)
    g0 = x * L_val + mix_ent + omega * x * (1.0 - x)

    return B_val + tau * g0


def g_mix_physical(x, T_K, P_MPa):
    """
    g_mix in physical coordinates for the phase diagram adapter.

    For the phase adapter, we only need the mixing part g0 (not B),
    since the adapter uses dg/dx and B doesn't depend on x.
    """
    tau = T_K / P.Tc
    P_Pa = P_MPa * 1e6
    p_red = (P_Pa - P.Pc * 1e6) / P.P_scale_Pa

    L_val = _L(tau - 1.0, p_red)
    omega = 2.0 + P.omega0 * p_red

    x_safe = jnp.clip(x, 1e-15, 1.0 - 1e-15)
    mix_ent = x_safe * jnp.log(x_safe) + (1.0 - x_safe) * jnp.log(1.0 - x_safe)

    # g0 = x*L + mix_ent + omega*x*(1-x)
    # For phase diagram (equal tangent), we need g0 (the x-dependent part)
    return x * L_val + mix_ent + omega * x * (1.0 - x)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Equilibrium solver
# ═══════════════════════════════════════════════════════════════════════════

# The equilibrium solver expects g_mix(x, arg1, arg2, arg3).
# We use (x, tau, pi, unused) signature.
_solve_x_eq = make_equilibrium_solver(_g_mix)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Total Gibbs functions for property computation
# ═══════════════════════════════════════════════════════════════════════════

def _G_hat_mix(tau, pi, _unused):
    """Total reduced Gibbs energy at equilibrium x."""
    x = _solve_x_eq(tau, pi, _unused)
    return _g_mix(x, tau, pi, _unused)


def _G_hat_A(tau, pi, _unused):
    """State A Gibbs energy (x=0): just B(tau, pi)."""
    return _B(tau, pi)


def _G_hat_B(tau, pi, _unused):
    """State B Gibbs energy (x=1): B + tau*L."""
    t = tau - 1.0
    p_red = pi + (P.P0 * 1e6 - P.Pc * 1e6) / P.P_scale_Pa
    L_val = _L(t, p_red)
    return _B(tau, pi) + tau * L_val


# ═══════════════════════════════════════════════════════════════════════════
# 4. Property computation via AD
# ═══════════════════════════════════════════════════════════════════════════

def _compute_props_from_G_hat(G_hat_3arg, tau, pi, T_K):
    """
    Compute physical properties from a reduced Gibbs function G_hat(tau, pi, unused)
    using AD. Holten-specific unit conversion.

    G_hat is the total reduced Gibbs energy. Derivatives w.r.t. tau and pi
    give all thermodynamic quantities.
    """
    def G2(tau_v, pi_v):
        return G_hat_3arg(tau_v, pi_v, tau_v)  # unused arg = tau (doesn't matter)

    G_val = G2(tau, pi)

    # First derivatives
    dG_dtau = jax.grad(G2, 0)(tau, pi)
    dG_dpi = jax.grad(G2, 1)(tau, pi)

    # Second derivatives
    d2G_dtau2 = jax.grad(jax.grad(G2, 0), 0)(tau, pi)
    d2G_dpi2 = jax.grad(jax.grad(G2, 1), 1)(tau, pi)
    d2G_dtau_dpi = jax.grad(jax.grad(G2, 0), 1)(tau, pi)

    # Holten reduced -> physical conversion
    # v = dG/dpi (reduced volume), s = -dG/dtau (reduced entropy)
    # But we need to be careful: the reduced variables are tau and pi,
    # and the Gibbs energy is g_red = G_hat. Physical properties use
    # the formulas from the Holten model:

    # v (reduced) = Bp + tau * [omega0/2 * (1-f²) + Lp*(f+1)] etc.
    # But with AD, v = dg_red/dpi, s = -dg_red/dtau automatically.
    v = dG_dpi
    s = -dG_dtau

    # kappa (reduced isothermal compressibility, dimensionless)
    kap_dimless = -d2G_dpi2 / v

    # alpha (reduced thermal expansion)
    alp_dimless = d2G_dtau_dpi / v

    # cp (reduced heat capacity)
    cp_dimless = -tau * d2G_dtau2

    # SI conversion (same as core.py)
    rho = P.rho0 / v
    V_spec = 1.0 / rho
    S_val = P.R * s
    Kap = kap_dimless / (P.rho0 * P.R * P.Tc)  # 1/Pa
    Alp = alp_dimless / P.Tc  # 1/K
    Cp_val = P.R * cp_dimless  # J/(kg·K)

    Cv_val = jnp.where(
        (Kap > 0) & jnp.isfinite(Kap),
        Cp_val - T_K * Alp**2 / (rho * Kap),
        Cp_val
    )

    kap_S = jnp.where(
        Cp_val > 0,
        Kap - T_K * V_spec * Alp**2 / Cp_val,
        Kap
    )
    vel = jnp.where(
        (rho > 0) & (kap_S > 0),
        jnp.sqrt(1.0 / (rho * kap_S)),
        jnp.nan
    )

    Kt = jnp.where(
        (Kap > 0) & jnp.isfinite(Kap),
        1.0 / Kap / 1e6,
        jnp.inf
    )
    Ks = jnp.where(kap_S > 0, 1.0 / kap_S / 1e6, jnp.inf)

    G_phys = P.R * P.Tc * G_val  # J/kg

    return {
        'rho': rho, 'V': V_spec, 'S': S_val,
        'Cp': Cp_val, 'Cv': Cv_val, 'Kt': Kt, 'Ks': Ks,
        'alpha': Alp, 'vel': vel, 'G': G_phys,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main entry points
# ═══════════════════════════════════════════════════════════════════════════

def _compute_properties_scalar(T_K, p_MPa):
    """Compute all properties at a single (T, P) point (JAX scalars)."""
    P_Pa = p_MPa * 1e6
    tau = T_K / P.Tc
    pi = (P_Pa - P.P0 * 1e6) / P.P_scale_Pa

    # Mixture (equilibrium x)
    mix = _compute_props_from_G_hat(_G_hat_mix, tau, pi, T_K)
    x_eq = _solve_x_eq(tau, pi, tau)
    mix['x'] = x_eq

    # State A (x=0)
    stateA = _compute_props_from_G_hat(_G_hat_A, tau, pi, T_K)

    # State B (x=1)
    stateB = _compute_props_from_G_hat(_G_hat_B, tau, pi, T_K)

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
    p_Pa_val = p_MPa * 1e6
    for suffix in ['', '_A', '_B']:
        G = result['G' + suffix]
        S = result['S' + suffix]
        V = result['V' + suffix]
        result['H' + suffix] = G + T_K * S
        result['U' + suffix] = G + T_K * S - p_Pa_val * V
        result['A' + suffix] = G - p_Pa_val * V

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

    if _compute_Kp:
        dp = 0.001
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
    Compute properties at given (T, p) with a forced x value.

    First derivatives (V, S) are partial at constant x.
    Second derivatives (Cp, Kt, etc.) are total derivatives including
    implicit dx/dT, dx/dP from the equilibrium condition.
    """
    T_jax = jnp.float64(T_K)
    p_jax = jnp.float64(p_MPa)
    x_jax = jnp.float64(x)

    P_Pa = p_jax * 1e6
    tau = T_jax / P.Tc
    pi = (P_Pa - P.P0 * 1e6) / P.P_scale_Pa

    # 3-arg g_mix for AD: g(x, tau, pi)
    def _g3(x_v, tau_v, pi_v):
        return _g_mix(x_v, tau_v, pi_v, tau_v)

    # Partial derivatives w.r.t. tau and pi at fixed x
    _dg_dpi = jax.grad(_g3, argnums=2)
    _dg_dtau = jax.grad(_g3, argnums=1)

    # Build forced solver with implicit function theorem JVP
    dg_dx = jax.grad(_g3, argnums=0)
    d2g_dx2 = jax.grad(dg_dx, argnums=0)
    d2g_dxdtau = jax.grad(dg_dx, argnums=1)
    d2g_dxdpi = jax.grad(dg_dx, argnums=2)

    @jax.custom_jvp
    def solve_forced(tau_v, pi_v):
        return x_jax

    @solve_forced.defjvp
    def solve_forced_jvp(primals, tangents):
        tau_v, pi_v = primals
        dtau, dpi = tangents
        gxx = d2g_dx2(x_jax, tau_v, pi_v)
        gxx_safe = jnp.where(jnp.abs(gxx) > 1e-30, gxx, jnp.sign(gxx + 1e-40) * 1e-30)
        gxt = d2g_dxdtau(x_jax, tau_v, pi_v)
        gxp = d2g_dxdpi(x_jax, tau_v, pi_v)
        dx = -(gxt * dtau + gxp * dpi) / gxx_safe
        return x_jax, dx

    # "Total" first-derivative functions
    def _v_total(tau_v, pi_v):
        x_v = solve_forced(tau_v, pi_v)
        return _dg_dpi(x_v, tau_v, pi_v)

    def _dG_dtau_total(tau_v, pi_v):
        x_v = solve_forced(tau_v, pi_v)
        return _dg_dtau(x_v, tau_v, pi_v)

    # Value
    G_val = _g3(x_jax, tau, pi)

    # First derivatives (partial at fixed x)
    v = _v_total(tau, pi)
    s = -_dG_dtau_total(tau, pi)

    # Second derivatives (total, including dx/dtau, dx/dpi)
    d2G_dpi2 = jax.grad(_v_total, 1)(tau, pi)
    d2G_dtau2 = jax.grad(_dG_dtau_total, 0)(tau, pi)
    d2G_dtau_dpi = jax.grad(_v_total, 0)(tau, pi)

    # Holten reduced -> physical
    kap_dimless = -d2G_dpi2 / v
    alp_dimless = d2G_dtau_dpi / v
    cp_dimless = -tau * d2G_dtau2

    rho = P.rho0 / v
    V_spec = 1.0 / rho
    S_val = P.R * s
    Kap = kap_dimless / (P.rho0 * P.R * P.Tc)
    Alp = alp_dimless / P.Tc
    Cp_val = P.R * cp_dimless

    Cv_val = jnp.where(
        (Kap > 0) & jnp.isfinite(Kap),
        Cp_val - T_jax * Alp**2 / (rho * Kap), Cp_val)
    kap_S = jnp.where(Cp_val > 0,
                       Kap - T_jax * V_spec * Alp**2 / Cp_val, Kap)
    vel = jnp.where((rho > 0) & (kap_S > 0),
                     jnp.sqrt(1.0 / (rho * kap_S)), jnp.nan)
    Kt = jnp.where((Kap > 0) & jnp.isfinite(Kap), 1.0 / Kap / 1e6, jnp.inf)
    Ks = jnp.where(kap_S > 0, 1.0 / kap_S / 1e6, jnp.inf)
    G_phys = P.R * P.Tc * G_val

    props = {
        'rho': float(rho), 'V': float(V_spec), 'S': float(S_val),
        'Cp': float(Cp_val), 'Cv': float(Cv_val), 'Kt': float(Kt),
        'Ks': float(Ks), 'alpha': float(Alp), 'vel': float(vel),
        'G': float(G_phys),
    }
    props['x'] = float(x)

    # IAPWS-95 alignment
    props['S'] += P.S_OFFSET
    props['G'] += P.H_OFFSET - T_K * P.S_OFFSET

    # Derived potentials
    p_Pa_val = p_MPa * 1e6
    G = props['G']
    S = props['S']
    V = props['V']
    props['H'] = G + T_K * S
    props['U'] = G + T_K * S - p_Pa_val * V
    props['A'] = G - p_Pa_val * V

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

    return {k: np.asarray(v) for k, v in result_jax.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 7. G_hat_mix for TMD/Widom solvers
# ═══════════════════════════════════════════════════════════════════════════

def _G_hat_mix_3arg(dTh, dPh, T_K):
    """
    G_hat(dTh, dPh, T_K) for TMD/Widom solvers.

    dTh and dPh use the Caupin-style convention:
      dTh = (T_K - Tc) / Tc = tau - 1
      dPh = (P_MPa - Pc) / P_scale_MPa

    We convert to Holten's tau/pi internally.
    """
    tau = 1.0 + dTh
    # dPh = (P_MPa - Pc) / P_scale_MPa
    # P_MPa = Pc + dPh * P_scale_MPa
    # P_Pa = (Pc + dPh * P_scale_MPa) * 1e6
    # pi = (P_Pa - P0*1e6) / P_scale_Pa
    P_MPa = P.Pc + dPh * P.P_scale_MPa
    P_Pa = P_MPa * 1e6
    pi = (P_Pa - P.P0 * 1e6) / P.P_scale_Pa

    x = _solve_x_eq(tau, pi, tau)
    return _g_mix(x, tau, pi, tau)
