"""
EOS-VaT (Duška 2020) two-state EoS: JAX automatic differentiation engine.

Defines G(T, P, x) once as a JAX-differentiable function. All thermodynamic
properties are computed via jax.grad. Batch computation uses jax.vmap.

Falls back to hand-coded core.py when JAX is not installed.

The total reduced Gibbs energy of mixing is:
    g_mix = g_A(ph, Th) + x*DG + Th*[x*ln(x) + (1-x)*ln(1-x)] + omega*x*(1-x)

where g_A is computed via Gauss-Legendre path integral, DG is a polynomial,
and omega depends on both Th and ph.

Reduced variables: Th = T/T_VLCP, ph = P/p_VLCP
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import params as P
from watereos_ad.equilibrium import make_equilibrium_solver


# ═══════════════════════════════════════════════════════════════════════════
# 0. Gauss-Legendre nodes/weights (precomputed, 16 points)
# ═══════════════════════════════════════════════════════════════════════════

_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(16)
_GL_NODES = jnp.array(_GL_NODES_NP)
_GL_WEIGHTS = jnp.array(_GL_WEIGHTS_NP)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Spinodal polynomials (Eqs. 6a-c)
# ═══════════════════════════════════════════════════════════════════════════

def _pS(Th):
    """Spinodal pressure: pS(Th) = 1 + d1*(Th-1) + d2*(Th-1)^2 + d3*(Th-1)^3"""
    dTh = Th - 1.0
    return 1.0 + P.d[0]*dTh + P.d[1]*dTh**2 + P.d[2]*dTh**3


def _VS(Th):
    """Spinodal volume: VS(Th) = c0 + c1*Th + c2*Th^2 + c3*Th^3 + c4*Th^4"""
    return P.c[0] + P.c[1]*Th + P.c[2]*Th**2 + P.c[3]*Th**3 + P.c[4]*Th**4


def _phi(Th):
    """Second derivative of pressure wrt density at spinodal."""
    return P.b[0] + P.b[1]*Th + P.b[2]*Th**2 + P.b[3]*Th**3 + P.b[4]*Th**4


# ═══════════════════════════════════════════════════════════════════════════
# 2. B parameter (Eq. 14-16)
# ═══════════════════════════════════════════════════════════════════════════

@jax.custom_jvp
def _sqrt_abs(x):
    """sqrt(|x|) with gradient convention matching hand-coded Duska model.

    The hand-coded version computes B = sqrt(abs(arg)) and dB = darg/(2*B),
    which is d/dx[sqrt(x)] regardless of sign. We replicate this convention
    so that SA and g_A match the hand-coded values exactly.
    """
    return jnp.sqrt(jnp.abs(x))


@_sqrt_abs.defjvp
def _sqrt_abs_jvp(primals, tangents):
    x, = primals
    dx, = tangents
    y = jnp.sqrt(jnp.abs(x))
    # Convention: always use dx/(2*sqrt(|x|)), matching hand-coded dB = darg/(2*B)
    dy = dx / (2.0 * jnp.where(y > 1e-30, y, 1e-30))
    return y, dy


def _B_param(Th):
    """B = sqrt(|-phi / (2 * pS * VS^2)|)

    Uses _sqrt_abs with custom gradient to match the hand-coded derivative
    convention: dB = darg/(2*B) regardless of the sign of arg.
    """
    pS = _pS(Th)
    VS = _VS(Th)
    phi_val = _phi(Th)
    arg = -phi_val / (2.0 * pS * VS**2)
    return _sqrt_abs(arg)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Volume of state A (Eqs. 17-18)
# ═══════════════════════════════════════════════════════════════════════════

def _VA(ph, Th):
    """Reduced volume of state A: VA = VS*B / (u + B) where u = sqrt(max(1-ph/pS, 0))"""
    pS = _pS(Th)
    VS = _VS(Th)
    B = _B_param(Th)
    u = jnp.sqrt(jnp.maximum(1.0 - ph / pS, 1e-30))
    return VS * B / (u + B)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Entropy of state A (Eqs. 19-26)
# ═══════════════════════════════════════════════════════════════════════════

def _SS(Th):
    """Entropy at the spinodal: integral of SS'(Th) = s0*ln(Th) + s1*Th + s2/2*Th^2 + s3/3*Th^3"""
    return (P.s[0] * jnp.log(Th)
            + P.s[1] * Th
            + P.s[2] / 2.0 * Th**2
            + P.s[3] / 3.0 * Th**3)


def _SA(ph, Th):
    """Reduced entropy of state A (Eq. 20)."""
    pS = _pS(Th)
    VS = _VS(Th)
    phi_val = _phi(Th)
    B = _B_param(Th)

    # Derivatives via AD
    dpS = jax.grad(_pS)(Th)
    dVS = jax.grad(_VS)(Th)
    dphi = jax.grad(_phi)(Th)
    dB = jax.grad(_B_param)(Th)

    VA = _VA(ph, Th)

    # Auxiliary coefficients
    A_coeff = dpS * B**2 + 2.0 * pS * B * dB
    C_coeff = 2.0 * pS * B**2 * dVS

    SS = _SS(Th)

    r = VA / VS
    ln_r = jnp.log(jnp.maximum(r, 1e-300))

    SA = (dpS * (VA - VS)
          + A_coeff * (VS**2 / VA + 2.0 * VS * ln_r - VA)
          + C_coeff * (VS / VA + ln_r - 1.0)
          + SS)
    return SA


# ═══════════════════════════════════════════════════════════════════════════
# 5. Gibbs energy of state A via path integral
# ═══════════════════════════════════════════════════════════════════════════

@jax.custom_jvp
def _g_A(ph, Th):
    """
    Reduced Gibbs energy of state A via 2D path integral.

    Leg 1: (0, 1) -> (0, Th), integrating -SA(0, Th') dTh'
    Leg 2: (0, Th) -> (ph, Th), integrating VA(ph', Th) dph'

    Uses custom_jvp to provide exact analytical derivatives:
      dg_A/dph = VA(ph, Th)     (exact, not quadrature-limited)
      dg_A/dTh = -SA(ph, Th)   (exact, not quadrature-limited)

    This avoids quadrature error in the derivatives while keeping
    the integral for the Gibbs energy value itself.
    """
    # Leg 1: T-integral from Th=1 to Th at ph=0
    dTh_val = Th - 1.0
    half_dTh = dTh_val / 2.0

    def T_integrand(i, acc):
        node = _GL_NODES[i]
        weight = _GL_WEIGHTS[i]
        Th_i = half_dTh * (node + 1.0) + 1.0
        SA_i = _SA(0.0, Th_i)
        return acc + weight * (-SA_i)

    g_T_leg = jax.lax.fori_loop(0, 16, T_integrand, 0.0) * half_dTh

    # Leg 2: P-integral from ph=0 to ph at fixed Th
    half_ph = ph / 2.0

    def P_integrand(i, acc):
        node = _GL_NODES[i]
        weight = _GL_WEIGHTS[i]
        ph_i = half_ph * (node + 1.0)
        VA_i = _VA(ph_i, Th)
        return acc + weight * VA_i

    g_P_leg = jax.lax.fori_loop(0, 16, P_integrand, 0.0) * half_ph

    return g_T_leg + g_P_leg


@_g_A.defjvp
def _g_A_jvp(primals, tangents):
    ph, Th = primals
    dph, dTh = tangents

    # Forward pass: compute g_A value
    g_A_val = _g_A(ph, Th)

    # Exact derivatives from thermodynamic identities:
    #   dg_A/dph = VA(ph, Th)
    #   dg_A/dTh = -SA(ph, Th)
    # These are computed from the analytical SA/VA functions, avoiding
    # the quadrature error that would come from differentiating through
    # the Gauss-Legendre integral. Higher-order derivatives (d²g_A/dTh²
    # etc.) are obtained by JAX differentiating through SA and VA, which
    # are fully analytical.
    VA_val = _VA(ph, Th)
    SA_val = _SA(ph, Th)

    dg_A = VA_val * dph + (-SA_val) * dTh
    return g_A_val, dg_A


# ═══════════════════════════════════════════════════════════════════════════
# 6. DeltaG and omega
# ═══════════════════════════════════════════════════════════════════════════

def _DeltaG(ph, Th):
    """G^B - G^A Gibbs difference polynomial (Eq. 5)."""
    return (P.a[0] + P.a[1]*ph*Th + P.a[2]*ph + P.a[3]*Th
            + P.a[4]*Th**2 + P.a[5]*ph**2 + P.a[6]*ph**3)


def _omega(ph, Th):
    """Cooperativity parameter (Eq. 33-35)."""
    return P.w[0] * (1.0 + P.w[1]*ph + P.w[2]*Th + P.w[3]*Th*ph)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Total Gibbs energy of mixing
# ═══════════════════════════════════════════════════════════════════════════

def _g_mix(x, Th, ph, _unused):
    """
    Total reduced Gibbs energy of mixing at composition x.

    g_mix = g_A(ph, Th) + x*DG + Th*[x*ln(x) + (1-x)*ln(1-x)] + omega*x*(1-x)

    The 4th argument is unused (matches equilibrium solver signature).
    """
    gA = _g_A(ph, Th)
    DG = _DeltaG(ph, Th)
    om = _omega(ph, Th)

    x_safe = jnp.clip(x, 1e-15, 1.0 - 1e-15)
    mix_ent = x_safe * jnp.log(x_safe) + (1.0 - x_safe) * jnp.log(1.0 - x_safe)

    return gA + x * DG + Th * mix_ent + om * x * (1.0 - x)


def g_mix_physical(x, T_K, P_MPa):
    """
    g_mix in physical coordinates for the phase diagram adapter.

    Only the x-dependent mixing part (g_A cancels in dg/dx).
    """
    Th = T_K / P.T_VLCP
    ph = P_MPa / P.p_VLCP

    DG = _DeltaG(ph, Th)
    om = _omega(ph, Th)

    x_safe = jnp.clip(x, 1e-15, 1.0 - 1e-15)
    mix_ent = x_safe * jnp.log(x_safe) + (1.0 - x_safe) * jnp.log(1.0 - x_safe)

    return x * DG + Th * mix_ent + om * x * (1.0 - x)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Equilibrium solver
# ═══════════════════════════════════════════════════════════════════════════

# Solver expects g_mix(x, arg1, arg2, arg3) = g_mix(x, Th, ph, unused)
_solve_x_eq = make_equilibrium_solver(_g_mix)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Total Gibbs functions for property computation
# ═══════════════════════════════════════════════════════════════════════════

def _G_hat_mix(Th, ph, _unused):
    """Total reduced Gibbs energy at equilibrium x."""
    x = _solve_x_eq(Th, ph, _unused)
    return _g_mix(x, Th, ph, _unused)


def _G_hat_A(Th, ph, _unused):
    """State A Gibbs energy (x=0): just g_A."""
    return _g_A(ph, Th)


def _G_hat_B(Th, ph, _unused):
    """State B Gibbs energy (x=1): g_A + DG."""
    return _g_A(ph, Th) + _DeltaG(ph, Th)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Property computation via AD (Duska-specific units)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_props_from_G_hat(G_hat_3arg, Th, ph, T_K):
    """
    Compute physical properties from a reduced Gibbs function G_hat(Th, ph, unused)
    using AD. Duska-specific unit conversion.

    Reduced variables: Th = T/T_VLCP, ph = P/p_VLCP
    g_red = G_hat (dimensionless)
    """
    def G2(Th_v, ph_v):
        return G_hat_3arg(Th_v, ph_v, Th_v)

    G_val = G2(Th, ph)

    # First derivatives
    dG_dTh = jax.grad(G2, 0)(Th, ph)
    dG_dph = jax.grad(G2, 1)(Th, ph)

    # Second derivatives
    d2G_dTh2 = jax.grad(jax.grad(G2, 0), 0)(Th, ph)
    d2G_dph2 = jax.grad(jax.grad(G2, 1), 1)(Th, ph)
    d2G_dTh_dph = jax.grad(jax.grad(G2, 0), 1)(Th, ph)

    # Reduced volume and entropy
    Vh = dG_dph
    Sh = -dG_dTh

    # Derivatives for response functions
    dVh_dph = d2G_dph2
    dVh_dTh = d2G_dTh_dph
    dSh_dTh = -d2G_dTh2

    # Physical conversion (same as core.py _pure_state_props)
    T = Th * P.T_VLCP
    V_phys = Vh * P.R_specific * P.T_VLCP / (P.p_VLCP * 1e6)  # m³/kg
    rho = jnp.where(V_phys > 0, 1.0 / V_phys, jnp.inf)
    S_phys = Sh * P.R_specific  # J/(kg·K)

    # Cp = R * Th * dSh/dTh
    Cp = P.R_specific * Th * dSh_dTh

    # Isothermal compressibility
    kappa_T_red = jnp.where(jnp.abs(Vh) > 1e-30,
                             -(1.0 / Vh) * dVh_dph, jnp.inf)
    kappa_T = kappa_T_red / (P.p_VLCP * 1e6)  # Pa^-1
    Kt = jnp.where((jnp.abs(kappa_T) > 1e-30) & jnp.isfinite(kappa_T),
                    1.0 / kappa_T / 1e6, 0.0)

    # Thermal expansion
    alpha = jnp.where(Vh > 0,
                       (1.0 / Vh) * dVh_dTh / P.T_VLCP, 0.0)

    # Cv
    Cv = jnp.where((kappa_T > 0) & jnp.isfinite(kappa_T),
                    Cp - T * V_phys * alpha**2 / kappa_T, Cp)

    # Adiabatic compressibility
    kap_S = jnp.where(Cp > 0,
                       kappa_T - T * V_phys * alpha**2 / Cp, kappa_T)
    Ks = jnp.where(kap_S > 0, 1.0 / kap_S / 1e6, jnp.inf)
    vel = jnp.where((rho > 0) & (kap_S > 0),
                     jnp.sqrt(1.0 / (rho * kap_S)), jnp.nan)

    G_phys = P.R_specific * P.T_VLCP * G_val  # J/kg

    return {
        'rho': rho, 'V': V_phys, 'S': S_phys,
        'Cp': Cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks,
        'alpha': alpha, 'vel': vel, 'G': G_phys,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 11. Main entry points
# ═══════════════════════════════════════════════════════════════════════════

def _compute_properties_scalar(T_K, p_MPa):
    """Compute all properties at a single (T, P) point (JAX scalars)."""
    Th = T_K / P.T_VLCP
    ph = p_MPa / P.p_VLCP

    # Mixture (equilibrium x)
    mix = _compute_props_from_G_hat(_G_hat_mix, Th, ph, T_K)
    x_eq = _solve_x_eq(Th, ph, Th)
    mix['x'] = x_eq

    # State A (x=0)
    stateA = _compute_props_from_G_hat(_G_hat_A, Th, ph, T_K)

    # State B (x=1)
    stateB = _compute_props_from_G_hat(_G_hat_B, Th, ph, T_K)

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
    implicit dx/dTh, dx/dph.
    """
    T_jax = jnp.float64(T_K)
    p_jax = jnp.float64(p_MPa)
    x_jax = jnp.float64(x)

    Th = T_jax / P.T_VLCP
    ph = p_jax / P.p_VLCP

    # 3-arg g_mix for AD
    def _g3(x_v, Th_v, ph_v):
        return _g_mix(x_v, Th_v, ph_v, Th_v)

    _dg_dph = jax.grad(_g3, argnums=2)
    _dg_dTh = jax.grad(_g3, argnums=1)

    # Build forced solver with implicit function theorem
    dg_dx = jax.grad(_g3, argnums=0)
    d2g_dx2 = jax.grad(dg_dx, argnums=0)
    d2g_dxdTh = jax.grad(dg_dx, argnums=1)
    d2g_dxdph = jax.grad(dg_dx, argnums=2)

    @jax.custom_jvp
    def solve_forced(Th_v, ph_v):
        return x_jax

    @solve_forced.defjvp
    def solve_forced_jvp(primals, tangents):
        Th_v, ph_v = primals
        dTh, dph = tangents
        gxx = d2g_dx2(x_jax, Th_v, ph_v)
        gxx_safe = jnp.where(jnp.abs(gxx) > 1e-30, gxx, jnp.sign(gxx + 1e-40) * 1e-30)
        gxT = d2g_dxdTh(x_jax, Th_v, ph_v)
        gxP = d2g_dxdph(x_jax, Th_v, ph_v)
        dx = -(gxT * dTh + gxP * dph) / gxx_safe
        return x_jax, dx

    # "Total" first-derivative functions
    def _v_total(Th_v, ph_v):
        x_v = solve_forced(Th_v, ph_v)
        return _dg_dph(x_v, Th_v, ph_v)

    def _dG_dTh_total(Th_v, ph_v):
        x_v = solve_forced(Th_v, ph_v)
        return _dg_dTh(x_v, Th_v, ph_v)

    # Value
    G_val = _g3(x_jax, Th, ph)

    # First derivatives (partial at fixed x)
    Vh = _v_total(Th, ph)
    Sh = -_dG_dTh_total(Th, ph)

    # Second derivatives (total)
    dVh_dph = jax.grad(_v_total, 1)(Th, ph)
    dVh_dTh = jax.grad(_v_total, 0)(Th, ph)
    dSh_dTh = -jax.grad(_dG_dTh_total, 0)(Th, ph)

    # Physical conversion
    T = Th * P.T_VLCP
    V_phys = Vh * P.R_specific * P.T_VLCP / (P.p_VLCP * 1e6)
    rho = jnp.where(V_phys > 0, 1.0 / V_phys, jnp.inf)
    S_phys = Sh * P.R_specific

    Cp = P.R_specific * Th * dSh_dTh

    kappa_T_red = jnp.where(jnp.abs(Vh) > 1e-30, -(1.0/Vh)*dVh_dph, jnp.inf)
    kappa_T = kappa_T_red / (P.p_VLCP * 1e6)
    Kt = jnp.where((jnp.abs(kappa_T) > 1e-30) & jnp.isfinite(kappa_T),
                    1.0 / kappa_T / 1e6, 0.0)

    alpha = jnp.where(Vh > 0, (1.0/Vh)*dVh_dTh/P.T_VLCP, 0.0)

    Cv = jnp.where((kappa_T > 0) & jnp.isfinite(kappa_T),
                    Cp - T*V_phys*alpha**2/kappa_T, Cp)
    kap_S = jnp.where(Cp > 0, kappa_T - T*V_phys*alpha**2/Cp, kappa_T)
    Ks = jnp.where(kap_S > 0, 1.0/kap_S/1e6, jnp.inf)
    vel = jnp.where((rho > 0) & (kap_S > 0),
                     jnp.sqrt(1.0/(rho*kap_S)), jnp.nan)
    G_phys = P.R_specific * P.T_VLCP * G_val

    props = {
        'rho': float(rho), 'V': float(V_phys), 'S': float(S_phys),
        'Cp': float(Cp), 'Cv': float(Cv), 'Kt': float(Kt),
        'Ks': float(Ks), 'alpha': float(alpha), 'vel': float(vel),
        'G': float(G_phys),
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
# 12. Vectorized batch computation via vmap
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
# 13. G_hat_mix for TMD/Widom solvers
# ═══════════════════════════════════════════════════════════════════════════

def _G_hat_mix_3arg(dTh, dPh, T_K):
    """
    G_hat(dTh, dPh, T_K) for TMD/Widom solvers.

    dTh and dPh use the Caupin-style convention:
      dTh = (T_K - Tc) / Tc  (where Tc is the LLCP temperature)
      dPh = (P_MPa - Pc) / P_scale_MPa  (where Pc is the LLCP pressure)

    We need to find the LLCP first to use this properly.
    For the Duska model, we convert to Th/ph internally.
    """
    # T_K is passed directly
    Th = T_K / P.T_VLCP
    # Recover P_MPa from dPh: this requires knowledge of Pc and P_scale_MPa
    # These are the LLCP values, computed at import time below.
    P_MPa = _Pc_LLCP + dPh * _P_scale_LLCP
    ph = P_MPa / P.p_VLCP

    x = _solve_x_eq(Th, ph, Th)
    return _g_mix(x, Th, ph, Th)


# LLCP values needed for _G_hat_mix_3arg coordinate conversion.
# Computed lazily on first use.
_Pc_LLCP = None
_P_scale_LLCP = None


def _ensure_llcp_cached():
    global _Pc_LLCP, _P_scale_LLCP
    if _Pc_LLCP is None:
        from .phase_diagram import find_LLCP
        llcp = find_LLCP()
        _Pc_LLCP = llcp['p_MPa']
        # Use same P_scale as Caupin convention for TMD/Widom
        _P_scale_LLCP = 1.0  # 1 MPa per unit of dPh


_ensure_llcp_cached()
