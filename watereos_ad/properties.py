"""
Model-agnostic property computation via automatic differentiation.

Given a reduced Gibbs function G_hat(dTh, dPh), computes all physical
thermodynamic properties using jax.grad for derivatives.
"""

import jax
import jax.numpy as jnp


def compute_properties_ad(G_hat_fn, dTh, dPh, T_K, params):
    """
    Compute physical properties from a reduced Gibbs function using AD.

    Parameters
    ----------
    G_hat_fn : callable
        G_hat(dTh, dPh) -> scalar reduced Gibbs energy.
    dTh : float — reduced temperature deviation
    dPh : float — reduced pressure deviation
    T_K : float — temperature in K
    params : module — model parameters (Tc, Vc, R, M_H2O, etc.)

    Returns
    -------
    dict with: rho, V, S, Cp, Cv, Kt, Ks, alpha, vel, G
    """
    Th = T_K / params.Tc

    # Value
    G_val = G_hat_fn(dTh, dPh)

    # First derivatives
    dG_dP = jax.grad(G_hat_fn, 1)(dTh, dPh)   # Vh = dG/dPh
    dG_dT = jax.grad(G_hat_fn, 0)(dTh, dPh)   # dG/dTh (Sh_red = -dG_dT)

    # Second derivatives
    d2G_dP2 = jax.grad(jax.grad(G_hat_fn, 1), 1)(dTh, dPh)
    d2G_dT2 = jax.grad(jax.grad(G_hat_fn, 0), 0)(dTh, dPh)
    d2G_dPdT = jax.grad(jax.grad(G_hat_fn, 1), 0)(dTh, dPh)

    Vh = dG_dP
    Sh_red = -dG_dT

    # Molar volume and specific volume
    V_molar = params.Vc * Vh
    V_spec = V_molar / params.M_H2O
    rho = jnp.where(V_spec > 0, 1.0 / V_spec, jnp.inf)

    # Entropy
    S_molar = params.R * Sh_red
    S_spec = S_molar / params.M_H2O

    # Heat capacity at constant pressure
    Cp_molar = -params.R * Th * d2G_dT2
    Cp = Cp_molar / params.M_H2O

    # Isothermal compressibility
    kappa_T = jnp.where(
        jnp.abs(Vh) > 1e-30,
        -(params.Vc / (params.R * params.Tc)) * d2G_dP2 / Vh,
        jnp.inf
    )
    Kt = jnp.where(
        (jnp.abs(kappa_T) > 1e-30) & jnp.isfinite(kappa_T),
        1.0 / kappa_T / 1e6,
        0.0
    )

    # Thermal expansion coefficient
    alpha = jnp.where(
        jnp.abs(Vh) > 1e-30,
        (1.0 / params.Tc) * d2G_dPdT / Vh,
        0.0
    )

    # Heat capacity at constant volume
    Cv = jnp.where(
        (kappa_T > 0) & jnp.isfinite(kappa_T),
        Cp - T_K * V_spec * alpha**2 / kappa_T,
        Cp
    )

    # Adiabatic compressibility
    kappa_S = jnp.where(
        Cp > 0,
        kappa_T - T_K * V_spec * alpha**2 / Cp,
        kappa_T
    )
    Ks = jnp.where(kappa_S > 0, 1.0 / kappa_S / 1e6, jnp.inf)

    # Speed of sound
    vel = jnp.where(
        (rho > 0) & (kappa_S > 0),
        jnp.sqrt(1.0 / (rho * kappa_S)),
        jnp.nan
    )

    # Gibbs energy
    G_phys = params.R * params.Tc * G_val / params.M_H2O

    return {
        'rho': rho, 'V': V_spec, 'S': S_spec,
        'Cp': Cp, 'Cv': Cv, 'Kt': Kt, 'Ks': Ks,
        'alpha': alpha, 'vel': vel, 'G': G_phys,
    }
