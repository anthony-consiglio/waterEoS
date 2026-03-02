"""
Differentiable equilibrium solver for two-state EoS models.

Given a mixing Gibbs energy g_mix(x, dTh, dPh, T_K), finds the
equilibrium composition x that minimizes g_mix. The solver is
differentiable w.r.t. (dTh, dPh) via the implicit function theorem,
implemented using jax.custom_jvp.
"""

import jax
import jax.numpy as jnp


def _newton(dg_dx, d2g_dx2, x0, dTh, dPh, T_K, n_iter=50):
    """
    Fixed-iteration Newton solver for dg/dx = 0.

    Uses jax.lax.fori_loop for vmap compatibility (no Python-level
    branching on array values).
    """
    def body(_, x):
        x_c = jnp.clip(x, 1e-15, 1.0 - 1e-15)
        F = dg_dx(x_c, dTh, dPh, T_K)
        Fx = d2g_dx2(x_c, dTh, dPh, T_K)
        # Guard against near-zero second derivative
        Fx_safe = jnp.where(jnp.abs(Fx) > 1e-30, Fx, jnp.sign(Fx + 1e-40) * 1e-30)
        x_new = x_c - F / Fx_safe
        return jnp.clip(x_new, 1e-15, 1.0 - 1e-15)

    return jax.lax.fori_loop(0, n_iter, body, jnp.float64(x0))


def make_equilibrium_solver(g_mix_fn):
    """
    Build a differentiable equilibrium solver from a g_mix function.

    Parameters
    ----------
    g_mix_fn : callable
        g_mix(x, dTh, dPh, T_K) -> scalar reduced Gibbs energy of mixing.

    Returns
    -------
    solve_x : callable
        solve_x(dTh, dPh, T_K) -> x_eq (scalar).
        Differentiable w.r.t. dTh and dPh via implicit function theorem.
    """
    dg_dx = jax.grad(g_mix_fn, argnums=0)
    d2g_dx2 = jax.grad(dg_dx, argnums=0)
    d2g_dxdT = jax.grad(dg_dx, argnums=1)
    d2g_dxdP = jax.grad(dg_dx, argnums=2)

    @jax.custom_jvp
    def solve_x(dTh, dPh, T_K):
        # Newton from two starting points; pick the one with lower g_mix
        x_lo = _newton(dg_dx, d2g_dx2, 0.05, dTh, dPh, T_K)
        x_hi = _newton(dg_dx, d2g_dx2, 0.95, dTh, dPh, T_K)
        g_lo = g_mix_fn(x_lo, dTh, dPh, T_K)
        g_hi = g_mix_fn(x_hi, dTh, dPh, T_K)
        return jnp.where(g_lo <= g_hi, x_lo, x_hi)

    @solve_x.defjvp
    def solve_x_jvp(primals, tangents):
        dTh, dPh, T_K = primals
        dTh_dot, dPh_dot, T_K_dot = tangents
        x_eq = solve_x(dTh, dPh, T_K)

        # Implicit function theorem:
        # dx/dTh = -(d2g/dxdTh) / (d2g/dx2)
        # dx/dPh = -(d2g/dxdPh) / (d2g/dx2)
        #
        # T_K = Tc * (1 + dTh), so dT_K = Tc * dTh => T_K_dot contributes
        # through dTh_dot only. But since T_K is passed as a separate arg,
        # we also need d2g/dxdT_K. For the Caupin model, T_K appears in
        # the spinodal pressure. We handle this via the chain rule.
        gxx = d2g_dx2(x_eq, dTh, dPh, T_K)
        gxx_safe = jnp.where(jnp.abs(gxx) > 1e-30, gxx, jnp.sign(gxx + 1e-40) * 1e-30)

        gxT = d2g_dxdT(x_eq, dTh, dPh, T_K)
        gxP = d2g_dxdP(x_eq, dTh, dPh, T_K)

        # d2g/dxdT_K (T_K is arg 3)
        d2g_dxdTK = jax.grad(dg_dx, argnums=3)(x_eq, dTh, dPh, T_K)

        dx = -(gxT * dTh_dot + gxP * dPh_dot + d2g_dxdTK * T_K_dot) / gxx_safe
        return x_eq, dx

    return solve_x
