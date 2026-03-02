"""
Universal AD-derived phase diagram adapter for two-state EoS models.

Given a single g_mix(x, T_K, P_MPa) function, builds a
fast_phase_diagram-compatible adapter dict using jax.grad for F_eq (dg/dx)
and d²g/dx² evaluations.

Eliminates per-model hand-coded omega_vec, disc_vec, F_eq_vec, g_mix_vec.

Mathematical basis
------------------
All two-state EoS models have the mixing Gibbs energy:

    g(x) = ... + f·[x ln x + (1-x) ln(1-x)] + W·x·(1-x)

where f is the entropy prefactor (1, Th, or Th depending on the model)
and W is the effective interaction parameter.  The second derivative is:

    d²g/dx² = f / [x(1-x)] - 2W

Evaluating at x = 1/2 and x = 1/4 yields two equations:

    D_half    = 4f - 2W
    D_quarter = (16/3)f - 2W

from which:  f = 3(D_quarter - D_half)/4,  W = (4f - D_half)/2

The spinodal discriminant is:  disc = 1 - 2f/W
Inflection points:  x = (1 ± sqrt(disc)) / 2
"""

import numpy as np
import jax
import jax.numpy as jnp


def make_phase_adapter(g_mix_fn, find_llcp_fn):
    """
    Build a fast_phase_diagram adapter from a scalar g_mix function using AD.

    Parameters
    ----------
    g_mix_fn : callable
        g_mix(x, T_K, P_MPa) -> scalar reduced Gibbs mixing energy.
        Must be JAX-differentiable w.r.t. x (argument 0).
    find_llcp_fn : callable
        find_llcp_fn() -> dict with at least 'T_K' and 'p_MPa' keys.

    Returns
    -------
    dict
        Adapter compatible with watereos.fast_phase_diagram:
        T_LLCP, p_LLCP, omega_vec, disc_vec, F_eq_vec, g_mix_vec
    """
    # AD derivatives w.r.t. x
    dg_dx = jax.grad(g_mix_fn, argnums=0)
    d2g_dx2 = jax.grad(dg_dx, argnums=0)

    # Vectorized via vmap + jit
    _F_eq_vmap = jax.jit(jax.vmap(dg_dx, in_axes=(0, 0, 0)))
    _g_mix_vmap = jax.jit(jax.vmap(g_mix_fn, in_axes=(0, 0, 0)))
    _d2g_dx2_vmap = jax.jit(jax.vmap(d2g_dx2, in_axes=(0, 0, 0)))

    llcp = find_llcp_fn()

    # Shared cache between omega_vec and disc_vec (always called in sequence)
    _cache = {}

    def _compute_f_W(T_arr, P_arr):
        """Extract entropy prefactor f and interaction W from d²g/dx²."""
        n = len(T_arr)
        T_j = jnp.asarray(T_arr, dtype=jnp.float64)
        P_j = jnp.asarray(P_arr, dtype=jnp.float64)

        D_half = np.asarray(_d2g_dx2_vmap(
            jnp.full(n, 0.5), T_j, P_j))
        D_quarter = np.asarray(_d2g_dx2_vmap(
            jnp.full(n, 0.25), T_j, P_j))

        # d²g/dx²(x) = f/(x(1-x)) - 2W
        # At x=1/2: D_half = 4f - 2W
        # At x=1/4: D_quarter = (16/3)f - 2W
        f = 3.0 * (D_quarter - D_half) / 4.0
        W = (4.0 * f - D_half) / 2.0
        return f, W

    def omega_vec(T_arr, P_arr):
        """Effective interaction parameter W (positive when phase separation possible)."""
        f, W = _compute_f_W(T_arr, P_arr)
        _cache['f'] = f
        _cache['n'] = len(T_arr)
        return W

    def disc_vec(W, T_arr, P_arr):
        """Discriminant: disc > 0 means spinodal exists at (T, P)."""
        f = _cache.get('f') if _cache.get('n') == len(T_arr) else None
        if f is None:
            f, _ = _compute_f_W(T_arr, P_arr)
        safe_W = np.where(np.abs(W) > 1e-30, W, 1e-30)
        return 1.0 - 2.0 * f / safe_W

    def F_eq_vec(x_arr, T_arr, P_arr):
        """dg_mix/dx at given (x, T, P) arrays."""
        x_j = jnp.asarray(x_arr, dtype=jnp.float64)
        T_j = jnp.asarray(T_arr, dtype=jnp.float64)
        P_j = jnp.asarray(P_arr, dtype=jnp.float64)
        return np.asarray(_F_eq_vmap(x_j, T_j, P_j))

    def g_mix_vec(x_arr, T_arr, P_arr):
        """g_mix at given (x, T, P) arrays."""
        x_j = jnp.asarray(x_arr, dtype=jnp.float64)
        T_j = jnp.asarray(T_arr, dtype=jnp.float64)
        P_j = jnp.asarray(P_arr, dtype=jnp.float64)
        return np.asarray(_g_mix_vmap(x_j, T_j, P_j))

    return {
        'T_LLCP': llcp['T_K'],
        'p_LLCP': llcp['p_MPa'],
        'omega_vec': omega_vec,
        'disc_vec': disc_vec,
        'F_eq_vec': F_eq_vec,
        'g_mix_vec': g_mix_vec,
    }
