"""
AD-enhanced TMD and Widom line computation.

TMD (Temperature of Maximum Density):
    Find T where alpha_P(T, P) = 0.
    Newton: T_{n+1} = T_n - alpha(T_n) / (dalpha/dT)(T_n)
    dalpha/dT involves d³G/(dTh² dPh) — computed by jax.grad.

Widom line (locus of max Cp below LLCP):
    Find T where dCp/dT = 0, using bisection on AD-computed dCp/dT.
    dCp/dT involves d³G/dTh³ — computed by jax.grad.
    Bisection is more robust than Newton near the LLCP where Cp diverges.
"""

import numpy as np
import jax
import jax.numpy as jnp


def make_tmd_solver(G_hat_mix_3arg, params):
    """
    Create a Newton-based TMD solver using AD.

    Parameters
    ----------
    G_hat_mix_3arg : callable
        G_hat(dTh, dPh, T_K) -> scalar reduced Gibbs energy at equilibrium x.
    params : module
        Model parameters with Tc, Pc, P_scale_MPa, R, M_H2O, Vc.

    Returns
    -------
    callable
        compute_tmd(P_arr, T_init, n_iter=15) -> numpy array of TMD temperatures.
    """
    def _G2(dTh, dPh):
        T_K = params.Tc * (1.0 + dTh)
        return G_hat_mix_3arg(dTh, dPh, T_K)

    def _alpha_scalar(T_K, P_MPa):
        """Thermal expansion coefficient at (T, P)."""
        dTh = (T_K - params.Tc) / params.Tc
        dPh = (P_MPa - params.Pc) / params.P_scale_MPa
        Vh = jax.grad(_G2, 1)(dTh, dPh)
        d2 = jax.grad(jax.grad(_G2, 1), 0)(dTh, dPh)
        return (1.0 / params.Tc) * d2 / Vh

    _dalpha_dT = jax.grad(_alpha_scalar, argnums=0)

    _alpha_vmap = jax.jit(jax.vmap(_alpha_scalar, in_axes=(0, 0)))
    _dalpha_vmap = jax.jit(jax.vmap(_dalpha_dT, in_axes=(0, 0)))

    def compute_tmd(P_arr, T_init, n_iter=15):
        """
        Newton TMD: find T where alpha=0 at each pressure.

        Parameters
        ----------
        P_arr : 1D array
            Pressures in MPa.
        T_init : 1D array or scalar
            Initial guess for TMD temperature at each pressure.
        n_iter : int
            Number of Newton iterations.

        Returns
        -------
        numpy array of TMD temperatures (K).
        """
        P_j = jnp.asarray(P_arr, dtype=jnp.float64)
        T = jnp.asarray(np.broadcast_to(T_init, P_arr.shape), dtype=jnp.float64)

        for _ in range(n_iter):
            a = _alpha_vmap(T, P_j)
            da = _dalpha_vmap(T, P_j)
            da_safe = jnp.where(jnp.abs(da) > 1e-30, da, jnp.sign(da + 1e-40) * 1e-30)
            T = T - a / da_safe
            T = jnp.clip(T, 100.0, 400.0)

        return np.asarray(T)

    return compute_tmd


def make_widom_solver(G_hat_mix_3arg, params):
    """
    Create a bisection-based Widom line solver using AD-computed dCp/dT.

    Uses jax.grad to compute dCp/dT (3rd derivative of G), then bisects
    on the sign of dCp/dT to find the Cp maximum. This is more robust
    than Newton near the LLCP where Cp diverges sharply.

    Parameters
    ----------
    G_hat_mix_3arg : callable
        G_hat(dTh, dPh, T_K) -> scalar reduced Gibbs energy at equilibrium x.
    params : module
        Model parameters with Tc, Pc, P_scale_MPa, R, M_H2O.

    Returns
    -------
    callable
        compute_widom(P_arr, T_lo, T_hi, n_iter=40) -> numpy array of Widom temperatures.
    """
    def _G2(dTh, dPh):
        T_K = params.Tc * (1.0 + dTh)
        return G_hat_mix_3arg(dTh, dPh, T_K)

    def _Cp_scalar(T_K, P_MPa):
        """Heat capacity Cp at (T, P)."""
        dTh = (T_K - params.Tc) / params.Tc
        dPh = (P_MPa - params.Pc) / params.P_scale_MPa
        Th = T_K / params.Tc
        d2G_dT2 = jax.grad(jax.grad(_G2, 0), 0)(dTh, dPh)
        return -params.R * Th * d2G_dT2 / params.M_H2O

    _dCp_dT = jax.grad(_Cp_scalar, argnums=0)
    _dCp_vmap = jax.jit(jax.vmap(_dCp_dT, in_axes=(0, 0)))
    _Cp_vmap = jax.jit(jax.vmap(_Cp_scalar, in_axes=(0, 0)))

    def compute_widom(P_arr, T_lo, T_hi, n_iter=40):
        """
        Bisection Widom: find T where dCp/dT=0 (Cp maximum) at each pressure.

        Parameters
        ----------
        P_arr : 1D array
            Pressures in MPa (below LLCP pressure).
        T_lo, T_hi : 1D arrays
            Lower and upper temperature brackets for bisection.
            dCp/dT should be positive at T_lo and negative at T_hi.
        n_iter : int
            Number of bisection iterations.

        Returns
        -------
        numpy array of Widom temperatures (K).
        """
        P_j = jnp.asarray(P_arr, dtype=jnp.float64)
        lo = jnp.asarray(np.broadcast_to(T_lo, P_arr.shape), dtype=jnp.float64)
        hi = jnp.asarray(np.broadcast_to(T_hi, P_arr.shape), dtype=jnp.float64)

        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            dCp = _dCp_vmap(mid, P_j)
            lo = jnp.where(dCp > 0, mid, lo)
            hi = jnp.where(dCp <= 0, mid, hi)

        return np.asarray((lo + hi) / 2.0)

    def get_Cp(T_arr, P_arr):
        """Evaluate Cp at given (T, P) arrays."""
        return np.asarray(_Cp_vmap(
            jnp.asarray(T_arr, dtype=jnp.float64),
            jnp.asarray(P_arr, dtype=jnp.float64)))

    def get_dCp_dT(T_arr, P_arr):
        """Evaluate dCp/dT at given (T, P) arrays."""
        return np.asarray(_dCp_vmap(
            jnp.asarray(T_arr, dtype=jnp.float64),
            jnp.asarray(P_arr, dtype=jnp.float64)))

    compute_widom.get_Cp = get_Cp
    compute_widom.get_dCp_dT = get_dCp_dT
    return compute_widom


def compute_tmd_curve(G_hat_mix_3arg, params, T_LLCP, p_LLCP,
                      compute_batch_fn, P_lo=-140.0, P_hi=None,
                      n_points=80, n_iter=15):
    """
    Compute the TMD curve using coarse scan + Newton refinement.

    Parameters
    ----------
    G_hat_mix_3arg : callable
        G_hat(dTh, dPh, T_K) -> scalar Gibbs energy at equilibrium x.
    params : module
        Model parameters.
    T_LLCP, p_LLCP : float
        LLCP location (TMD terminates at LLCP pressure).
    compute_batch_fn : callable
        compute_batch(T_arr, P_arr) -> dict with 'alpha' key.
    P_lo : float
        Lower pressure bound (MPa).
    P_hi : float or None
        Upper pressure bound (defaults to p_LLCP).
    n_points : int
        Number of pressure points.
    n_iter : int
        Newton iterations for refinement.

    Returns
    -------
    dict with 'T_K' and 'p_MPa' arrays, or None if no TMD found.
    """
    if P_hi is None:
        P_hi = p_LLCP
    if P_lo >= P_hi:
        return None

    P_arr = np.linspace(P_lo, P_hi, n_points)

    # Coarse scan: evaluate alpha on a T-P grid
    n_T_scan = 60
    T_scan = np.linspace(125.0, 350.0, n_T_scan)
    T_grid, P_grid = np.meshgrid(T_scan, P_arr)

    batch = compute_batch_fn(T_grid.ravel(), P_grid.ravel())
    alpha_grid = batch['alpha'].reshape(n_points, n_T_scan)

    # Find first sign change along T axis
    valid = ~np.isnan(alpha_grid[:, :-1]) & ~np.isnan(alpha_grid[:, 1:])
    sign_change = (alpha_grid[:, :-1] * alpha_grid[:, 1:] < 0) & valid
    has_change = sign_change.any(axis=1)
    if not has_change.any():
        return None

    first_idx = sign_change.argmax(axis=1)
    mask = has_change
    j_bracket = first_idx[mask]
    P_bracket = P_arr[mask]

    # Initial guess: midpoint of bracketing interval
    T_init = 0.5 * (T_scan[j_bracket] + T_scan[j_bracket + 1])

    # Newton refinement
    tmd_solver = make_tmd_solver(G_hat_mix_3arg, params)
    T_tmd = tmd_solver(P_bracket, T_init, n_iter=n_iter)

    return {'T_K': T_tmd, 'p_MPa': P_bracket}


def compute_widom_line(G_hat_mix_3arg, params, T_LLCP, p_LLCP,
                       compute_batch_fn, n_points=60, n_iter=40):
    """
    Compute the Widom line using coarse Cp scan + AD bisection on dCp/dT.

    Parameters
    ----------
    G_hat_mix_3arg : callable
        G_hat(dTh, dPh, T_K) -> scalar Gibbs energy at equilibrium x.
    params : module
        Model parameters.
    T_LLCP, p_LLCP : float
        LLCP location.
    compute_batch_fn : callable
        compute_batch(T_arr, P_arr) -> dict with 'Cp' key.
    n_points : int
        Number of pressure points.
    n_iter : int
        Bisection iterations for refinement.

    Returns
    -------
    dict with 'T_K' and 'p_MPa' arrays, or None if no Widom line found.
    """
    P_lo = max(p_LLCP - 100.0, -200.0)
    P_hi = p_LLCP - 0.5
    if P_lo >= P_hi:
        return None

    P_arr = np.linspace(P_hi, P_lo, n_points)

    # Coarse Cp scan to find initial brackets
    dP = p_LLCP - P_arr
    T_centers = T_LLCP + dP * 0.3

    n_T_scan = 120
    T_global_lo = max(float(np.min(T_centers)) - 30.0, 100.0)
    T_global_hi = min(float(np.max(T_centers)) + 30.0, 400.0)
    T_scan = np.linspace(T_global_lo, T_global_hi, n_T_scan)

    T_grid, P_grid = np.meshgrid(T_scan, P_arr)
    batch = compute_batch_fn(T_grid.ravel(), P_grid.ravel())
    Cp_grid = batch['Cp'].reshape(n_points, n_T_scan)

    # Per-pressure window mask
    T_lo_per_P = np.maximum(T_centers - 30.0, 100.0)[:, np.newaxis]
    T_hi_per_P = np.minimum(T_centers + 30.0, 400.0)[:, np.newaxis]
    T_row = T_scan[np.newaxis, :]
    window_mask = (T_row >= T_lo_per_P) & (T_row <= T_hi_per_P)

    Cp_masked = np.where(window_mask & ~np.isnan(Cp_grid), Cp_grid, -np.inf)
    idx_max = np.argmax(Cp_masked, axis=1)
    peak_Cp = Cp_masked[np.arange(n_points), idx_max]
    has_valid = peak_Cp > -np.inf

    # Interior check
    first_valid = np.argmax(window_mask, axis=1)
    last_valid = n_T_scan - 1 - np.argmax(window_mask[:, ::-1], axis=1)
    interior = (idx_max > first_valid) & (idx_max < last_valid) & has_valid

    if not interior.any():
        return None

    # Build brackets: T_lo = one step below peak, T_hi = one step above
    T_lo_bracket = T_scan[np.maximum(idx_max - 1, 0)]
    T_hi_bracket = T_scan[np.minimum(idx_max + 1, n_T_scan - 1)]

    # Filter to interior points
    P_valid = P_arr[interior]
    T_lo_b = T_lo_bracket[interior]
    T_hi_b = T_hi_bracket[interior]

    # Bisection refinement using AD-computed dCp/dT
    widom_solver = make_widom_solver(G_hat_mix_3arg, params)
    T_widom = widom_solver(P_valid, T_lo_b, T_hi_b, n_iter=n_iter)

    # Final filter
    T_min = T_LLCP - 30.0
    T_max = T_LLCP + 60.0
    good = (T_widom > T_min) & (T_widom < T_max) & ~np.isnan(T_widom)

    if not good.any():
        return None

    return {'T_K': T_widom[good], 'p_MPa': P_valid[good]}
