"""
Singh et al. (2017) two-state transport properties: core engine.

Computes viscosity (eta), self-diffusion coefficient (D), and rotational
correlation time (tau_r) from the LDS fraction f(T,P) obtained from a
two-state thermodynamic backbone model.

The model equation (Eq. 1) is:

  A(T,P) = A0 * (T/Tref)^nu * exp{ eps * [(1-f)*HDS_term + f*LDS_term] }

where:
  HDS_term = (E_HDS/kB + dv_HDS*P_Pa/kB) / (T - T0)
  LDS_term = E_LDS/kB / T

Reference: L. P. Singh, B. Issenmann, F. Caupin, PNAS 114, 4312 (2017).
"""

import numpy as np
from . import params as P


def _singh_property(T_K, f, P_Pa, A0, E_LDS_k, E_HDS_k, dv_HDS, nu, eps):
    """
    Evaluate the Singh Eq. 1 for a single transport property.

    Works for both scalar and array inputs (numpy broadcasting).

    Parameters
    ----------
    T_K     : temperature (K)
    f       : LDS fraction (from backbone model)
    P_Pa    : pressure (Pa)
    A0      : prefactor (property-specific units)
    E_LDS_k : E_LDS / k_B (K)
    E_HDS_k : E_HDS / k_B (K)
    dv_HDS  : activation volume for HDS (m^3)
    nu      : power-law exponent
    eps     : sign factor (+1 or -1)

    Returns
    -------
    Transport property value(s) in the units of A0.
    """
    HDS_term = (E_HDS_k + dv_HDS * P_Pa / P.k_B) / (T_K - P.T_0)
    LDS_term = E_LDS_k / T_K
    exponent = (1.0 - f) * HDS_term + f * LDS_term
    return A0 * (T_K / P.T_ref) ** nu * np.exp(eps * exponent)


def compute_properties(T_K, p_MPa):
    """
    Compute transport properties at a single (T, P) point.

    Uses Holten et al. (2014) as the backbone model to obtain the LDS
    fraction f = x.

    Parameters
    ----------
    T_K   : float — temperature in K
    p_MPa : float — pressure in MPa

    Returns
    -------
    dict with keys:
        'eta'   : dynamic viscosity (Pa*s)
        'D'     : self-diffusion coefficient (m^2/s)
        'tau_r' : rotational correlation time (s)
        'f'     : LDS fraction from backbone model
    """
    from holten_eos.core import compute_properties as holten_props

    props = holten_props(T_K, p_MPa)
    f = props['x']
    P_Pa = p_MPa * 1e6

    eta = _singh_property(T_K, f, P_Pa,
                          P.A0_eta, P.E_LDS_k_eta, P.E_HDS_k_eta,
                          P.dv_HDS_eta, P.nu_eta, P.eps_eta)
    D = _singh_property(T_K, f, P_Pa,
                        P.A0_D, P.E_LDS_k_D, P.E_HDS_k_D,
                        P.dv_HDS_D, P.nu_D, P.eps_D)
    tau_r = _singh_property(T_K, f, P_Pa,
                            P.A0_tau, P.E_LDS_k_tau, P.E_HDS_k_tau,
                            P.dv_HDS_tau, P.nu_tau, P.eps_tau)

    return {'eta': eta, 'D': D, 'tau_r': tau_r, 'f': f}


def compute_batch(T_K, p_MPa):
    """
    Vectorized computation of transport + thermodynamic properties.

    Calls holten_eos.core.compute_batch once to get the LDS fraction and
    all thermodynamic properties, then computes eta, D, tau_r vectorized.

    Parameters
    ----------
    T_K   : 1-D array — temperature in K
    p_MPa : 1-D array — pressure in MPa (same length as T_K)

    Returns
    -------
    dict of 1-D arrays.  Contains:
        Transport: 'eta', 'D', 'tau_r'
        LDS fraction: 'f'  (same as 'x' from Holten)
        All Holten thermodynamic properties passed through:
            rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, alpha, vel, x,
            rho_A, V_A, ..., rho_B, V_B, ...
    """
    from holten_eos.core import compute_batch as holten_batch

    T_K = np.asarray(T_K, dtype=float)
    p_MPa = np.asarray(p_MPa, dtype=float)

    # Get all thermodynamic properties + LDS fraction from Holten
    thermo = holten_batch(T_K, p_MPa)
    f = thermo['x']
    P_Pa = p_MPa * 1e6

    # Vectorized transport property computation
    eta = _singh_property(T_K, f, P_Pa,
                          P.A0_eta, P.E_LDS_k_eta, P.E_HDS_k_eta,
                          P.dv_HDS_eta, P.nu_eta, P.eps_eta)
    D = _singh_property(T_K, f, P_Pa,
                        P.A0_D, P.E_LDS_k_D, P.E_HDS_k_D,
                        P.dv_HDS_D, P.nu_D, P.eps_D)
    tau_r = _singh_property(T_K, f, P_Pa,
                            P.A0_tau, P.E_LDS_k_tau, P.E_HDS_k_tau,
                            P.dv_HDS_tau, P.nu_tau, P.eps_tau)

    # Build output: transport properties + pass-through thermodynamics
    result = dict(thermo)  # shallow copy
    result['eta'] = eta
    result['D'] = D
    result['tau_r'] = tau_r
    result['f'] = f

    return result
