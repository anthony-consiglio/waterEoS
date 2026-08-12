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

The original paper used Holten et al. (2014) as the backbone; this module
also supports the Caupin (2019) and Duska (2020) two-state EoS with
parameters refitted against the same experimental datasets (see
scripts/refit_singh_transport.py).

Reference: L. P. Singh, B. Issenmann, F. Caupin, PNAS 114, 4312 (2017).
"""

import numpy as np
from . import params as P

DEFAULT_BACKBONE = 'holten2014'


def _backbone_compute_batch(backbone):
    """Return the compute_batch callable of the requested backbone EoS."""
    if backbone == 'holten2014':
        from holten_eos.core import compute_batch
    elif backbone == 'caupin2019':
        from caupin_eos.core import compute_batch
    elif backbone == 'duska2020':
        from duska_eos.core import compute_batch
    else:
        raise ValueError(
            f"unknown Singh-2017 backbone {backbone!r}; "
            f"available: ['holten2014', 'caupin2019', 'duska2020']")
    return compute_batch


def _singh_property(T_K, f, P_Pa, A0, E_LDS_k, E_HDS_k, dv_HDS, nu, eps, T0):
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
    T0      : VFT singularity temperature of the HDS term (K)

    Returns
    -------
    Transport property value(s) in the units of A0.
    """
    HDS_term = (E_HDS_k + dv_HDS * P_Pa / P.k_B) / (T_K - T0)
    LDS_term = E_LDS_k / T_K
    exponent = (1.0 - f) * HDS_term + f * LDS_term
    return A0 * (T_K / P.T_ref) ** nu * np.exp(eps * exponent)


def _transport_from_fraction(T_K, f, P_Pa, pset):
    """Evaluate eta, D, tau_r from a fraction array and a parameter set."""
    T0 = pset['T0']
    out = {}
    for key in ('eta', 'D', 'tau_r'):
        c = pset[key]
        out[key] = _singh_property(T_K, f, P_Pa, c['A0'], c['E_LDS_k'],
                                   c['E_HDS_k'], c['dv_HDS'], c['nu'],
                                   c['eps'], T0)
    return out


def compute_properties(T_K, p_MPa, backbone=DEFAULT_BACKBONE):
    """
    Compute transport properties at a single (T, P) point.

    Parameters
    ----------
    T_K      : float — temperature in K
    p_MPa    : float — pressure in MPa
    backbone : str — two-state EoS supplying the LDS fraction:
               'holten2014' (default, published parameters),
               'caupin2019' or 'duska2020' (refitted parameters)

    Returns
    -------
    dict with keys:
        'eta'   : dynamic viscosity (Pa*s)
        'D'     : self-diffusion coefficient (m^2/s)
        'tau_r' : rotational correlation time (s)
        'f'     : LDS fraction from backbone model
    """
    pset = P.get_params(backbone)
    batch = _backbone_compute_batch(backbone)(
        np.atleast_1d(np.float64(T_K)), np.atleast_1d(np.float64(p_MPa)))
    f = float(np.asarray(batch['x']).ravel()[0])

    tr = _transport_from_fraction(T_K, f, p_MPa * 1e6, pset)
    return {'eta': tr['eta'], 'D': tr['D'], 'tau_r': tr['tau_r'], 'f': f}


def compute_batch(T_K, p_MPa, backbone=DEFAULT_BACKBONE):
    """
    Vectorized computation of transport + thermodynamic properties.

    Calls the backbone EoS compute_batch once to get the LDS fraction and
    all thermodynamic properties, then computes eta, D, tau_r vectorized.

    Parameters
    ----------
    T_K      : 1-D array — temperature in K
    p_MPa    : 1-D array — pressure in MPa (same length as T_K)
    backbone : str — 'holten2014' (default), 'caupin2019', or 'duska2020'

    Returns
    -------
    dict of 1-D arrays.  Contains:
        Transport: 'eta', 'D', 'tau_r'
        LDS fraction: 'f'  (same as 'x' from the backbone)
        All backbone thermodynamic properties passed through:
            rho, V, S, G, H, U, A, Cp, Cv, Kt, Ks, alpha, vel, x,
            rho_A, V_A, ..., rho_B, V_B, ...
    """
    pset = P.get_params(backbone)

    T_K = np.asarray(T_K, dtype=float)
    p_MPa = np.asarray(p_MPa, dtype=float)

    # Get all thermodynamic properties + LDS fraction from the backbone
    thermo = _backbone_compute_batch(backbone)(T_K, p_MPa)
    f = thermo['x']

    tr = _transport_from_fraction(T_K, f, p_MPa * 1e6, pset)

    # Build output: transport properties + pass-through thermodynamics
    result = dict(thermo)  # shallow copy
    result['eta'] = tr['eta']
    result['D'] = tr['D']
    result['tau_r'] = tr['tau_r']
    result['f'] = f

    return result
