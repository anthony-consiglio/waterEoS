"""
Parameters for Singh et al. (2017) two-state transport model.

The model equation (Eq. 1) for a generic transport property A is:

  A(T,P) = A0 * (T/Tref)^nu * exp{ eps * [(1-f)*(E_HDS/kB + dv_HDS*P/kB)/(T-T0) + f*E_LDS/kB/T] }

where f is the LDS fraction supplied by a two-state thermodynamic backbone
model.  The original paper fitted the parameters using the Holten et al.
(2014) fraction:

  L. P. Singh, B. Issenmann, F. Caupin, PNAS 114, 4312 (2017), Table 1.

Because the LDS fraction differs substantially between two-state EoS models
(e.g. at 273.15 K / 0.1 MPa: Holten 0.097, Caupin 0.264, Duska 0.227), the
parameters are backbone-specific.  The 'caupin2019' and 'duska2020' sets
below were refitted in this repo against the same experimental datasets as
the paper (viscosity: Singh 2017 Table S1 + Dehaoui 2015; self-diffusion:
Prielmeier 1988 + Harris & Newitt 1997; rotational correlation time:
Lang & Ludemann 1981 + Arnold & Ludemann 2002), using a common T0 per
backbone as in Table 1.  See scripts/refit_singh_transport.py and
references/data/ for data and fit provenance.
"""

# ---------------------------------------------------------------------------
# Common constants
# ---------------------------------------------------------------------------
T_ref = 273.15      # K, reference temperature
k_B = 1.380649e-23  # J/K, Boltzmann constant

# ---------------------------------------------------------------------------
# Per-backbone parameter sets
#
# Each entry: {property: dict(A0, E_LDS_k, E_HDS_k, dv_HDS, nu, eps)}
# plus 'T0' (common VFT singularity temperature of the HDS term).
# Units: A0 in SI units of the property (Pa*s, m^2/s, s); E_*_k = E/kB in K;
# dv_HDS in m^3.
# ---------------------------------------------------------------------------
BACKBONE_PARAMS = {
    # Published values, Table 1 (common T0 = 147.75 K)
    'holten2014': {
        'T0': 147.75,
        'eta':   {'A0': 38.75e-6,  'E_LDS_k': 2262.0, 'E_HDS_k': 421.9,
                  'dv_HDS': 2.44e-30, 'nu': 0.5,  'eps': +1},
        'D':     {'A0': 40330e-12, 'E_LDS_k': 1984.0, 'E_HDS_k': 402.2,
                  'dv_HDS': 1.79e-30, 'nu': 0.5,  'eps': -1},
        'tau_r': {'A0': 86.2e-15,  'E_LDS_k': 2585.0, 'E_HDS_k': 395.0,
                  'dv_HDS': 1.62e-30, 'nu': -0.5, 'eps': +1},
    },
    # Refitted in this repo against the original experimental datasets
    # (eta: Singh Table S1 + Dehaoui 2015 + 6 IAPWS-2008 ambient anchors,
    # N=184; D: Prielmeier 1988 + Harris 1997 + 3 Dehaoui power-law points,
    # N=157; tau_r: Lang 1981 + Arnold 2002, N=101), joint fit with common
    # T0 per backbone, same weighting convention for all backbones.  Total
    # reduced chi^2: holten2014 1.07, caupin2019 1.40, duska2020 1.52.
    # See scripts/refit_singh_transport.py and
    # references/data/singh_refit_results.json for provenance.
    'caupin2019': {
        'T0': 145.2547,
        'eta':   {'A0': 2.241002e-05, 'E_LDS_k': 1970.90, 'E_HDS_k': 435.66,
                  'dv_HDS': 3.880858e-30, 'nu': 0.5,  'eps': +1},
        'D':     {'A0': 6.897980e-08, 'E_LDS_k': 1750.41, 'E_HDS_k': 430.82,
                  'dv_HDS': 2.714239e-30, 'nu': 0.5,  'eps': -1},
        'tau_r': {'A0': 4.356403e-14, 'E_LDS_k': 2143.12, 'E_HDS_k': 417.46,
                  'dv_HDS': 2.778246e-30, 'nu': -0.5, 'eps': +1},
    },
    'duska2020': {
        'T0': 141.0512,
        'eta':   {'A0': 2.051995e-05, 'E_LDS_k': 2094.41, 'E_HDS_k': 471.26,
                  'dv_HDS': 4.154132e-30, 'nu': 0.5,  'eps': +1},
        'D':     {'A0': 7.261842e-08, 'E_LDS_k': 1830.75, 'E_HDS_k': 461.54,
                  'dv_HDS': 2.861620e-30, 'nu': 0.5,  'eps': -1},
        'tau_r': {'A0': 4.257861e-14, 'E_LDS_k': 2261.35, 'E_HDS_k': 448.51,
                  'dv_HDS': 2.996342e-30, 'nu': -0.5, 'eps': +1},
    },
}


def get_params(backbone):
    """Return the parameter set for a backbone, with a clear error if absent."""
    if backbone not in BACKBONE_PARAMS:
        raise ValueError(
            f"unknown Singh-2017 backbone {backbone!r}; "
            f"available: {sorted(BACKBONE_PARAMS)}")
    pset = BACKBONE_PARAMS[backbone]
    if pset is None:
        raise ValueError(
            f"Singh-2017 parameters for backbone {backbone!r} have not been "
            f"fitted yet (see scripts/refit_singh_transport.py)")
    return pset


# ---------------------------------------------------------------------------
# Backward-compatible module-level constants (holten2014 backbone).
# Kept because core.py and external code import these directly.
# ---------------------------------------------------------------------------
_H = BACKBONE_PARAMS['holten2014']

T_0 = _H['T0']          # K, VFT singularity temperature of the HDS term

A0_eta = _H['eta']['A0']
E_LDS_k_eta = _H['eta']['E_LDS_k']
E_HDS_k_eta = _H['eta']['E_HDS_k']
dv_HDS_eta = _H['eta']['dv_HDS']
nu_eta = _H['eta']['nu']
eps_eta = _H['eta']['eps']

A0_D = _H['D']['A0']
E_LDS_k_D = _H['D']['E_LDS_k']
E_HDS_k_D = _H['D']['E_HDS_k']
dv_HDS_D = _H['D']['dv_HDS']
nu_D = _H['D']['nu']
eps_D = _H['D']['eps']

A0_tau = _H['tau_r']['A0']
E_LDS_k_tau = _H['tau_r']['E_LDS_k']
E_HDS_k_tau = _H['tau_r']['E_HDS_k']
dv_HDS_tau = _H['tau_r']['dv_HDS']
nu_tau = _H['tau_r']['nu']
eps_tau = _H['tau_r']['eps']
