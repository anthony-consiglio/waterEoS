"""
Parameters for Singh et al. (2017) two-state transport model.

All values from Table 1 of:
  L. P. Singh, B. Issenmann, F. Caupin, PNAS 114, 4312 (2017).

The model equation (Eq. 1) for a generic transport property A is:

  A(T,P) = A0 * (T/Tref)^nu * exp{ eps * [(1-f)*(E_HDS/kB + dv_HDS*P/kB)/(T-T0) + f*E_LDS/kB/T] }

where f = x (LDS fraction from the backbone two-state thermodynamic model).
"""

# ---------------------------------------------------------------------------
# Common parameters (shared by all three transport properties)
# ---------------------------------------------------------------------------
T_ref = 273.15      # K, reference temperature
T_0 = 147.75        # K, VFT-like singularity temperature
k_B = 1.380649e-23  # J/K, Boltzmann constant


# ---------------------------------------------------------------------------
# Viscosity (eta)  —  units: Pa*s
# ---------------------------------------------------------------------------
A0_eta = 38.75e-6       # Pa*s  (paper: 38.75 uPa*s)
E_LDS_k_eta = 2262.0    # K  (E_LDS / k_B)
E_HDS_k_eta = 421.9     # K  (E_HDS / k_B)
dv_HDS_eta = 2.44e-30   # m^3
nu_eta = 0.5
eps_eta = +1


# ---------------------------------------------------------------------------
# Self-diffusion coefficient (D)  —  units: m^2/s
# ---------------------------------------------------------------------------
A0_D = 40330e-12         # m^2/s  (paper: 40,330 um^2/s = 4.0330e-8 m^2/s)
E_LDS_k_D = 1984.0      # K
E_HDS_k_D = 402.2       # K
dv_HDS_D = 1.79e-30     # m^3
nu_D = 0.5
eps_D = -1


# ---------------------------------------------------------------------------
# Rotational correlation time (tau_r)  —  units: s
# ---------------------------------------------------------------------------
A0_tau = 86.2e-15        # s  (paper: 86.2 fs)
E_LDS_k_tau = 2585.0     # K
E_HDS_k_tau = 395.0      # K
dv_HDS_tau = 1.62e-30    # m^3
nu_tau = -0.5
eps_tau = +1
