"""
Gartner et al. (2020): two-state equation of state of a SCAN-based
deep-neural-network model water.

MODEL WATER, NOT REAL WATER.  The equation describes a classical-nuclei
deep-neural-network potential trained on SCAN density-functional energies
and forces.  The paper notes that this model is about 47 K too warm relative
to real water (structure, density maximum, ice Ih melting near 312.5 K).

Form (Eqs. 2-5 of the paper; molar Gibbs energy, x = fraction of the
low-density structure B):

    G      = G_A + x G_BA + G_mix
    G_A    = R Tc sum c_mn dT^m dP^n
    G_BA   = R T lambda (dT + a dP + b dT dP)
    G_mix  = R T [x ln x + (1 - x) ln(1 - x) + (2 + w0 dP)/(T/Tc) x (1 - x)]

with dT = (T - Tc)/Tc and dP = (P - Pc)/Pc.  Note that the pressure is
reduced by Pc itself, not by rho_c R Tc as in Biddle et al. (2017).  The
non-ideal term is R Tc (2 + w0 dP) x (1 - x), independent of T.

All parameters are from Table 1 (p. 5), checked against the page image.
The molar mass is 18.015 g/mol (Methods: "we used water's average molar
mass of 18.015 g/mol for analysis").  The Gibbs energy keeps the paper's
zero (c00 carries the DFT energy zero, about -1549 kJ/mol), so G, H, U and A
are not comparable with real water; S has no absolute meaning either.

Fitted to multithermal-multibaric simulation data, 240-330 K and
1-3000 bar.  The paper's own results used as checks (p. 3):
liquid-liquid critical point Tc = 224 +- 3 K, Pc = 2687 +- 68 bar,
rho_c = 1072 +- 16 kg/m3; coexisting densities at T/Tc = 0.965:
rho_HDL = 1102 kg/m3, rho_LDL = 1047 kg/m3.

Reference: T. E. Gartner III, L. Zhang, P. M. Piaggi, R. Car,
A. Z. Panagiotopoulos and P. G. Debenedetti, "Signatures of a liquid-liquid
transition in an ab initio deep neural network model for water",
PNAS 117, 26040 (2020), doi:10.1073/pnas.2015440117.
"""

import numpy as np

from watereos.two_state_general import (
    M_H2O, R, compute_two_state, poly_field, times_T,
)

KEY = 'gartner2020_scan'

# Table 1
Tc = 224.0                 # K
Pc = 2687e5                # Pa (2687 bar)
LAMBDA = 1.660
A = 0.3603
B = -0.2694
W0 = 0.1268
C = {
    (0, 0): -831.7,
    (0, 1): 2.091,
    (0, 2): -0.01794,
    (1, 1): 0.2733,
    (2, 0): -6.066,
    (1, 2): -0.1736,
    (2, 1): -0.1816,
    (3, 0): -0.8471,
    (2, 2): 0.4378,
    (3, 1): 0.3749,
    (4, 0): 1.968,
    (2, 3): 0.1490,
    (3, 2): -0.4847,
    (3, 3): -0.1868,
}

M = M_H2O

# Fitted range (Methods)
T_MIN, T_MAX = 240.0, 330.0     # K
P_MIN, P_MAX = 0.1, 300.0       # MPa


def terms(T, P_Pa):
    """G_A, G_BA and the non-ideal term, as fields in J/mol."""
    u = (T - Tc) / Tc
    v = (P_Pa - Pc) / Pc
    GA = poly_field(C, u, v, 1.0 / Tc, 1.0 / Pc, R * Tc)
    inner = poly_field({(1, 0): 1.0, (0, 1): A, (1, 1): B}, u, v,
                       1.0 / Tc, 1.0 / Pc, R * LAMBDA)
    dG = times_T(inner, T)
    W = poly_field({(0, 0): 2.0, (0, 1): W0}, u, v, 1.0 / Tc, 1.0 / Pc,
                   R * Tc)
    return GA, dG, W


def compute_batch(T_K, P_MPa, branch='stable'):
    """All two-state properties on flat (T, P) arrays; see module docstring."""
    return compute_two_state(T_K, P_MPa, terms, M=M, branch=branch)


def critical_point():
    """Liquid-liquid critical point (T in K, P in MPa, rho in kg/m3)."""
    out = compute_batch(np.array([Tc]), np.array([Pc / 1e6]))
    return {'T_K': Tc, 'p_MPa': Pc / 1e6, 'rho': float(out['rho'][0])}


MODELS = {KEY: compute_batch}
BRANCH_MODELS = (KEY,)
