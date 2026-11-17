"""MODEL WATER, NOT REAL WATER. Two-state Gibbs fit for the monatomic ML-BOP
potential from Zorzi, Neophytou and Sciortino, Mol. Phys. 122, e2407025
(2024), doi:10.1080/00268976.2024.2407025.

Equations (1)-(4) and (A1), in molar form, use t = T/Tc and p = P/Pc:

    G_A = gA(p) + t gB(p) + t ln(t) gC(p)
    dG = G_B - G_A = sum a_ij (t - 1)**i (p - 1)**j
    W = R Tc [2 + omega0 (p - 1)]
    G = G_A + x dG + R T [x ln(x) + (1 - x) ln(1 - x)] + W x (1 - x)

Each gA, gB, gC is quadratic in p. The energy coefficients are in kJ/mol
and omega0 is dimensionless (Table A1, p. 7). Eq. (4) fixes a00 = 0.
Here x is the low-density fraction psi. The energy and entropy use the
reference implicit in the printed Gibbs fit. The mass is 18.0153 g/mol
per ML-BOP site, as in the local Dhabal et al. (2024) simulation input.
The fixed critical coordinates are 181.5 K and 175 MPa from the central
values in ref. 20 (Neophytou and Sciortino, J. Chem. Phys. 160, 114502).

The liquid data span 100-270 K and 0.1-700 MPa, with the low-temperature
data confined to the HDL side by crystallization. The shared branch
solver follows the metastable liquid minima as well as the equilibrium
liquid. The authors caution that the pure-state density anomalies grow
under extrapolation into regions without data.

Using the printed omega0 and natural logarithms gives psi = 0.497 and
0.0756 at 250 K and 0.1 and 700 MPa, respectively (Fig. 3, about 0.50
and 0.075). At 100 K and 700 MPa, Figs. 2 and 4 give about 1256 kg/m3,
-38.05 kJ/mol and 0.125e-4 /bar for rho, U and kappa_T, versus computed
1257.8 kg/m3, -38.039 kJ/mol and 0.1197e-4 /bar.
The printed fit gives 1022.2 kg/m3 at 270 K and 0.1 MPa versus about
997 in Fig. 2a. It yields 1040.4 kg/m3 at the critical point versus
about 1020 in Fig. 1, and at 150 K yields coexistence densities
986.6 / 1149.7 kg/m3 (LDL / HDL) versus about 974 / 1132 in Fig. 1.
The source of these density discrepancies is unresolved. Interpreting
omega0 as per bar would alter the order-parameter curves, and changing
the log base or the fixed critical pressure fails other figure checks.
The evaluator preserves the printed coefficients and equations.
"""

import numpy as np

from watereos.two_state_general import (
    R, add_fields, compute_two_state, poly_field, scale_field, times_T,
)

KEY = 'zorzi2024_mlbop'
M = 0.0180153                  # kg/mol
Tc, Pc = 181.5, 175e6           # K, Pa
T_MIN, T_MAX = 100.0, 270.0      # K
P_MIN, P_MAX = 0.1, 700.0        # MPa

# Table A1. The g coefficients multiply ascending powers of P/Pc.
_G_A = {(0, 0): -41.06, (0, 1): 2.291, (0, 2): 0.02316}
_G_B = {(0, 0): 29.03, (0, 1): 0.4378, (0, 2): -0.04821}
_G_C = {(0, 0): -5.494, (0, 1): -0.4272, (0, 2): 0.05571}
_DELTA_G = {
    (1, 0): 2.729, (0, 1): 0.7031,
    (2, 0): -2.584, (1, 1): -0.3721, (0, 2): -0.06663,
    (3, 0): 2.239, (2, 1): 1.285, (1, 2): 0.08969, (0, 3): -0.003008,
}
_W = {(0, 0): 2.0, (0, 1): 1.174e-4}


def terms(T, P_Pa):
    """Return the fields G_A, dG and W in J/mol for the shared solver."""
    t = T / Tc
    p = P_Pa / Pc
    gA = poly_field(_G_A, t, p, 1.0 / Tc, 1.0 / Pc, 1e3)
    gB = poly_field(_G_B, t, p, 1.0 / Tc, 1.0 / Pc, 1e3)
    gC, _, gC_P, _, _, gC_PP = poly_field(
        _G_C, t, p, 1.0 / Tc, 1.0 / Pc, 1e3,
    )
    log_t = np.log(t)
    h = t * log_t
    h_T = (1.0 + log_t) / Tc
    h_TT = 1.0 / (T * Tc)
    thermal = (gC * h, gC * h_T, gC_P * h,
               gC * h_TT, gC_P * h_T, gC_PP * h)
    GA = add_fields(gA, scale_field(times_T(gB, T), 1.0 / Tc), thermal)
    dG = poly_field(_DELTA_G, t - 1, p - 1, 1.0 / Tc, 1.0 / Pc, 1e3)
    W = poly_field(_W, t - 1, p - 1, 1.0 / Tc, 1.0 / Pc, R * Tc)
    return GA, dG, W


def compute_batch(T_K, P_MPa, branch='stable'):
    """Compute mixture and pure-state properties on flat (T, P) arrays."""
    return compute_two_state(T_K, P_MPa, terms, M=M, branch=branch)


def critical_point():
    """Return the fitted liquid-liquid critical point in K, MPa and kg/m3."""
    GA, dG, W = terms(np.array([Tc]), np.array([Pc]))
    # At criticality x = 1/2 exactly. Density only needs the first derivative.
    volume = GA[2][0] + 0.5 * dG[2][0] + 0.25 * W[2][0]
    return {'T_K': Tc, 'p_MPa': Pc / 1e6, 'rho': float(M / volume)}


MODELS = {KEY: compute_batch}
BRANCH_MODELS = (KEY,)
