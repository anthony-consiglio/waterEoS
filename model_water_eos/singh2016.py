"""
Singh, Biddle, Debenedetti & Anisimov (2016): two-structure equation of
state of TIP4P/2005 model water.

MODEL WATER, NOT REAL WATER.  The equation was fitted to molecular-dynamics
simulations of the rigid, classical TIP4P/2005 potential (Abascal & Vega
2005), mostly NVT runs of 216 molecules along isochores.  Its liquid-liquid
critical point, densities and response functions are those of the model.

Form (Eqs. 1-5, pp. 144504-3 and -4; molar Gibbs energy, x = fraction of the
low-density structure B, A = high-density structure):

    G      = G_A + x G_BA + R T [x ln x + (1 - x) ln(1 - x) + w x (1 - x)]
    G_A    = R Tc sum c_mn dT^m dP^n                                (Eq. 2)
    G_BA   = R T lambda (dT + a dP + b dT dP)                       (Eq. 3)
    w      = (2 + w0 dP) / (T / Tc)                                 (Eq. 5)

with dT = (T - Tc)/Tc and dP = (P - Pc)/(rho_c R Tc), rho_c the molar
critical density.  Eq. 3 is reduced by R T (ln K = -G_BA/(R T)), not by
R Tc as in Biddle et al. (2017).  The non-ideal term R T w x(1 - x) equals
R Tc (2 + w0 dP) x(1 - x): an enthalpy of mixing only ("energy-driven"
transition).  The x of the paper is already the low-density fraction.

Parameters, Table I (p. 144504-5), every value and sign checked against a
400 dpi render of the page.  The c_mn are "made dimensionless by the
critical parameters" (table footnote), i.e. G_A/(R Tc) = sum c_mn dT^m dP^n:

    Tc = 182 K, Pc = 170 MPa, rho_c = 1017 kg/m3
    lambda = 1.407, a = 0.171, b = -0.100, w0 = 0.0717
    c01 = 0.8617      c02 = -0.003412   c11 = 0.01351    c20 = -5.3481
    c12 = 0.000493    c21 = 0.1094      c30 = 1.3293     c22 = -0.02129
    c31 = -0.02446    c40 = -0.13173    c23 = 0.003687   c32 = 0.01229
    c33 = -0.003513

The paper prints no molar mass; TIP4P/2005 is H2O and M = 18.015268 g/mol is
used, which enters only the conversion rho_c R Tc = 85.43 MPa of the
reduced pressure.  c00 and c10 are not printed (they would only set the
zeros of G and S) and are taken as zero: G, H, U, A and S keep this
arbitrary zero (G_A = 0 and S_A = 0 at the critical point) and are not
comparable with real water or with other models.

Fitted range: 216-molecule NVT isochores at 960-1120 kg/m3 from 300 K down
to 180-190 K (pressures from about -80 to 400 MPa) and NPT runs at 0.1 MPa,
200-300 K.  The authors' validity region (dotted contour of Fig. 6) is
180-300 K, from -80 MPa (and densities >= 960 kg/m3) up to about 240 MPa at
180 K and 335 MPa at 300 K.  The paper states the form becomes increasingly
inaccurate below 960 kg/m3 and at negative pressure (superseded there by
Biddle et al. 2017, module ``biddle2017``) and that it deviates from the
Sumi-Sekino densities below about 200 K.

Checks against the paper (see tests/test_model_water_singh2016.py):
- Liquid-liquid critical point, Table I and p. 144504-8: 182 K, 170 MPa,
  1017 kg/m3.  At (Tc, Pc), G_BA = 0 and w = 2, so x = 1/2 and the equation
  gives 1017.08 kg/m3 (rho_c divided by c01 + lambda a / 2 + w0 / 4
  = 0.99992).  Two liquid branches exist below Tc on the line G_BA = 0 and
  merge at Tc.
- p. 144504-8: the TSEOS places the C_P maximum at 0.1 MPa within about 5 K
  of 220 K and the kappa_T maximum within about 5 K of 230 K.  Computed:
  224.4 K and 233.7 K.
- Fig. 3(b) (TSEOS curves, figure reading): kappa_T maxima about
  0.00165 1/MPa near 199 K (120 MPa), 0.00124 near 205 K (100 MPa),
  0.00095 near 213 K (70 MPa), 0.00080 near 222 K (40 MPa) and 0.00067
  near 233 K (0.1 MPa).  Computed: 0.001642 at 198.7 K, 0.001238 at 204.8 K,
  0.000935 at 213.7 K, 0.000776 at 222.4 K, 0.000660 at 233.7 K.
- Fig. 6 (figure reading): density maximum at about 277 K on the 0.1 MPa
  isobar; computed 277.2 K (1000.9 kg/m3).
- Fig. 1 (figure reading): liquid-liquid transition at about 228 MPa at
  160 K; G_BA = 0 gives 226.4 MPa.
- Biddle et al. (2017), Fig. 3 (dotted extrapolations of this equation,
  figure reading): about 855 kg/m3 at 300 K and 890 kg/m3 at 200 K on
  the -300 MPa isobar, far outside the validity region; computed 855.1 and
  893.2 kg/m3.

Reference: R. S. Singh, J. W. Biddle, P. G. Debenedetti and
M. A. Anisimov, "Two-state thermodynamics and the possibility of a
liquid-liquid phase transition in supercooled TIP4P/2005 water",
J. Chem. Phys. 144, 144504 (2016), doi:10.1063/1.4944986.
"""

import numpy as np

from watereos.two_state_general import (
    M_H2O, R, compute_two_state, poly_field, times_T,
)

KEY = 'singh2016_tip4p2005'

M = M_H2O                  # kg/mol (TIP4P/2005 is H2O; not printed)

# Table I
Tc = 182.0                 # K
Pc = 170e6                 # Pa
RHO_C = 1017.0             # kg/m3
LAMBDA = 1.407
A = 0.171
B = -0.100
W0 = 0.0717
C = {
    (0, 1): 0.8617,
    (0, 2): -0.003412,
    (1, 1): 0.01351,
    (2, 0): -5.3481,
    (1, 2): 0.000493,
    (2, 1): 0.1094,
    (3, 0): 1.3293,
    (2, 2): -0.02129,
    (3, 1): -0.02446,
    (4, 0): -0.13173,
    (2, 3): 0.003687,
    (3, 2): 0.01229,
    (3, 3): -0.003513,
}

# Pressure scale of the reduced variables: rho_c (molar) R Tc, about 85.43 MPa
P_SCALE = RHO_C / M * R * Tc   # Pa

# Validity region (Fig. 6) and simulated range
T_MIN, T_MAX = 180.0, 300.0     # K
P_MIN, P_MAX = -80.0, 400.0     # MPa


def terms(T, P_Pa):
    """G_A, G_BA and the non-ideal term R T w, as fields in J/mol."""
    u = (T - Tc) / Tc
    v = (P_Pa - Pc) / P_SCALE
    GA = poly_field(C, u, v, 1.0 / Tc, 1.0 / P_SCALE, R * Tc)
    inner = poly_field({(1, 0): 1.0, (0, 1): A, (1, 1): B}, u, v,
                       1.0 / Tc, 1.0 / P_SCALE, R * LAMBDA)
    dG = times_T(inner, T)
    W = poly_field({(0, 0): 2.0, (0, 1): W0}, u, v, 1.0 / Tc,
                   1.0 / P_SCALE, R * Tc)
    return GA, dG, W


def compute_batch(T_K, P_MPa, branch='stable'):
    """All two-state properties on flat (T, P) arrays; see module docstring."""
    return compute_two_state(T_K, P_MPa, terms, M=M, branch=branch)


def llt_pressure(T_K):
    """Pressure (MPa) of G_BA = 0: the liquid-liquid transition below Tc,
    the Widom line above it."""
    u = (np.asarray(T_K, dtype=float) - Tc) / Tc
    return (Pc - u / (A + B * u) * P_SCALE) / 1e6


def critical_point():
    """Liquid-liquid critical point (T in K, P in MPa, rho in kg/m3)."""
    out = compute_batch(np.array([Tc]), np.array([Pc / 1e6]))
    return {'T_K': Tc, 'p_MPa': Pc / 1e6, 'rho': float(out['rho'][0])}


MODELS = {KEY: compute_batch}
BRANCH_MODELS = (KEY,)
