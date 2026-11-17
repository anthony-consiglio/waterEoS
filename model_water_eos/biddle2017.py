"""
Biddle et al. (2017): two-structure equation of state of TIP4P/2005 model
water with a liquid-vapor spinodal, covering supercooled and deeply
stretched states.

MODEL WATER, NOT REAL WATER.  The equation was fitted to molecular-dynamics
simulations of the rigid, classical TIP4P/2005 potential (Princeton
isochores, 216 molecules, and Madrid isobars and isotherms).  Its
liquid-liquid critical point, spinodal, densities and response functions
are those of the model.

Form (Eqs. 1-5, 9, 15 and 16, pp. 034502-2 to -4; molar Gibbs energy,
x = fraction of the low-density structure B, A = high-density structure;
a hat denotes reduction by R Tc for energies and rho_c R Tc for pressures):

    G       = G_A + x G_BA + R T [x ln x + (1 - x) ln(1 - x) + w x (1 - x)]
    G_A/(R Tc)  = Gs + sum c_mn dT^m dP^n                           (Eq. 3)
    Gs      = A(T) (p - p_s(T))^(3/2)                                (Eq. 9)
    A(T)    = A0 + A1 dT                                            (Eq. 15)
    p_s(T)  = S0 + S1 dT + S2 dT^2                                  (Eq. 16)
    G_BA/(R Tc) = lambda (dT + a dP + b dT dP + d dP^2 + f dT^2)     (Eq. 4)
    w       = (2 + w0 dP) / (T / Tc)                                 (Eq. 5)

with dT = (T - Tc)/Tc, dP = (P - Pc)/(rho_c R Tc), p = P/(rho_c R Tc) and
rho_c the molar critical density.  The spinodal pressure p_s is absolute
(no Pc offset), unlike dP; see the caveats.  The non-ideal term R T w x(1-x)
equals R Tc (2 + w0 dP) x(1 - x).  The x of the paper is already the
low-density fraction.

The spinodal term (P - P_s)^(3/2) is not a polynomial and is implemented
here with its analytic first and second derivatives in T and P.  It is
real only for P > P_s(T); below the liquid-vapor spinodal every property
is NaN (G_A does not exist there).  kappa_T, C_P and alpha_P diverge as
(P - P_s)^(-1/2) on approach, as Eqs. 10-13 state.

Parameters, Table I (p. 034502-6), every value and sign checked against a
400 dpi render of the page, with the erratum applied:

    Tc = 182 K, Pc = 170 MPa, rho_c = 1017 kg/m3
    lambda = 1.55607, a = 0.154014, b = 0.125093, d = 0.00854418,
    f = 1.14576, w0 = 0.03
    A0 = -0.0547873, A1 = -0.0822462
    S0 = -5.40845, S1 = 5.56087, S2 = -2.5205  (S1, S2 from the erratum)
    c02 = -0.00261876   c11 = 0.257249     c20 = -6.30589
    c03 = 0.000605678   c12 = 0.0248091    c21 = -0.0400033
    c30 = 2.18819       c13 = -0.000994166 c22 = -0.00840543
    c31 = 0.0719058     c40 = -0.256674
    c01 = 1.096205 (not printed; derived here, see the caveats)

Erratum (J. Chem. Phys. 148, 069901 (2018), page image checked): Table I's
S1 = 0.0305542 and S2 = -7.61e-5 had not been made dimensionless; the
correct values are S1 = 5.56087 and S2 = -2.5205 (the old values are the
new ones divided by Tc and Tc^2).  S0 is unchanged.

The paper prints no molar mass; TIP4P/2005 is H2O and M = 18.015268 g/mol
is used.  It enters the pressure scale rho_c R Tc = 85.43 MPa.

Caveats and how they were resolved:
- Reduced spinodal pressure.  Neither the paper nor the erratum says whether
  p_s is P_s/(rho_c R Tc) or (P_s - Pc)/(rho_c R Tc).  The supplement (Fig.
  S2 caption) gives the spinodal at 230 K as -351 MPa.  The absolute reading
  with the erratum coefficients gives -351.7 MPa; the Pc-offset reading would
  give -181.7 MPa and the uncorrected Table I values about -461 MPa.  The
  absolute reading also matches the dashed spinodal of Fig. 9 (about
  -226 MPa at 320 K, -265 MPa at 280 K, -330 MPa at 240 K; computed -225.6,
  -268.7, -332.5 MPa).
- Reduction of G_BA.  Eq. 4 writes the hatted G_BA, and the paper defines
  the hat as G/(R Tc) (p. 034502-3), so G_BA = R Tc lambda (...), unlike
  Singh et al. (2016), Eq. 3, where G_BA/(R T) = lambda (...).  This matches
  the paper's statement that lambda b, lambda d and lambda f are the A-B
  differences in alpha_P, kappa_T and C_P at the critical point.  The
  R T reading fails the supplement's Fig. S1: at 320 K it gives x = 0.16,
  0.08 and 0.04 on the -150, 0.1 and 150 MPa isobars against the plotted
  0.26, 0.17 and 0.11 (the R Tc reading gives 0.259, 0.172, 0.110).
- c01 is missing from Table I (as are c00 and c10).  c00 and c10 only set
  the zeros of G and S and are taken as zero.  c01 sets the volume scale and
  cannot be zero: without it the molar volume at the critical point would be
  negative (-0.096 / rho_c), because the spinodal term contributes
  (3/2) A0 (p_c - S0)^(1/2) = -0.2235.  It is fixed here by requiring the
  density at the critical point (x = 1/2) to be rho_c:
      c01 = 1 - (3/2) A0 (Pc/(rho_c R Tc) - S0)^(1/2) - lambda a / 2 - w0 / 4
          = 1.096205.
  Singh et al. (2016) obey the same closure with their printed c01 (to
  1e-4).  With this c01 the equation reproduces, to figure-reading
  accuracy, the supplement's Fig. S2 (density maximum 1046.5 kg/m3 at
  259 K on the 100 MPa isobar against about 1046 kg/m3 near 255-260 K; the
  pure-A background 1141.3 kg/m3 at 180 K and 1084.7 kg/m3 at 320 K against
  about 1141 and 1085) and the 0.1 MPa density curve of Fig. 2 (1000.2 kg/m3
  maximum at 276 K, 996.3 kg/m3 at 300 K).  This is a reconstruction, not a
  printed value.
- Zero of energy and entropy: G, H, U, A and S keep the arbitrary zero
  c00 = c10 = 0 and are not comparable with real water or other models.

Fitted data: Princeton isochores 920-1160 kg/m3, 180-300 K; Madrid isobars
and isotherms -170 to 300 MPa, 195-320 K (p. 034502-4).  The validity box of
Fig. 1 spans about 180-320 K and from about -275 MPa (above the spinodal)
to about 400-540 MPa, and excludes the lowest-T, lowest-P corner (below
about 0 MPa at 180 K and -275 MPa at 240 K), where long C_P runs disagree
with short ones (p. 034502-8).

Checks against the paper (see tests/test_model_water_biddle2017.py):
- Liquid-liquid critical point (Table I, fixed input): 182 K, 170 MPa,
  1017 kg/m3.  x = 1/2 and G_BA = 0, w = 2 there; the density equals
  1017 kg/m3 by the c01 closure above, so it is not an independent check.
  Two liquid branches coexist below Tc on the line G_BA = 0.
- Liquid-vapor spinodal at 230 K: -351 MPa (supplement, Fig. S2 caption);
  computed -351.7 MPa.
- Supplement Fig. S2 (figure reading), 100 MPa isobar: kappa_T maximum about
  0.00139 1/MPa near 203 K, C_P maximum about 7900 J/(kg K) near 200 K;
  computed 0.001393 at 202.7 K and 7938 J/(kg K) at 200.2 K.  230 K
  isotherm: kappa_T maximum about 0.0007 1/MPa near +15 MPa, C_P minimum
  about 5650 J/(kg K) near -200 MPa and maximum about 6550 J/(kg K) near
  -35 MPa; computed 0.000714 at 24 MPa, 5655 at -235 MPa, 6572 at -28 MPa.
- Supplement Fig. S1 (figure reading), x at 180 K: 0.91, 0.87, 0.74 on the
  -150, 0.1, 150 MPa isobars; computed 0.908, 0.868, 0.735.
- Fig. 8 (figure reading): C_P maxima about 6480 J/(kg K) near 257 K at
  -170 MPa and about 6550 J/(kg K) near 230 K at -40 MPa; computed 6485 at
  257.1 K and 6552 at 230.7 K.
- Fig. 5 (figure reading): kappa_T maximum about 0.00068 1/MPa near 230 K
  at 0.1 MPa; computed 0.000683 at 229.8 K.

Reference: J. W. Biddle, R. S. Singh, E. M. Sparano, F. Ricci,
M. A. Gonzalez, C. Valeriani, J. L. F. Abascal, P. G. Debenedetti,
M. A. Anisimov and F. Caupin, "Two-structure thermodynamics for the
TIP4P/2005 model of water covering supercooled and deeply stretched
regions", J. Chem. Phys. 146, 034502 (2017), doi:10.1063/1.4973546;
erratum J. Chem. Phys. 148, 069901 (2018), doi:10.1063/1.5024509.
"""

import numpy as np

from watereos.two_state_general import (
    M_H2O, R, add_fields, compute_two_state, poly_field,
)

KEY = 'biddle2017_tip4p2005'

M = M_H2O                  # kg/mol (TIP4P/2005 is H2O; not printed)

# Table I
Tc = 182.0                 # K
Pc = 170e6                 # Pa
RHO_C = 1017.0             # kg/m3
LAMBDA = 1.55607
A = 0.154014
B = 0.125093
D = 0.00854418
F = 1.14576
W0 = 0.03
A0 = -0.0547873
A1 = -0.0822462
S0 = -5.40845
# Erratum (2018): Table I printed 0.0305542 and -7.61e-5 (per kelvin)
S1 = 5.56087
S2 = -2.5205

# Pressure scale of the reduced variables: rho_c (molar) R Tc, about 85.43 MPa
P_SCALE = RHO_C / M * R * Tc   # Pa

# c01 is not printed; fixed so that rho = rho_c at the critical point
# (x = 1/2, G_BA = 0, w = 2); see the module docstring.
C01 = 1.0 - (1.5 * A0 * np.sqrt(Pc / P_SCALE - S0)
             + 0.5 * LAMBDA * A + 0.25 * W0)

C = {
    (0, 1): C01,
    (0, 2): -0.00261876,
    (1, 1): 0.257249,
    (2, 0): -6.30589,
    (0, 3): 0.000605678,
    (1, 2): 0.0248091,
    (2, 1): -0.0400033,
    (3, 0): 2.18819,
    (1, 3): -0.000994166,
    (2, 2): -0.00840543,
    (3, 1): 0.0719058,
    (4, 0): -0.256674,
}

# Fitted data and validity box (Fig. 1); the spinodal bounds P from below
T_MIN, T_MAX = 180.0, 320.0     # K
P_MIN, P_MAX = -275.0, 400.0    # MPa


def spinodal_pressure(T_K):
    """Liquid-vapor spinodal pressure P_s(T) in MPa (Eq. 16, erratum)."""
    u = (np.asarray(T_K, dtype=float) - Tc) / Tc
    return (S0 + S1 * u + S2 * u * u) * P_SCALE / 1e6


def spinodal_field(T, P_Pa):
    """R Tc A(T) (p - p_s(T))^(3/2) and its derivatives, in J/mol.

    NaN where P <= P_s(T).
    """
    u = (T - Tc) / Tc
    p = P_Pa / P_SCALE
    a = A0 + A1 * u                     # A(T), reduced
    ps1 = S1 + 2.0 * S2 * u             # d p_s / d dT
    ps2 = 2.0 * S2                      # d2 p_s / d dT2
    z = p - (S0 + S1 * u + S2 * u * u)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.where(z > 0.0, z, np.nan)
        r = np.sqrt(z)
        g = a * z * r
        g_p = 1.5 * a * r
        g_pp = 0.75 * a / r
        g_u = A1 * z * r - 1.5 * a * r * ps1
        g_up = 1.5 * A1 * r - 0.75 * a / r * ps1
        g_uu = (-3.0 * A1 * r * ps1
                + a * (0.75 / r * ps1 ** 2 - 1.5 * r * ps2))
    s = R * Tc
    return (s * g, s * g_u / Tc, s * g_p / P_SCALE, s * g_uu / Tc ** 2,
            s * g_up / (Tc * P_SCALE), s * g_pp / P_SCALE ** 2)


def terms(T, P_Pa):
    """G_A (with the spinodal term), G_BA and R T w, as fields in J/mol."""
    u = (T - Tc) / Tc
    v = (P_Pa - Pc) / P_SCALE
    GA = add_fields(poly_field(C, u, v, 1.0 / Tc, 1.0 / P_SCALE, R * Tc),
                    spinodal_field(T, P_Pa))
    dG = poly_field({(1, 0): 1.0, (0, 1): A, (1, 1): B, (0, 2): D,
                     (2, 0): F}, u, v, 1.0 / Tc, 1.0 / P_SCALE,
                    R * Tc * LAMBDA)
    W = poly_field({(0, 0): 2.0, (0, 1): W0}, u, v, 1.0 / Tc,
                   1.0 / P_SCALE, R * Tc)
    return GA, dG, W


def compute_batch(T_K, P_MPa, branch='stable'):
    """All two-state properties on flat (T, P) arrays; see module docstring.

    NaN below the liquid-vapor spinodal.
    """
    return compute_two_state(T_K, P_MPa, terms, M=M, branch=branch)


def llt_pressure(T_K):
    """Pressure (MPa) of G_BA = 0: the liquid-liquid transition below Tc,
    the Widom line above it (root of Eq. 4 continuous with Pc at Tc)."""
    u = (np.asarray(T_K, dtype=float) - Tc) / Tc
    qa = D
    qb = A + B * u
    qc = u + F * u * u
    v = -2.0 * qc / (qb + np.sqrt(qb * qb - 4.0 * qa * qc))
    return (Pc + v * P_SCALE) / 1e6


def critical_point():
    """Liquid-liquid critical point (T in K, P in MPa, rho in kg/m3)."""
    out = compute_batch(np.array([Tc]), np.array([Pc / 1e6]))
    return {'T_K': Tc, 'p_MPa': Pc / 1e6, 'rho': float(out['rho'][0])}


MODELS = {KEY: compute_batch}
BRANCH_MODELS = (KEY,)
