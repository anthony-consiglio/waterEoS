"""MODEL WATER, NOT REAL WATER. Classical-nuclei WAIL model water.

Weis et al., J. Chem. Phys. 157, 024502 (2022), doi:10.1063/5.0099520.
Section II B describes the simulation mass substitution for a longer time
step. The published results are on an H2O mass basis (18.015268 g/mol).

Eqs. 1-4 (article p. 2) give the molar Gibbs energy in J/mol:

    G = G_A + x G_BA + R T [x ln x + (1 - x) ln(1 - x)] + W x (1 - x)
    G_A = R Tc sum c_jk u^j v^k
    G_BA = R T lambda (u + a v + b u v)
    W = R Tc (2 + omega0 v)

Here u = (T - Tc)/Tc, v = (P - Pc)/Pc, and x is the fraction of the
low-density structure B. Derivatives use K and Pa. Batch inputs are K and
MPa, and outputs use the package's mass-specific units. The background
coefficients retain the fit's energy reference (c00) and entropy reference
(no c10 term). No absolute reference for either is specified.

The fit uses density and total energy along isobars and isochores. The
positive-pressure isobar data in Fig. 1A span 215-260 K and 10-80 MPa.
The fit omitted the liquid-vapor spinodal (Section II A). All parameters
below are from the rendered Table III (Appendix A, article p. 6). In
particular, c40 is -49.5 times 10**-1 and c23 is +7.54 times 10**-2.

The TSEOS critical point is 208.5 K, 35.63 MPa, 965.57 kg/m3. The density
agrees with the red marker at about 965 kg/m3 in Fig. 1A. The independent
Ising estimate in the paper is 210.5 K and 36.5 MPa. The Fig. 1A binodal
at 200 K reads about 1048/908 kg/m3 (HDL/LDL), versus 1050.47/909.21 at
53.42438 MPa from the equation. At 10 MPa, Fig. 1C gives x about 0.81
at 200 K and 0.23 at 260 K (computed 0.8090 and 0.2278).

The printed Table III and Fig. 1A diverge at high pressures: at 250 K,
80 MPa the table gives 998.21 kg/m3 and Kt = -504.70 MPa, while the plot
shows about 1079 kg/m3. The table is used verbatim and the figure check
is an expected failure. A sign change or smaller exponent for c23 alone
does not reconcile the curves. Negative Kt indicates mechanical
instability of the printed EOS; the branch labels refer to minima in x.
On a 100 by 100 grid over the reported isobar fit bounds, 4007 points have
negative Kt on the stable branch selected in x.
The 273.15 K, 0.1 MPa probe and the liquid-liquid critical and coexistence
checks extrapolate beyond the isobar fit data.
"""

from watereos.two_state_general import (
    M_H2O, R, compute_two_state, poly_field, times_T,
)

KEY = 'weis2022_wail'
M = M_H2O

# Table III (Appendix A).
Tc = 208.5
Pc = 356.3e5  # 356.3 bar in Pa
LAMBDA = 1.795
A = 7.88e-2
B = -6.94e-2
W0 = 5.10e-2
C = {
    (0, 0): -27.9,
    (0, 1): 3.00e-1,
    (0, 2): 2.20e-3,
    (1, 1): 7.54e-2,
    (2, 0): -6.62,
    (1, 2): 1.22e-2,
    (2, 1): 2.01e-2,
    (3, 0): 2.10,
    (2, 2): 1.03e-2,
    (3, 1): 1.17e-2,
    (4, 0): -4.95,
    (2, 3): 7.54e-2,
    (3, 2): 9.82e-4,
    (3, 3): -8.92e-4,
}

T_MIN, T_MAX = 215.0, 260.0  # K
P_MIN, P_MAX = 10.0, 80.0    # MPa


def terms(T, P_Pa):
    """Return G_A, G_BA and W as molar Gibbs fields."""
    u = (T - Tc) / Tc
    v = (P_Pa - Pc) / Pc
    GA = poly_field(C, u, v, 1.0 / Tc, 1.0 / Pc, R * Tc)
    inner = poly_field({(1, 0): 1.0, (0, 1): A, (1, 1): B}, u, v,
                       1.0 / Tc, 1.0 / Pc, R * LAMBDA)
    W = poly_field({(0, 0): 2.0, (0, 1): W0}, u, v, 1.0 / Tc, 1.0 / Pc, R * Tc)
    return GA, times_T(inner, T), W


def compute_batch(T_K, P_MPa, branch='stable'):
    """Evaluate the WAIL fit on flat arrays in K and MPa."""
    return compute_two_state(T_K, P_MPa, terms, M=M, branch=branch)


MODELS = {KEY: compute_batch}
BRANCH_MODELS = (KEY,)
