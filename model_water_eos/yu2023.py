"""MODEL WATER, NOT REAL WATER. Hierarchical two-state fit for TIP4P/2005
from Yu, Shi and Tanaka, J. Phys. Chem. B 127, 3452-3462 (2023),
doi:10.1021/acs.jpcb.3c00869.

For Eqs. (1)-(6), define u = T/Tc - 1 and v = P/Pc - 1. The molar
Gibbs terms are G_A = (R Tc/n) sum c_ij u**i v**j for the disordered
state, dG = (R/n) sum a_ij u**i v**j for the tetrahedral state relative
to the disordered state, and W = (R Tc/n)(2 + J10 u + J01 v).
The mixture adds x dG, (R T/n)[x ln(x) + (1-x) ln(1-x)] and W x(1-x)
to G_A, with x the tetrahedral fraction s. Table 1 (p. 3456) gives a_ij
in K, c_ij and J_ij dimensionless, and n molecules per local structure.
The fitted fraction uses s = 1 - g_OO(3.5 angstrom) as its proxy.

The fit to density, compressibility, structure and dynamics uses seven
isobars from 0.1 to 180 MPa, with data spanning approximately 170-350 K.
We use the package H2O molar mass of 18.015268 g/mol. The unlisted
pressure-independent c_i0 are set to zero internally to check Gibbs
derivatives. These coefficients determine the caloric background and
the energy and entropy reference. The public thermodynamic outputs
are rho, V, Kt and alpha for the mixture and pure structures, and x
for the mixture. All depend only on the pressure-dependent Gibbs
background and the fitted mixing terms.

The dynamics (Eqs. 7-10 and Table 2, p. 3456) use the same equilibrium
condition with their own coefficients aD_ij (K) and JD_ij (dimensionless).
The dynamic cluster fraction sD is returned as f. Each dynamic quantity
uses log10(X) = log10(X0) + (E0a + P dVa/R + sD dEa)/T in the table's
units, with E0a and dEa in K. Numerically, MPa times cm3/mol gives
J/mol for the pressure correction. The activation volumes are used
with their printed units and their small effects cannot be resolved
from the figure. We interpret the Table 2 Arrhenius coefficients in
common-log space despite the exp notation of Eq. (10). The Price et al.
(1999) data plotted in Fig. 4A establish the base: 2.30e-9 m2/s at
298.15 K gives a plotted ordinate of 2.64 on the stated 1e6 m^-2 s scale.
The same convention is applied to both panels of Fig. 3. The evaluator
returns D in m2/s and tau_r in s. The paper supplies equilibrium dynamics,
so requests for missing or metastable liquid branches yield NaN for
D, tau_r and f.

Table 1 / Fig. 5 put Tc and Pc at 175.9 K and 177.2 MPa, with s and sD
both 1/2. The printed fit implies rho_c = 1023.44 kg/m3. At 0.1 MPa
Fig. 2 gives rho about 997 kg/m3 at 280 K (computed 996.58) and the
Fig. 1 caption gives s about 0.5 at 235 K (computed 0.506). Fig. 3
at 200 K gives log10(1/D in table units) about 6.7 and 4.6, and
log10(tau_r in ps) about 4.9 and 2.3, at 0.1 and 180 MPa respectively.
Computed pairs are (6.723, 4.624) and (4.945, 2.306). The calculated
ambient-pressure peak in d log10(1/D) / d(1000/T) is at 207.3 K,
compared with about 205 K (p. 3457) and about 207 K in the fit curve
of Fig. 4B.
"""

import numpy as np

from watereos.two_state_branch import select_branch
from watereos.two_state_general import M_H2O, R, compute_two_state, poly_field

KEY = 'yu2023_tip4p2005'
M = M_H2O
Tc, Pc = 175.9, 177.2e6        # K, Pa
N = 3.99925                   # molecules per local structure
T_MIN, T_MAX = 170.0, 350.0    # K
P_MIN, P_MAX = 0.1, 180.0      # MPa

# Table 1, p. 3456, with coefficient indices (temperature, pressure).
_DELTA_G_K = {
    (1, 0): 543.643, (0, 1): 246.771,
    (2, 0): 540.598, (1, 1): -1.36e-7, (0, 2): -0.230619,
}
_W = {(0, 0): 2.0, (1, 0): -2.33436, (0, 1): 0.110927}
_G_A = {
    (0, 1): 7.80027, (0, 2): -0.0830773, (0, 3): 0.00107958,
    (1, 1): 2.21e-6, (1, 2): -0.296790, (1, 3): 0.0204134,
    (2, 1): -0.0450466, (2, 2): 0.0253941, (2, 3): 0.00273835,
    (3, 1): 0.725173, (3, 2): 0.229689, (3, 3): 0.0116219,
}

# Table 2, p. 3456. Coefficient rows encode increasing powers of u,
# and columns encode increasing powers of v as required by polyval2d.
_DYNAMIC_DELTA_G_K = (
    (0.0, 150.441, -30.3707),
    (490.333, -0.00295333, 0.0),
    (3196.06, 0.0, 0.0),
)
_JD10, _JD01 = -0.0595336, 0.971780
# log10(X0), E0a (K), dVa (cm3/mol), dEa (K).
_INVERSE_DIFFUSION = (-0.350453, 892.570, 0.0147224, 845.731)
_ROTATION = (-2.62830, 858.807, 0.000119607, 1062.49)

_VOLUMETRIC_KEYS = ('rho', 'V', 'Kt', 'alpha')


def terms(T, P_Pa):
    """Return the fields G_A, dG and W in J/mol for the shared solver."""
    u = T / Tc - 1.0
    v = P_Pa / Pc - 1.0
    GA = poly_field(_G_A, u, v, 1.0 / Tc, 1.0 / Pc, R * Tc / N)
    dG = poly_field(_DELTA_G_K, u, v, 1.0 / Tc, 1.0 / Pc, R / N)
    W = poly_field(_W, u, v, 1.0 / Tc, 1.0 / Pc, R * Tc / N)
    return GA, dG, W


def _dynamic_fraction(T, P_MPa):
    u = T / Tc - 1.0
    v = P_MPa * 1e6 / Pc - 1.0
    dG_K = np.polynomial.polynomial.polyval2d(u, v, _DYNAMIC_DELTA_G_K)
    J_K = Tc * (2.0 + _JD10 * u + _JD01 * v)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction, _ = select_branch(dG_K / T, J_K / T, 'stable')
    return fraction


def _log10_arrhenius(T, P_MPa, fraction, params):
    log_x0, e0_K, dv_cm3_mol, de_K = params
    return log_x0 + (e0_K + P_MPa * dv_cm3_mol / R + fraction * de_K) / T


def _compute_thermodynamics(T_K, P_MPa, branch='stable'):
    """Evaluate the Gibbs derivatives with c_i0 = 0."""
    return compute_two_state(T_K, P_MPa, terms, M=M, N=N, branch=branch)


def compute_batch(T_K, P_MPa, branch='stable'):
    """Return volumetric properties, x, and equilibrium D, tau_r and f."""
    T, P = np.broadcast_arrays(np.atleast_1d(np.asarray(T_K, dtype=float)),
                               np.atleast_1d(np.asarray(P_MPa, dtype=float)))
    thermo = _compute_thermodynamics(T, P, branch)
    out = {key + suffix: thermo[key + suffix]
           for suffix in ('', '_A', '_B') for key in _VOLUMETRIC_KEYS}
    out['x'] = thermo['x']
    available = np.isfinite(out['x'])
    if 'metastable' in thermo:
        out['metastable'] = thermo['metastable']
        available &= ~out['metastable']
    fraction = np.where(available, _dynamic_fraction(T, P), np.nan)
    log_inv_D = _log10_arrhenius(T, P, fraction, _INVERSE_DIFFUSION)
    log_tau = _log10_arrhenius(T, P, fraction, _ROTATION)
    out.update(D=10.0 ** (-6.0 - log_inv_D),
               tau_r=10.0 ** (log_tau - 12.0),
               f=fraction)
    return out


def critical_point():
    """Return the fitted liquid-liquid critical point in K, MPa and kg/m3."""
    GA, dG, W = terms(np.array([Tc]), np.array([Pc]))
    # At criticality x = 1/2 exactly. Density only needs the first derivative.
    volume = GA[2][0] + 0.5 * dG[2][0] + 0.25 * W[2][0]
    return {'T_K': Tc, 'p_MPa': Pc / 1e6, 'rho': float(M / volume)}


MODELS = {KEY: compute_batch}
BRANCH_MODELS = (KEY,)
NAN_KEYS = {KEY: ()}
