"""MODEL WATER, NOT REAL WATER. Classical q-TIP4P/F liquid (reaction field).

Eltareb, Lopez, and Giovambattista, J. Chem. Phys. 160, 154510 (2024),
doi:10.1063/5.0200306. The Gaussian landscape and basin free energies
(Eqs. 4-9 and 16-19) give the molar Helmholtz energy

    F = Q1 - sigma2/(2*R*T) - R*T*Q3 - c2*T**2 - c3*T**3/2
        + 9*R*T*ln(A0/(R*T)),
    Q1 = E0 - b*sigma2,
    Q3 = alpha - a - b*E0 + b**2*sigma2/2.

Eq. (7) uses nine modes per molecule and A0 = 1 kJ/mol. The SI Table
S2 rows use v in cm**3/g, consistent with Figs. 3, 5 and 7. The table
does not state units. The sum in SI Eq. (12) uses a_i*v**i, following
the table column indices. E0 is in kJ/mol, sigma2 in (kJ/mol)**2, b in
mol/kJ, alpha and a are per-molecule dimensionless quantities, and ci
is in kJ/(mol K**i). H2O molar mass is 18.015268 g/mol. Dividing the
molar energy derivative by the molar mass converts P = -dF/dv to
MPa. The published factors and signs in Eqs. (22)-(27) are retained.
Energies and entropies retain the PEL convention above. The total
entropy can be negative because the intramolecular modes are classical.

SI Sec. V gives 200-280 K and 0.94-1.40 g/cm**3 for this fit. The
density edges span roughly -161 to 2579 MPa in total (98 to 2122 MPa
at 200 K and -161 to 2579 MPa at 280 K). Inversion picks the minimum
Gibbs energy among mechanically stable liquid roots in that interval,
or returns NaNs when none exists. Outside the fitted temperatures
this is an extrapolation. The Gaussian liquid approximation assumes
positive configurational entropy. Glass, crystal and vapor are omitted.

This implementation substitutes the seven printed polynomials directly.
At v = 0.88 cm**3/g, Eq. (29) yields 137.2 K, consistent with about
137 K in Fig. 7(b). At 200 K and 1.20 g/cm**3, it gives 448.42 MPa,
versus about 435 MPa in Fig. 8(a). At the independent SI Table S1
reference state (260 K, 1.00 cm**3/g), it gives -37.2881 kJ/mol and
28.91 MPa versus
-37.219 kJ/mol and about 0.1 MPa. It also fails to recover the SI
Table S3 critical point (197 K, 162 MPa, 1.03 g/cm**3). At the
reported T and density, P is 148.08 MPa and dP/dv is -400.46 MPa
per cm**3/g. An additional quartic fit of Q1 and Q3, analogous to
Handle and Sciortino (2018),
can recover a nearby critical point, but the choice of sampling grid
affects it and the paper does not supply those combined fits.
"""

import numpy as np
from numpy.polynomial import Polynomial

from watereos.pv_eos import (
    deriv_coeffs,
    polyval_rows,
    stable_volume,
    volume_roots,
    volumetric_properties,
)
from watereos.two_state_branch import normalize_branch
from watereos.two_state_general import M_H2O, R

_R = R / 1000  # kJ/(mol K)
_PRESSURE_SCALE = 1 / M_H2O  # kJ/mol per cm^3/g to MPa
_V_LO, _V_HI = 1 / 1.40, 1 / 0.94

# SI Table S2, ascending powers of specific volume in cm^3/g.
_SIGMA2 = (-166.1716, 920.6647, -1548.8626, 965.5023, -156.0925)
_E0 = (453.3317, -2181.7485, 3594.5475, -2695.618, 777.0159)
_A = (-133.7592, 485.0213, -409.8918, -56.7383, 125.4372)
_B = (-3.3921, 11.1494, -10.8102, 1.1937, 1.7119)
_ALPHA = (30.0742, -144.1233, 283.8437, -249.1893, 82.3261)
_C2 = (-1.699e-4, 4.116e-4, -2.538e-4)
_C3 = (2.577e-7, -5.991e-7, 4.732e-7)

# Shift to u = v - 1 before forming products to reduce cancellation.
_v = Polynomial([1, 1])
_sigma2, _e0, _a, _b, _alpha, _c2, _c3 = [
    Polynomial(coeffs)(_v) for coeffs in (_SIGMA2, _E0, _A, _B, _ALPHA, _C2, _C3)
]
_q1 = _e0 - _b * _sigma2
_q3 = _alpha - _a - _b * _e0 + _b**2 * _sigma2 / 2
_PEL_COEFFS = np.zeros((5, len(_q3.coef)))
for _row, _term in zip(_PEL_COEFFS, (_q1, _sigma2, _q3, _c2, _c3)):
    _row[:len(_term.coef)] = _term.coef


def _helmholtz_coefficients(T):
    """Rows of molar F, F_T, and F_TT coefficients in powers of v - 1."""
    one = np.ones_like(T)
    zero = np.zeros_like(T)
    rt = _R * T
    log_beta = -np.log(rt)  # Eq. (7), A0 = 1 kJ/mol
    f = np.column_stack((one, -0.5 / rt, -rt, -T**2, -0.5 * T**3)) @ _PEL_COEFFS
    f_t = np.column_stack((zero, 0.5 / (rt * T), -_R * one, -2 * T,
                           -1.5 * T**2)) @ _PEL_COEFFS
    f_tt = np.column_stack((zero, -1 / (rt * T**2), zero, -2 * one,
                            -3 * T)) @ _PEL_COEFFS
    f[:, 0] += 9 * rt * log_beta
    f_t[:, 0] += 9 * _R * (log_beta - 1)
    f_tt[:, 0] -= 9 * _R / T
    return f, f_t, f_tt


def compute_batch(T_K, P_MPa, branch='stable'):
    """Evaluate flat arrays at liquid roots, returning package units."""
    if normalize_branch(branch) != 'stable':
        raise ValueError("Eltareb (2024) supports only branch='stable'.")

    T = np.asarray(T_K, dtype=float)
    P = np.asarray(P_MPa, dtype=float)
    T = np.where(np.isfinite(T) & (T > 0), T, np.nan)
    f, f_t, f_tt = _helmholtz_coefficients(T)
    p = -_PRESSURE_SCALE * deriv_coeffs(f)
    p_v = deriv_coeffs(p)

    # A tiny extension brackets roots exactly at the fitted endpoints.
    roots, g_rel = volume_roots(p, P, _V_LO - 1 - 1e-12, _V_HI - 1 + 1e-12)
    u, _ = stable_volume(roots, g_rel)
    v = u + 1
    V = v * 1e-3
    Kt, alpha, Kp = volumetric_properties(
        v, polyval_rows(p_v, u),
        -_PRESSURE_SCALE * polyval_rows(deriv_coeffs(f_t), u),
        polyval_rows(deriv_coeffs(p_v), u),
    )
    A = polyval_rows(f, u) * 1000 / M_H2O
    S = -polyval_rows(f_t, u) * 1000 / M_H2O
    G = A + P * V * 1e6
    Cv = -T * polyval_rows(f_tt, u) * 1000 / M_H2O
    Cp = Cv + T * V * Kt * 1e6 * alpha**2
    with np.errstate(divide='ignore', invalid='ignore'):
        Ks = Kt * Cp / Cv
        vel = np.sqrt(Ks * V * 1e6)
    return {
        'rho': 1 / V, 'V': V, 'S': S, 'G': G, 'H': G + T * S,
        'U': A + T * S, 'A': A, 'Cp': Cp, 'Cv': Cv, 'Kt': Kt,
        'Ks': Ks, 'Kp': Kp, 'alpha': alpha, 'vel': vel,
    }


MODELS = {'eltareb2024_qtip4pf_classical': compute_batch}
BRANCH_MODELS = ()
NAN_KEYS = dict.fromkeys(MODELS, ())
