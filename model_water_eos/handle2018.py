"""MODEL WATER, NOT REAL WATER. Rigid classical TIP4P/2005 liquid.

Handle and Sciortino, J. Chem. Phys. 148, 134505 (2018),
doi:10.1063/1.5023894. In the Gaussian landscape approximation, the
molar Helmholtz energy from Eqs. (5), (12), (13), and (16) is

    F = Q1 - sigma2/(2*R*T) - R*T*Q3 - c2*T**2 - c3*T**3/2
        + (6 - 3/N)*R*T*ln(A0/(R*T)).

Q1 = E0 - b*sigma2 and Q3 = alpha*N - a - b*E0 + b**2*sigma2/2,
with the extensive quantities expressed per mole of water molecules.
Table II (Appendix B, p. 134505-11) tabulates Q1 (kJ/mol), sigma2
((kJ/mol)**2), Q3 (dimensionless), and ci (kJ/(mol K**i)) as
polynomials in v (cm**3/mol). Eq. (27), Fig. 10(c), and Table II use
-b*E0 despite the plus sign in the Sec. V C prose. Eq. (5) supplies
A0 = 1 kJ/mol and Sec. IV A supplies N = 1000. The H2O molar mass is
18.015268 g/mol.
The explicit thermal term retains the paper's energy and entropy reference
without shifting to a real-water reference. P = -dF/dv, with kJ/(cm**3)
converted to MPa, reproduces Eqs. (24)-(28).

The fit covers 200-270 K and 0.90-1.42 g/cm**3. Over that window the
outer density edges span approximately -142 to 2411 MPa in total
(about -28 to 1910 MPa at 200 K and -141 to 2411 MPa at 270 K).
Inversion selects the lowest Gibbs energy among mechanically stable
roots in the density window. A state with no such root gives NaNs.
The liquid fit can be evaluated at positive temperatures outside the
fit window, including the registry probe at 273.15 K. The Gaussian
liquid model is extrapolative below 200 K and ends physically at the
Kauzmann line. The ideal-glass and vapor branches are excluded.

The extrapolated liquid-liquid critical point (Fig. 12 and Sec. V C)
is 175.469 K, 175.099 MPa, 0.997035 g/cm**3, within the reported
175 +/- 2 K, 175 +/- 2 MPa, and 0.9970 +/- 0.001 g/cm**3.
At 250 K and 1.10 g/cm**3 the calculated F is -55.9765 kJ/mol,
versus the independently integrated -55.95 in Table I. A 0.05 kJ/mol
tolerance tests agreement with the fitted surface across that table.
"""

import numpy as np

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
_M_G = M_H2O * 1000  # g/mol
_MODES = 6 - 3 / 1000  # Eq. (5), per molecule
_V_LO, _V_HI = _M_G / 1.42, _M_G / 0.90

# Table II, ascending powers of molar volume in cm^3/mol.
_PEL_COEFFS = np.array([
    [4.69736e2, -1.19385e2, 1.02918e1, -4.00425e-1, 5.98043e-3],  # Q1
    [5.50606e2, -1.16995e2, 9.61032e0, -3.63301e-1, 5.46496e-3],  # sigma2
    [5.08848e1, -1.53888e1, 1.55381e0, -6.90207e-2, 1.15370e-3],  # Q3
    [-1.46726e-4, 1.92760e-5, -6.59881e-7, 0, 0],  # c2
    [1.53829e-7, -1.84834e-8, 9.56831e-10, 0, 0],  # c3
])


def _helmholtz_coefficients(T):
    """Rows of molar F, F_T, and F_TT coefficients in kJ and K units."""
    one = np.ones_like(T)
    zero = np.zeros_like(T)
    rt = _R * T
    log_beta = -np.log(rt)  # A0 = 1 kJ/mol
    f = np.column_stack((one, -0.5 / rt, -rt, -T**2, -0.5 * T**3)) @ _PEL_COEFFS
    f_t = np.column_stack((zero, 0.5 / (rt * T), -_R * one, -2 * T,
                           -1.5 * T**2)) @ _PEL_COEFFS
    f_tt = np.column_stack((zero, -1 / (rt * T**2), zero, -2 * one,
                            -3 * T)) @ _PEL_COEFFS
    f[:, 0] += _MODES * rt * log_beta
    f_t[:, 0] += _MODES * _R * (log_beta - 1)
    f_tt[:, 0] -= _MODES * _R / T
    return f, f_t, f_tt


def compute_batch(T_K, P_MPa, branch='stable'):
    """Evaluate flat arrays at liquid roots, returning package units."""
    if normalize_branch(branch) != 'stable':
        raise ValueError("Handle (2018) supports only branch='stable'.")

    T = np.asarray(T_K, dtype=float)
    P = np.asarray(P_MPa, dtype=float)
    T = np.where(np.isfinite(T) & (T > 0), T, np.nan)
    f, f_t, f_tt = _helmholtz_coefficients(T)
    p = -1000 * deriv_coeffs(f)
    p_v = deriv_coeffs(p)

    # A tiny extension brackets roots exactly at the fitted endpoints.
    roots, g_rel = volume_roots(p, P, _V_LO - 1e-10, _V_HI + 1e-10)
    v, _ = stable_volume(roots, g_rel)
    V = v * 1e-6 / M_H2O
    Kt, alpha, Kp = volumetric_properties(
        v, polyval_rows(p_v, v),
        -1000 * polyval_rows(deriv_coeffs(f_t), v),
        polyval_rows(deriv_coeffs(p_v), v),
    )
    A = polyval_rows(f, v) * 1000 / M_H2O
    S = -polyval_rows(f_t, v) * 1000 / M_H2O
    G = A + P * V * 1e6
    Cv = -T * polyval_rows(f_tt, v) * 1000 / M_H2O
    Cp = Cv + T * V * Kt * 1e6 * alpha**2
    with np.errstate(divide='ignore', invalid='ignore'):
        Ks = Kt * Cp / Cv
        vel = np.sqrt(Ks * V * 1e6)
    return {
        'rho': 1 / V, 'V': V, 'S': S, 'G': G, 'H': G + T * S,
        'U': A + T * S, 'A': A, 'Cp': Cp, 'Cv': Cv, 'Kt': Kt,
        'Ks': Ks, 'Kp': Kp, 'alpha': alpha, 'vel': vel,
    }


MODELS = {'handle2018_tip4p2005': compute_batch}
BRANCH_MODELS = ()
NAN_KEYS = dict.fromkeys(MODELS, ())
