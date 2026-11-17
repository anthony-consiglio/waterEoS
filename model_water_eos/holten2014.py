"""MODEL WATER, NOT REAL WATER. Holten et al. (2014) regular-solution fits
for the five-site ST2 potential. ST2(I) uses reaction-field electrostatics
and ST2(II) uses Ewald electrostatics with vacuum boundary conditions.

For the mean-field model, x is the fraction of low-density structure B.

    G = G_A + x G_BA + RT [x ln x + (1-x) ln(1-x)] + W_molar x(1-x)
    G_A = R Tc sum c_mn t**m p**n
    G_BA = -RT lambda (t + a p)
    W_molar = R Tc (2 + omega_r p)

Here t = (T-Tc)/Tc and p = (P-Pc)/(rho_c R Tc). These are Eqs. (3), (5),
(8)-(9), (29), and (30) with delta = 1, as selected in Sec. III B.
Equation (8) with the positive tabulated a gives a negative dP/dT on
ln K = 0 (Eq. (10) omits the minus sign). Mean-field coefficients and
critical parameters are from Tables I and III (p. 104502-11).
Appendix B sets c01 = 1 + a lambda/2 - omega_r/4 to enforce the critical
volume and c10 = 0 to set the entropy reference. The printed c00 preserves
the simulation energy reference (molecules at infinite separation).

Crossover uses Eqs. (31), (36)-(47) and Tables II and IV (p. 104502-11).
All four tables were checked against the printed page image.
With h1 = ln K, h2 = 2 - W, phi = 2x - 1, and W = W_molar/(RT),
the crossover mixing Gibbs energy uses the following rescaling.

    (G - G_A)/(RT) = -x h1 + W/4 + f
    f = x_cross ln(x_cross) + (1-x_cross) ln(1-x_cross)
        - W_cross phi_cross**2/4 + k/2

Here phi_cross = 2x_cross - 1. The code uses q = ln(kappa**2/Lambda**2)
and y = 1/(1 + exp(-q)), so Y = y**(Delta/(2 nu)). The combined powers
of y for h2_cross, phi_cross, and the kernel are in the constants below.
Delta = 0.5, u_star = 0.472, and Lambda = 0.3 (Eqs. (39)-(43) and
Appendix B). The crossover search follows the larger positive kappa
root of Eq. (46) where it is multivalued, connected to the mean-field
limit at large kappa. Equation (47) is solved on the local-minimum
portion of each composition branch. The metastable continuation is a
prediction of the fit. The pure-state exports use the formal reference
energies G_A and G_A + G_BA.
Differentiating G directly gives the pressure contribution
`chi1 * a * lambda - chi12 * W_P` to the derivative of phi with respect
to reduced pressure, following Eqs. (A1)-(A2) and (A10). The indices and
sign in the printed Eq. (A13) differ from this chain rule.

Use the package H2O molar mass, 0.018015268 kg/mol. Pressure enters the
batch functions in MPa and the Gibbs derivatives use Pa. ST2(I) was fit
over 240-322 K, 100-250 MPa. ST2(II) uses simulation data at 240-315 K
(Figs. 1 and 7-8) with a fit restricted to 100-300 MPa (Sec. III A).
Data coverage is uneven, and the ST2(I) equations yield negative Cv
and Ks and undefined sound speed near the cold, low-pressure corner
(for example 240 K, 100 MPa). The registry probe at ambient pressure
is also an extrapolation.

At the tabulated critical T and P the code reproduces molar densities
52.478 (ST2(I) mean field), 53.261 (ST2(II) mean field), 52.739 (ST2(I)
crossover), and 53.645 (ST2(II) crossover) kmol/m^3. At 240 K Fig. 5
gives ST2(I) coexistence densities near 1083/916 kg/m^3 (mean field)
and 1086/918 kg/m^3 (crossover). Computed HDL/LDL densities are
1083.34/915.61 and 1085.96/917.65 kg/m^3. At Tc +/- 0.1 K on the
ST2(II) critical isobars, Fig. 10 gives Cv near 7.05/5.90 (mean field)
and 7.44/6.00 (crossover), compared with 7.053/5.899 and 7.443/5.995
kJ/(kg K), respectively. Critical response limits are explicit at Tc, Pc.

V. Holten, J. C. Palmer, P. H. Poole, P. G. Debenedetti, and
M. A. Anisimov, J. Chem. Phys. 140, 104502 (2014), doi:10.1063/1.4867287.
"""

from functools import partial

import numpy as np
from scipy.special import expit, xlogy

from watereos.two_state_branch import normalize_branch
from watereos.two_state_general import (
    M_H2O, R, add_fields, compute_two_state, poly_field, properties_from_gibbs,
    times_T,
)

M = M_H2O

KEY_ST2_I = 'holten2014_st2_i_mf'
KEY_ST2_II = 'holten2014_st2_ii_mf'
KEY_ST2_I_CROSSOVER = 'holten2014_st2_i_crossover'
KEY_ST2_II_CROSSOVER = 'holten2014_st2_ii_crossover'

FIT_RANGES = {
    KEY_ST2_I: ((240.0, 322.0), (100.0, 250.0)),  # K, MPa
    KEY_ST2_II: ((240.0, 315.0), (100.0, 300.0)),
    KEY_ST2_I_CROSSOVER: ((240.0, 322.0), (100.0, 250.0)),
    KEY_ST2_II_CROSSOVER: ((240.0, 315.0), (100.0, 300.0)),
}

# Tables I and III. Tc in K, Pc in MPa, rho_c in kmol/m^3.
_ST2_I = (
    253.5, 160.0, 52.478, -3.4915, 0.085811, 0.23165,
    {
        (0, 0): -16.646,
        (0, 2): -7.2751e-3,
        (0, 3): -2.3302e-3,
        (1, 1): 1.9503e-1,
        (1, 2): -2.7651e-2,
        (1, 3): 1.2020e-2,
        (2, 0): -5.8303,
        (3, 0): 2.1150,
    },
)
_ST2_II = (
    249.0, 146.0, 53.261, -3.2021, 0.1199, 0.17181,
    {
        (0, 0): -15.509,
        (0, 2): -8.5241e-3,
        (0, 3): -1.3232e-3,
        (1, 1): 2.4052e-1,
        (1, 2): -1.1999e-2,
        (1, 3): 3.4151e-3,
        (2, 0): -4.4007,
        (3, 0): 1.7685,
    },
)

# Tables II and IV, with the same units as the mean-field fits.
_ST2_I_CROSSOVER = (
    252.0, 168.0, 52.739, -3.4643, 0.085496, 0.23802,
    {
        (0, 0): -16.777,
        (0, 2): -8.3506e-3,
        (0, 3): -2.0066e-3,
        (1, 1): 1.8550e-1,
        (1, 2): -2.1457e-2,
        (1, 3): 1.0782e-2,
        (2, 0): -5.8566,
        (3, 0): 2.0925,
    },
)
_ST2_II_CROSSOVER = (
    246.75, 155.0, 53.645, -3.1793, 0.12053, 0.17181,
    {
        (0, 0): -15.685,
        (0, 2): -9.4717e-3,
        (0, 3): -1.0999e-3,
        (1, 1): 2.3400e-1,
        (1, 2): -8.6853e-3,
        (1, 3): 2.8369e-3,
        (2, 0): -4.4088,
        (3, 0): 1.7146,
    },
)

# Eqs. (36)-(46) and Appendix B.
_NU, _GAMMA, _ALPHA = 0.63, 1.24, 0.11
_CUTOFF = 0.3
_CT = np.sqrt(0.472 * _CUTOFF) / 2.0
_H_POWER = (1.5 * _NU - 1.0) / (2.0 * _NU)
_PHI_POWER = (_GAMMA / 2.0 - 0.75 * _NU) / (2.0 * _NU)
_K_POWER = -_ALPHA / (2.0 * _NU)
_K_AMPLITUDE = -_CT**2 * _NU / (4.0 * _ALPHA * _CUTOFF)


def _reduced_coordinates(T, P_Pa, Tc, Pc, rho_c):
    pressure_scale = 1000.0 * rho_c * R * Tc
    t = (T - Tc) / Tc
    p = (P_Pa - Pc * 1e6) / pressure_scale
    return t, p, pressure_scale


def _on_symmetry_line(t, ap):
    tolerance = 4.0 * np.finfo(float).eps * (np.abs(t) + np.abs(ap))
    return np.abs(t + ap) <= tolerance


def _terms(T, P_Pa, *, parameters):
    Tc, Pc, rho_c, lam, a, omega_r, background = parameters
    t, p, pressure_scale = _reduced_coordinates(T, P_Pa, Tc, Pc, rho_c)
    dt, dp = 1.0 / Tc, 1.0 / pressure_scale

    coefficients = {(0, 1): 1.0 + a * lam / 2.0 - omega_r / 4.0, **background}
    GA = poly_field(coefficients, t, p, dt, dp, R * Tc)
    dG = times_T(poly_field({(1, 0): 1.0, (0, 1): a}, t, p, dt, dp, -R * lam), T)
    W = poly_field({(0, 0): 2.0, (0, 1): omega_r}, t, p, dt, dp, R * Tc)
    return GA, dG, W


def _compute_mean_field(T_K, P_MPa, branch='stable', *, parameters):
    out = compute_two_state(
        T_K, P_MPa, partial(_terms, parameters=parameters), M=M, branch=branch
    )
    Tc, Pc, rho_c, _, a, _, _ = parameters
    T, P_MPa = np.broadcast_arrays(
        np.atleast_1d(np.asarray(T_K, dtype=float)),
        np.atleast_1d(np.asarray(P_MPa, dtype=float)),
    )
    if 'metastable' in out:
        t, p, _ = _reduced_coordinates(T, P_MPa * 1e6, Tc, Pc, rho_c)
        out['metastable'] &= ~_on_symmetry_line(t, a * p)
    critical = (T == Tc) & (P_MPa == Pc)
    if np.any(critical):
        # The flat minimum at x=1/2 is ill-conditioned in the generic
        # root solver. Cancel the common singularity in Cv and Ks first.
        T = np.full(np.count_nonzero(critical), Tc)
        P = np.full_like(T, Pc * 1e6)
        GA, dG, W = _terms(T, P, parameters=parameters)
        fixed = tuple(a + b / 2.0 + w / 4.0 for a, b, w in zip(GA, dG, W))
        G, GT, GP, GTT, GTP, GPP = fixed
        limit = properties_from_gibbs(G - R * T * np.log(2), GT - R * np.log(2),
                                      GP, GTT, GTP, GPP, T, P, M)
        slope = dG[1] / dG[2]
        cv = -T * (GTT - 2.0 * slope * GTP + slope**2 * GPP) / M
        ks_pa = T / (M * cv) * slope**2 * GP
        limit.update(x=0.5, Kt=0.0, alpha=-np.inf, Cp=np.inf, Cv=cv,
                     Ks=ks_pa / 1e6, vel=np.sqrt(ks_pa / limit['rho']))
        for key, value in limit.items():
            out[key][critical] = value
    return out


def _renormalized(q, h):
    """Rescale h2 and solve Eq. (46) explicitly for phi_cross**2."""
    y = expit(q)
    log_y = -np.logaddexp(0.0, -q)
    r = (_CUTOFF**2 / _CT) * np.exp(q - 0.25 * log_y)
    A = np.exp(_H_POWER * log_y)
    B = np.exp(_PHI_POWER * log_y)
    b = r - h * A
    return y, log_y, r, A, B, b


def _atanh_minus_z(z):
    # The leading cubic term remains resolvable as the critical point is approached.
    small = z**3 * (1.0/3.0 + z**2 * (1.0/5.0 + z**2 * (1.0/7.0 + z**2/9.0)))
    return np.where(np.abs(z) < 0.02, small, np.arctanh(z) - z)


def _fold(q, h):
    """Sign of d(phi**2)/dq, with phi = phi_cross / B."""
    y, _, r, A, _, b = _renormalized(q, h)
    dlog = 1.0 - y
    b_q = r * (1.0 - 0.25 * dlog) - h * A * _H_POWER * dlog
    return b_q - _PHI_POWER * dlog * b * (b + 2.0)


def _ordering_field(q, h):
    """Required h1 on the positive-phi branch, from Eq. (47)."""
    y, log_y, r, A, B, b = _renormalized(q, h)
    dlog = 1.0 - y
    b = np.maximum(b, 0.0)
    z_sq = b / (2.0 + b)
    z = np.sqrt(z_sq)
    A_q = A * _H_POWER * dlog
    z_zq = (r * (1.0 - 0.25 * dlog) - h * A_q) / (2.0 + b)**2
    phi_q_z = (z_zq - _PHI_POWER * dlog * z_sq) / B
    atanh_ratio = np.divide(
        _atanh_minus_z(z), z, out=np.zeros_like(z), where=z > 0,
    )
    kernel_q = _K_AMPLITUDE * h**2 * np.exp(_K_POWER * log_y) * _K_POWER * dlog
    S_q = z_zq * (atanh_ratio + h * A / 2.0) + h * A_q * z_sq / 4.0 + kernel_q
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2.0 * S_q * z / phi_q_z


def _bisect_increasing(function, lower, upper):
    """Vectorized zero search for bracketed increasing functions."""
    for _ in range(54):
        middle = (lower + upper) / 2.0
        above = function(middle) > 0.0
        lower, upper = np.where(above, lower, middle), np.where(above, middle, upper)
    return (lower + upper) / 2.0


def _equilibrium_q(h1, h, branch):
    """Find the local minimum on the branch connected to large kappa."""
    if branch == 'stable':
        sign = np.where(h1 > 0.0, 1.0, -1.0)
    else:
        sign = -1.0 if branch == 'hdl' else 1.0
    target = sign * h1
    lo, hi = np.full_like(h, -110.0), np.full_like(h, 24.0)
    positive, negative = h > 0.0, h < 0.0
    if np.any(positive):
        lo[positive] = _bisect_increasing(
            lambda q: _renormalized(q, h[positive])[-1], lo[positive], hi[positive],
        )
    minimum_field = np.zeros_like(h)
    if np.any(negative):
        hn = h[negative]
        fold = _bisect_increasing(
            lambda q: _fold(q, hn), lo[negative], hi[negative],
        )
        # The increasing branch starts at its spinodal, the minimum of
        # h1(q). Find it between the fold of Eq. (46) and the large-q limit.
        left, right = fold, hi[negative]
        for _ in range(48):
            left_third = (2.0 * left + right) / 3.0
            right_third = (left + 2.0 * right) / 3.0
            first_lower = _ordering_field(left_third, hn) < _ordering_field(right_third, hn)
            left = np.where(first_lower, left, left_third)
            right = np.where(first_lower, right_third, right)
        lo[negative] = (left + right) / 2.0
        minimum_field[negative] = _ordering_field(lo[negative], hn)

    q = _bisect_increasing(lambda q: _ordering_field(q, h) - target, lo, hi)
    q = np.where(positive & (target == 0.0), lo, q)
    valid = (target >= minimum_field) & (target <= _ordering_field(hi, h))
    return q, sign, valid, valid & (target < 0.0)


def _mixing_derivatives(h1, h, q, phi, z):
    """Reduced Gibbs energy of mixing and derivatives in (h1, h2)."""
    y, log_y, r, A, B, _ = _renormalized(q, h)
    dlog, d2log = 1.0 - y, -y * (1.0 - y)
    A_q = A * _H_POWER * dlog
    A_qq = A * ((_H_POWER * dlog)**2 + _H_POWER * d2log)
    k_factor = np.exp(_K_POWER * log_y)
    J = np.expm1(_K_POWER * log_y)
    J_q = k_factor * _K_POWER * dlog
    J_qq = k_factor * ((_K_POWER * dlog)**2 + _K_POWER * d2log)
    z_q = z * _PHI_POWER * dlog
    z_qq = z * ((_PHI_POWER * dlog)**2 + _PHI_POWER * d2log)
    z_qphi = B * _PHI_POWER * dlog
    x, x_cross = (1.0 + phi) / 2.0, (1.0 + z) / 2.0

    S = (xlogy(x_cross, x_cross) + xlogy(1.0 - x_cross, 1.0 - x_cross)
         - (2.0 - h * A) * z**2 / 4.0 + _K_AMPLITUDE * h**2 * J)
    S_z = _atanh_minus_z(z) + h * A * z / 2.0
    S_zz = z**2 / (1.0 - z**2) + h * A / 2.0
    S_q = S_z * z_q + h * A_q * z**2 / 4.0 + _K_AMPLITUDE * h**2 * J_q
    S_h = A * z**2 / 4.0 + 2.0 * _K_AMPLITUDE * h * J
    S_qq = S_zz * z_q**2 + S_z * z_qq + h * (A_q * z * z_q + A_qq * z**2 / 4.0)
    S_qq += _K_AMPLITUDE * h**2 * J_qq
    S_qphi = S_zz * z_q * B + S_z * z_qphi + h * A_q * z * B / 2.0
    S_qh = A * z * z_q / 2.0 + A_q * z**2 / 4.0 + 2.0 * _K_AMPLITUDE * h * J_q

    # Implicit derivatives of Q(q, phi, h) = 0 (Eq. 46).
    E_z = 4.0 * z / (1.0 - z**2)**2
    E_zz = 4.0 * (1.0 + 3.0 * z**2) / (1.0 - z**2)**3
    r_log_q = 1.0 - dlog / 4.0
    Q_q = r * r_log_q - h * A_q - E_z * z_q
    Q_qq = r * (r_log_q**2 - d2log / 4.0) - h * A_qq - E_z * z_qq - E_zz * z_q**2
    Q_qphi = -E_zz * z_q * B - E_z * z_qphi
    q_phi, q_h = E_z * B / Q_q, A / Q_q
    multiplier = -S_q / Q_q
    qq = S_qq + multiplier * Q_qq
    qphi = S_qphi + multiplier * Q_qphi
    qh = S_qh - multiplier * A_q
    S_phiphi = (S_zz - multiplier * E_zz) * B**2 + 2.0 * qphi * q_phi + qq * q_phi**2
    S_phih = A * z * B / 2.0 + qphi * q_h + qh * q_phi + qq * q_phi * q_h
    S_hh = 2.0 * _K_AMPLITUDE * J + 2.0 * qh * q_h + qq * q_h**2

    # Eliminate the equilibrium composition from the Hessian by a Schur complement.
    with np.errstate(divide='ignore', invalid='ignore'):
        return (S - x * h1 + (2.0 - h) / 4.0, -x, S_h + S_q * q_h - 0.25,
                -0.25 / S_phiphi, 0.5 * S_phih / S_phiphi,
                S_hh - S_phih**2 / S_phiphi)


def _compute_crossover(T_K, P_MPa, branch='stable', *, parameters):
    branch = normalize_branch(branch)
    T, P = np.broadcast_arrays(
        np.atleast_1d(np.asarray(T_K, dtype=float)),
        np.atleast_1d(np.asarray(P_MPa, dtype=float)) * 1e6,
    )
    Tc, Pc, rho_c, lam, a, omega_r, _ = parameters
    GA, dG, W = _terms(T, P, parameters=parameters)
    t, p, _ = _reduced_coordinates(T, P, Tc, Pc, rho_c)
    h1 = lam * (t + a * p)
    # Recognize the symmetry line when the two contributions cancel to roundoff.
    h1 = np.where(_on_symmetry_line(t, a * p), 0.0, h1)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = (2.0 * t - omega_r * p) / (T / Tc)
    critical = (T == Tc) & (P == Pc * 1e6)
    usable = (T > 0.0) & np.isfinite(h1) & np.isfinite(h)
    # Substitute regular inputs for points handled by the critical limit or NaN mask.
    regular = usable & ~critical
    h1_safe, h_safe = np.where(regular, h1, 0.1), np.where(regular, h, 0.1)
    q, sign, valid, metastable = _equilibrium_q(h1_safe, h_safe, branch)
    _, _, _, _, B, b = _renormalized(q, h_safe)
    b = np.where((h1_safe == 0.0) & (h_safe > 0.0), 0.0, np.maximum(b, 0.0))
    z = sign * np.sqrt(b / (2.0 + b))
    phi = z / B
    g, g1, g2, g11, g12, g22 = _mixing_derivatives(h1_safe, h_safe, q, phi, z)
    g = np.where(critical, 0.5 - np.log(2.0), g)
    g1, g2 = np.where(critical, -0.5, g1), np.where(critical, -0.25, g2)
    g11, g12, g22 = (np.where(critical, 0.0, d) for d in (g11, g12, g22))

    with np.errstate(divide='ignore', invalid='ignore'):
        h1_T, h1_P = lam / Tc, -dG[2] / (R * T)
        h_T, h_P = W[0] / (R * T**2), -W[2] / (R * T)
        g_T, g_P = g1 * h1_T + g2 * h_T, g1 * h1_P + g2 * h_P
        g_TT = g11 * h1_T**2 + 2.0 * g12 * h1_T * h_T + g22 * h_T**2 - 2.0 * g2 * h_T / T
        g_TP = g11 * h1_T * h1_P + g12 * (h1_T * h_P + h1_P * h_T)
        g_TP += g22 * h_T * h_P - g2 * h_P / T
        g_PP = g11 * h1_P**2 + 2.0 * g12 * h1_P * h_P + g22 * h_P**2
        mix = tuple(R * d for d in times_T((g, g_T, g_P, g_TT, g_TP, g_PP), T))

    out = properties_from_gibbs(*add_fields(GA, mix), T, P, M)
    for key, value in {'Cp': np.inf, 'Cv': np.inf, 'alpha': -np.inf,
                       'Kt': 0.0, 'Ks': 0.0, 'vel': 0.0}.items():
        out[key][critical] = value
    out['x'] = -g1
    exists = critical | (usable & valid & (np.abs(phi) <= 1.0))
    for key in out:
        out[key] = np.where(exists, out[key], np.nan)
    for suffix, field in (('_A', GA), ('_B', add_fields(GA, dG))):
        for key, value in properties_from_gibbs(*field, T, P, M).items():
            out[key + suffix] = value
    if branch != 'stable':
        out['metastable'] = regular & exists & metastable
    return out


MODELS = {
    KEY_ST2_I: partial(_compute_mean_field, parameters=_ST2_I),
    KEY_ST2_II: partial(_compute_mean_field, parameters=_ST2_II),
    KEY_ST2_I_CROSSOVER: partial(_compute_crossover, parameters=_ST2_I_CROSSOVER),
    KEY_ST2_II_CROSSOVER: partial(_compute_crossover, parameters=_ST2_II_CROSSOVER),
}
BRANCH_MODELS = tuple(MODELS)
