"""MODEL WATER, NOT REAL WATER. Holten et al. (2013) fits for the monatomic,
coarse-grained mW potential. The fitted simulation isobars span 0.1-1013 MPa
and roughly 170-280 K across pressures (Figs. 1 and 7). At 0.1 MPa the
equilibrium data only extend down to about 200 K. Lower-temperature
atmospheric-pressure structure, enthalpy, entropy, and heat capacity in
Figs. 2 and 4-5 include nonequilibrium 10 K/ns quenches.

The athermal fits follow Eqs. (2)-(7), (13), (16), and (A11).

    G = G_A + x G_BA + (RT/N) [x ln x + (1-x) ln(1-x)] + RT omega x(1-x)
    G_A = R T0 sum c_mn t**m p**n
    G_BA = RT lambda (t + a p)
    omega = omega1 - (omega1-omega0) ((p-p1)/p1)**2

with x the low-density fraction, t = (T-T0)/T0, and p = P V0/(R T0).
Equation (3) uses positive G_BA/(RT), which agrees with Eq. (14) and
the decrease of x with temperature in Fig. 2. Equation (5) requires
molar density in its pressure scale even though Tables I and II report
rho0 in kg/m^3. Use V0 = M/rho0, with the package H2O molar mass
M = 0.018015268 kg/mol. The background and interaction constants are
from Tables I (N=1) and II (N=6), both on p. 174501-9. The upper bounds
on N*omega are 1.6777 and 1.19136, respectively, so each composition
has a single equilibrium minimum with no liquid-liquid critical point.
The named branches select the corresponding half of the composition
range using the package convention. The constant-omega and regular-
solution trials, the ideal-solution trial, and the N=6.5 optimum lack
full printed coefficients. The N=1 fit has a systematic compressibility
misfit at 0.1 MPa reported in Sec. II D. Its extrapolation to 170 K at
0.1 MPa gives negative Cv and Ks and undefined sound speed.

The empirical weak-crystallization fit (Eqs. (20)-(25), Table III,
p. 174501-9) uses G/(R Ts0) = sum c_mn (T/Ts0)**m p**n + beta sqrt(d),
where p = P V0/(R Ts0), Ts = Ts0 (1-a_prime p), and d = T/Ts - 1.
Equation (24) uses the unshifted reduced temperature. The batch function
returns NaN where Ts <= 0 or T <= Ts. The authors emphasize that the
fitted coupling is too large for the weak-coupling theory.

The published backgrounds omit c00 and c10, which are set to zero.
The resulting energies and entropies have a polynomial reference.
Equations (18)-(19) are separate fits to liquid-minus-ice data.
Coefficients were checked against images of Tables I-III.

At 220 K and 0.1 MPa, Fig. 2 shows x near 0.28 (N=1)
and 0.18 (N=6), versus 0.28000 and 0.18032 here. At 240 K, Fig. 1
gives N=1 densities of about 1003 and 1120 kg/m^3 at 0.1 and 1013 MPa,
versus 1003.11 and 1119.75. At 220 K and 0.1 MPa, Fig. 5 gives
clustered Cp near 3.2 kJ/(kg K), versus 3.222, and Fig. 7 gives
weak-crystallization Cp near 3.2 kJ/(kg K) and density near 998 kg/m^3,
versus 3.201 and 998.33. The Table III parameters give a stability
line of 175.28 K - 0.09279 K/MPa * P, consistent with rounded Eq. (26).

V. Holten, D. T. Limmer, V. Molinero, and M. A. Anisimov,
J. Chem. Phys. 138, 174501 (2013), doi:10.1063/1.4802992.
"""

from functools import partial

import numpy as np

from watereos.two_state_branch import normalize_branch
from watereos.two_state_general import (
    M_H2O, R, add_fields, compute_two_state, poly_field, properties_from_gibbs,
    scale_field, times_T,
)

M = M_H2O
RHO0 = 1000.0  # kg/m^3, Tables I-III
T0 = 203.07    # K, Tables I-II
A = 0.039968   # Tables I-II
V0 = M / RHO0  # m^3/mol
KEY = 'holten2013_mw'
KEY_CLUSTERED = 'holten2013_mw_n6'
KEY_WEAK_CRYSTALLIZATION = 'holten2013_mw_weak_crystallization'
T_MIN, T_MAX = 170.0, 280.0    # K, plotted simulation data span
P_MIN, P_MAX = 0.1, 1013.0     # MPa

# Table I and Table II, respectively. lambda, omega0, omega1, p1, c_mn.
_ATHERMAL = (
    2.6917, 1.6362, 1.6777, 2.3219,
    {
        (0, 1): 9.5440e-1,
        (0, 2): -5.0517e-3,
        (0, 3): 1.5125e-4,
        (0, 5): -1.5795e-7,
        (1, 1): 8.4408e-2,
        (1, 2): -3.9163e-3,
        (1, 3): 1.2385e-4,
        (2, 0): -1.7092,
        (2, 1): -4.3879e-2,
        (3, 0): 4.0687e-1,
        (3, 1): 6.1937e-2,
    },
)
_CLUSTERED = (
    1.529, 0.19523, 0.19856, 1.1637,
    {
        (0, 1): 9.8530e-1,
        (0, 2): -8.1978e-3,
        (0, 3): 2.8467e-4,
        (0, 5): -3.1836e-7,
        (1, 1): 3.9710e-2,
        (1, 2): -6.4543e-4,
        (1, 3): 4.6055e-5,
        (2, 0): -1.8630,
        (2, 1): -3.2135e-2,
        (3, 0): 3.3420e-1,
        (3, 1): 5.7351e-2,
    },
)

# Table III, weak-crystallization fit.
TS0 = 175.28  # K
BETA = 2.3254
A_PRIME = 0.042825
_WEAK_CRYSTALLIZATION = {
    (0, 1): 6.1915e-1,
    (0, 2): -8.2724e-3,
    (0, 3): 3.6026e-4,
    (0, 5): -6.6486e-7,
    (1, 1): 4.2389e-1,
    (1, 2): -3.3672e-4,
    (1, 3): -1.3649e-4,
    (2, 0): 3.4178e-1,
    (2, 1): -2.3514e-1,
    (3, 0): -2.2560e-1,
    (3, 1): 4.9163e-2,
}


def _athermal_terms(T, P_Pa, *, parameters):
    lam, omega0, omega1, p1, coefficients = parameters
    dt, dp = 1.0 / T0, V0 / (R * T0)
    t, p = (T - T0) * dt, P_Pa * dp

    GA = poly_field(coefficients, t, p, dt, dp, R * T0)
    dG = times_T(poly_field({(1, 0): 1.0, (0, 1): A}, t, p, dt, dp, R * lam), T)
    interaction = {(0, 0): omega1, (0, 2): -(omega1 - omega0) / p1**2}
    W = times_T(poly_field(interaction, t, p - p1, dt, dp, R), T)
    return GA, dG, W


def _compute_athermal(T_K, P_MPa, branch='stable', *, parameters, N):
    return compute_two_state(
        T_K, P_MPa, partial(_athermal_terms, parameters=parameters),
        M=M, N=N, branch=branch,
    )


def _compute_weak_crystallization(T_K, P_MPa, branch='stable'):
    if normalize_branch(branch) != 'stable':
        raise ValueError('The weak-crystallization fit has a single liquid branch')

    T, P = np.broadcast_arrays(
        np.atleast_1d(np.asarray(T_K, dtype=float)),
        np.atleast_1d(np.asarray(P_MPa, dtype=float)) * 1e6,
    )
    dt, dp = 1.0 / TS0, V0 / (R * TS0)
    regular = poly_field(_WEAK_CRYSTALLIZATION, T * dt, P * dp, dt, dp, R * TS0)

    # Differentiate d = T/Ts - 1 and then sqrt(d), retaining all chain-rule terms.
    Ts = TS0 * (1.0 - A_PRIME * P * dp)
    Ts_P = -TS0 * A_PRIME * dp
    Ts = np.where((Ts > 0) & (T > Ts), Ts, np.nan)
    root = np.sqrt((T - Ts) / Ts)
    d_T = 1.0 / Ts
    d_P = -T * Ts_P / Ts**2
    d_TP = -Ts_P / Ts**2
    d_PP = 2.0 * T * Ts_P**2 / Ts**3
    first = 0.5 / root
    second = -0.25 / root**3
    singular = scale_field(
        (root, first * d_T, first * d_P, second * d_T**2,
         first * d_TP + second * d_T * d_P, first * d_PP + second * d_P**2),
        R * TS0 * BETA,
    )
    return properties_from_gibbs(*add_fields(regular, singular), T, P, M)


MODELS = {
    KEY: partial(_compute_athermal, parameters=_ATHERMAL, N=1),
    KEY_CLUSTERED: partial(_compute_athermal, parameters=_CLUSTERED, N=6),
    KEY_WEAK_CRYSTALLIZATION: _compute_weak_crystallization,
}
BRANCH_MODELS = (KEY, KEY_CLUSTERED)
NAN_KEYS = {KEY_WEAK_CRYSTALLIZATION: ('Kp',)}
