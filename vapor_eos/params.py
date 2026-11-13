"""
Constants of the water-vapor equation of state.

The vapor is the low-density limit of the IAPWS-95 formulation (Wagner &
Pruss 2002; IAPWS R6-95(2018)): its ideal-gas part, plus the second and
third virial coefficients that follow from its residual part. Nothing is
fitted here; every number below is an IAPWS-95 coefficient or one of the
two IAPWS vapor-pressure correlations.
"""

# ── IAPWS-95 reducing parameters ─────────────────────────────────────────
T_C = 647.096            # K, critical temperature
RHO_C = 322.0            # kg/m^3, critical density
R = 461.51805            # J/(kg K), specific gas constant of IAPWS-95
P_STAR = RHO_C * R * T_C  # Pa, pressure scale rho_c R T_c (about 96.16 MPa)

# ── Ideal-gas part (IAPWS-95, Table 1; n1 and n2 as revised in 2016) ─────
#   phi0 = ln(delta) + n1 + n2 tau + n3 ln(tau)
#          + sum_i n_i ln(1 - exp(-gamma_i tau)),   i = 4..8
# n1 and n2 carry the reference state: zero internal energy and entropy
# of the saturated liquid at the triple point, the reference state of the
# liquids and ices in this package.
N0_1 = -8.3204464837497
N0_2 = 6.6832105275932
N0_3 = 3.00632
N0_EXP = (0.012436, 0.97315, 1.27950, 0.96956, 0.24873)
GAMMA0 = (1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105)

# ── Virial coefficients from the IAPWS-95 residual part ──────────────────
# Expanding the residual part at small reduced density,
#   phir = b1(tau) delta + b2(tau) delta^2 + O(delta^3),
# gives the second and third virial coefficients B = b1 / rho_c and
# C = 2 b2 / rho_c^2. Each entry is (coefficient, exponent of tau).
#
# b1 collects the terms with d = 1: polynomial terms 1-3 and the
# exponential terms 8-10 (c = 1) and 23 (c = 2).
B1_TERMS = (
    (0.12533547935523e-1, -0.5),     # n1
    (0.78957634722828e1, 0.875),     # n2
    (-0.87803203303561e1, 1.0),      # n3
    (-0.66856572307965, 4.0),        # n8
    (0.20433810950965, 6.0),         # n9
    (-0.66212605039687e-4, 12.0),    # n10
    (-0.10793600908932, 7.0),        # n23
)
# b2 collects the terms with d = 2 (4, 5, 11, 12, 24, 25, 26) and the
# second-order part of the d = 1, c = 1 terms: delta exp(-delta) =
# delta - delta^2 + ..., so terms 8-10 enter with the opposite sign.
B2_TERMS = (
    (0.31802509345418, 0.5),         # n4
    (-0.26145533859358, 0.75),       # n5
    (0.66856572307965, 4.0),         # -n8
    (-0.20433810950965, 6.0),        # -n9
    (0.66212605039687e-4, 12.0),     # -n10
    (-0.19232721156002, 1.0),        # n11
    (-0.25709043003438, 5.0),        # n12
    (0.17611491008752e-1, 1.0),      # n24
    (0.22132295167546, 9.0),         # n25
    (-0.40247669763528, 10.0),       # n26
)
# The Gaussian terms 52-54 start at delta^3. The two non-analytic terms
# 55-56 are linear in delta but carry exp(-28) or smaller at delta = 0;
# their share of b1 is below 1e-13 and is left out.

# ── Vapor-pressure correlations ──────────────────────────────────────────
T_TRIPLE = 273.16        # K
P_TRIPLE = 611.657e-6    # MPa
P_C = 22.064             # MPa

# Sublimation pressure of ice Ih, IAPWS R14-08(2011):
#   ln(p / p_t) = (1/theta) sum_i a_i theta^b_i,   theta = T / T_t
SUBLIMATION_A = (-0.212144006e2, 0.273203819e2, -0.610598130e1)
SUBLIMATION_B = (0.333333333e-2, 0.120666667e1, 0.170333333e1)

# Saturation pressure of liquid water, Wagner & Pruss (1993), IAPWS
# SR1-86(1992):
#   ln(p / p_c) = (T_c / T) sum_i a_i (1 - T/T_c)^e_i
SATURATION_A = (-7.85951783, 1.84408259, -11.7866497, 22.6807411,
                -15.9618719, 1.80122502)
SATURATION_E = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)

# ── Suggested range ──────────────────────────────────────────────────────
# The IAPWS-95 ideal-gas part holds down to 130 K. Upward, the truncation
# at the third virial coefficient limits the accuracy near saturation: at
# 400 K (saturation pressure 0.246 MPa) the volume is within 2e-4 and the
# heat capacity within 1.3 % of the full IAPWS-95 formulation, and the
# deviations grow quickly beyond.
T_MIN = 130.0            # K
T_MAX = 400.0            # K
P_MIN = 0.0              # MPa (exclusive: p must be positive)
P_MAX = 0.25             # MPa
