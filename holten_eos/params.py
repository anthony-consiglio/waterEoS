"""
Holten, Sengers & Anisimov (2014) two-state EoS parameters.

Reference: V. Holten, J. V. Sengers, M. A. Anisimov,
           J. Phys. Chem. Ref. Data 43, 014101 (2014).
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────
R = 461.523087        # J/(kg·K) — specific gas constant for water (per-kg)

# ── LLCP coordinates (mean-field) ─────────────────────────────────────────
Tc = 228.2            # K
Pc = 0.0              # MPa

# ── Reference density ────────────────────────────────────────────────────
rho0 = 1081.6482      # kg/m³

# ── Background pressure offset ───────────────────────────────────────────
P0 = -300.0           # MPa

# ── Derived pressure scale ───────────────────────────────────────────────
# P_scale = rho0 * R * Tc  [Pa]  — the unit for reduced pressure
P_scale_Pa = rho0 * R * Tc        # ~1.1383e8 Pa
P_scale_MPa = P_scale_Pa / 1e6    # ~113.83 MPa

# ── Mixing / field parameters (Table 6) ──────────────────────────────────
omega0 = 0.52122690
L0 = 0.76317954
k0 = 0.072158686
k1 = -0.31569232
k2 = 5.2992608

# ── Background coefficients (Table 7) ────────────────────────────────────
# 20 terms: c[i], a[i], b[i], d[i]
c_bg = np.array([
    -8.1570681381655, 1.2875032e+000, 7.0901673598012,
    -3.2779161e-002, 7.3703949e-001, -2.1628622e-001, -5.1782479e+000,
    4.2293517e-004, 2.3592109e-002, 4.3773754e+000, -2.9967770e-003,
    -9.6558018e-001, 3.7595286e+000, 1.2632441e+000, 2.8542697e-001,
    -8.5994947e-001, -3.2916153e-001, 9.0019616e-002, 8.1149726e-002,
    -3.2788213e+000,
])

a_bg = np.array([
    0, 0, 1, -0.2555, 1.5762, 1.64, 3.6385, -0.3828,
    1.6219, 4.3287, 3.4763, 5.1556, -0.3593, 5.0361, 2.9786, 6.2373,
    4.046, 5.3558, 9.0157, 1.2194,
])

b_bg = np.array([
    0, 1, 0, 2.1051, 1.1422, 0.951, 0, 3.6402,
    2.076, -0.0016, 2.2769, 0.0008, 0.3706, -0.3975, 2.973, -0.318,
    2.9805, 2.9265, 0.4456, 0.1298,
])

d_bg = np.array([
    0, 0, 0, -0.0016, 0.6894, 0.013, 0.0002, 0.0435,
    0.05, 0.0004, 0.0528, 0.0147, 0.8584, 0.9924, 1.0041, 1.0961,
    1.0228, 1.0303, 1.618, 0.5213,
])

# ── IAPWS-95 reference state alignment ─────────────────────────────────
# Offsets calibrated at T_ref=273.15 K, P_ref=0.1 MPa so that
# S_aligned = S_raw + S_OFFSET matches IAPWS-95 entropy,
# G_aligned = G_raw + H_OFFSET - T * S_OFFSET matches IAPWS-95 Gibbs.
T_REF = 273.15                # K
S_OFFSET = -0.0000995240      # J/(kg·K)
H_OFFSET = -0.0272789916      # J/kg
