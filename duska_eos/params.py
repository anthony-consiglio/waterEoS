"""
EOS-VaT model parameters from Duška (2020), Table I.

Reference: M. Duška, "Water above the spinodal",
J. Chem. Phys. 152, 174501 (2020).
"""

import numpy as np

# ── Vapor-Liquid Critical Point (VLCP) of water ──────────────────────────
T_VLCP = 647.096       # K
p_VLCP = 22.064        # MPa
R_specific = 461.523   # J/(kg·K), specific gas constant R_u / M_H2O

# Unit conversion: reduced volume -> physical density
# rho = rho_scale / Vhat,  where rho_scale = p_VLCP / (R_specific * T_VLCP)
rho_scale = p_VLCP * 1e6 / (R_specific * T_VLCP)  # kg/m³  (≈ 73.86)

# ── Polynomial coefficients: G^B - G^A  (Eq. 5) ─────────────────────────
# DeltaG = a0 + a1*ph*Th + a2*ph + a3*Th + a4*Th² + a5*ph² + a6*ph³
a = np.array([
    -4.3743227e-1,   # a0
    -1.3836753e-2,   # a1
     1.8525106e-2,   # a2
     4.3306058e-1,   # a3
     2.1944047e+0,   # a4
    -1.6301740e-5,   # a5
     7.6204693e-6,   # a6
])

# ── Spinodal: second derivative of pressure wrt density (Eq. 6c) ────────
# phi(Th) = b0 + b1*Th + b2*Th² + b3*Th³ + b4*Th⁴
# NOTE: Author confirmed coefficients are listed in reversed order in Table I.
# Values below are corrected (reversed from the published table).
b = np.array([
     2.6732998e+1,   # b0
    -1.0405443e+2,   # b1
     2.1364435e+2,   # b2
    -2.3582144e+2,   # b3
     1.0783316e+2,   # b4
])

# ── Spinodal: volume (Eq. 6b) ───────────────────────────────────────────
# VS(Th) = c0 + c1*Th + c2*Th² + c3*Th³ + c4*Th⁴
# NOTE: Author confirmed coefficients are listed in reversed order in Table I.
# Values below are corrected (reversed from the published table).
c = np.array([
     7.3009898e-2,   # c0
    -8.9096098e-3,   # c1  (NOTE: exponent is -3, not -2; misprint in Table I)
     5.7261662e-2,   # c2
    -1.3084560e-2,   # c3
     7.7905108e-3,   # c4
])

# ── Spinodal: pressure (Eq. 6a) ─────────────────────────────────────────
# pS(Th) = 1 + d1*(Th-1) + d2*(Th-1)² + d3*(Th-1)³
# Note: d0 is implicitly 1 (pS = 1 at Th = 1, the VLCP)
# NOTE: In Table I, d2 and d3 are listed in swapped positions.
# Corrected ordering below (confirmed against working MATLAB implementation).
d = np.array([
     1.2756957e+1,   # d1
    -2.6960321e+0,   # d2  (listed as d3 in Table I)
     2.8548221e+1,   # d3  (listed as d2 in Table I)
])

# ── Cooperativity (Eq. omega) ────────────────────────────────────────────
# omega = w0 * (1 + w1*ph + w2*Th + w3*Th*ph)
w = np.array([
     4.1420925e-1,   # omega0
     3.6615174e-2,   # omega1
     1.6181775e+0,   # omega2
     7.1477190e-3,   # omega3
])

# ── Entropy at the spinodal (Eq. 8) ─────────────────────────────────────
# Th * SS'(Th) = s0 + s1*Th + s2*Th² + s3*Th³
# NOTE: Author confirmed coefficients are listed in reversed order in Table I.
# Values below are corrected (reversed from the published table).
s = np.array([
    -6.3674996e+0,   # s0
     8.7732559e+1,   # s1
    -1.7214704e+2,   # s2
     1.1210116e+2,   # s3
])

# ── IAPWS-95 reference state alignment ─────────────────────────────────
# Offsets calibrated at T_ref=273.15 K, P_ref=0.1 MPa so that
# S_aligned = S_raw + S_OFFSET matches IAPWS-95 entropy,
# G_aligned = G_raw + H_OFFSET - T * S_OFFSET matches IAPWS-95 Gibbs.
T_REF = 273.15                     # K
S_OFFSET = -13463.3332601599       # J/(kg·K)
H_OFFSET = -9540603.2122842800     # J/kg
