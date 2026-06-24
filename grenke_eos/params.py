"""
Parameters for the Grenke & Elliott (2025) Tait-Tammann EoS.

Reference:
  Grenke & Elliott, J. Phys. Chem. B 129, 1997-2012 (2025)
  Correction: J. Phys. Chem. B 129, 9850-9853 (2025)
"""

# Reference conditions
T0 = 273.15        # K
P0 = 101325.0      # Pa

# v0(T, P0) parameters (eq 20, Table 3)
a1 = 68.4089       # m^3/kg
a2 = -0.0611145    # 1/K
a3 = 2.26928e-8    # m^3/kg
a4 = 0.0215553     # 1/K
a5 = 9.88107e-4    # m^3/kg

# B(T) parameters (eq 23, Table 8) — output *1e8 gives Pa
b1 = 3.1520397
b2 = 203.8085375   # K
b3 = -11.1985548
b4 = 7.2689427

# C(T) parameters (eq 24, Table 8)
c1 = 0.0790029
c2 = 237.9009619   # K
c3 = -14.8806681
c4 = 0.8778057
c5 = 0.0532605

# cp0(T, P0) parameters (eq 29, Table 6)
d1 = 4.44575e12    # J/(kg*K) — note: x10^12 from Table 6
d2 = -0.0928377    # 1/K
d3 = 4172.09       # J/(kg*K)

# IAPWS-95 reference state alignment at (273.15 K, 0.1 MPa)
T_REF = 273.15     # K
S_OFFSET = -0.147659   # J/(kg*K)
H_OFFSET = 61.009672   # J/kg
