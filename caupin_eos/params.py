"""
Caupin & Anisimov (2019) two-state EoS parameters.

Corrected Table III from 2025 erratum (without Kim et al. kT data).
Reference: F. Caupin and M. A. Anisimov, J. Chem. Phys. 151, 034503 (2019).
Erratum: J. Chem. Phys. 163, 039902 (2025).
"""

# ── Physical constants ────────────────────────────────────────────────────
R = 8.314462          # J/(mol·K), gas constant
M_H2O = 0.018015268   # kg/mol, molar mass of water

# ── LLCP coordinates ──────────────────────────────────────────────────────
Tc = 218.1348          # K
Pc = 71.94655          # MPa
Vc = 18.22426e-6       # m³/mol

# ── Derived pressure scale ────────────────────────────────────────────────
# P_scale = R*Tc/Vc  [Pa]  — the pressure unit for ΔP̂
P_scale_Pa = R * Tc / Vc              # ~99.53e6 Pa
P_scale_MPa = P_scale_Pa / 1e6        # ~99.53 MPa

# Reduced critical pressure: P̂c = Pc / P_scale
Phc = Pc / P_scale_MPa                # ~0.7229

# ── Interaction parameter ─────────────────────────────────────────────────
omega0 = 0.1854443

# ── State B−A Gibbs difference (Eq. 7) ────────────────────────────────────
lam = 1.653737         # λ
a = 0.1030250
b = -0.0392417
d = -0.01039947
f = 1.021512

# ── Spinodal Gibbs contribution Â(T) (Eq. 8) ─────────────────────────────
A0 = -0.08118730       # Â₀  (< 0 ensures κT divergence)
A1 = 0.05070641        # Â₁

# ── Polynomial coefficients c_mn for Ĝ^A (Eq. 6, corrected signs) ────────
# c_mn: m = ΔT̂ power, n = ΔP̂ power
c01 = 1.126869         # constrained by Eq. 9
c02 = 0.01005341
c11 = -0.2092770       # CORRECTED sign (erratum)
c20 = -2.520114
c03 = -0.001149367     # CORRECTED sign (erratum)
c12 = -0.008992042
c21 = 0.2118502
c30 = 0.1087670
c04 = 0.00007573062
c13 = 0.002393927
c22 = -0.01831198      # CORRECTED sign (erratum)
c40 = 0.02803712
c14 = -0.0001641608    # CORRECTED sign (erratum)

# ── Liquid-vapor spinodal (Eq. 2, from TIP4P/2005) ───────────────────────
# Ps(T) = ps_a + ps_b*(T - 182) + ps_c*(T - 182)²   [MPa, T in K]
ps_a = -462.0          # MPa
ps_b = 2.61            # MPa/K
ps_c = -0.0065         # MPa/K²
ps_T0 = 182.0          # K, reference temperature

# ── IAPWS-95 reference state alignment ─────────────────────────────────
# Offsets calibrated at T_ref=273.15 K, P_ref=0.1 MPa so that
# S_aligned = S_raw + S_OFFSET matches IAPWS-95 entropy,
# G_aligned = G_raw + H_OFFSET - T * S_OFFSET matches IAPWS-95 Gibbs.
T_REF = 273.15                  # K
S_OFFSET = -146.1582559570      # J/(kg·K)
H_OFFSET = 112500.2342686583    # J/kg
