"""
Caupin & Anisimov (2019) two-state EoS parameters — WITH Kim et al. data.

This is Table II of the reference: the fit that includes the isothermal
compressibility data of Kim et al. (Science 358, 1589, 2017). The
companion module ``params`` holds Table III (the fit without those data,
which the paper treats as its preferred result).

Table II is used exactly as printed in the 2019 paper. The 2025 erratum
(J. Chem. Phys. 163, 039902) corrects sign misprints in c11, c03, c22,
c14 of *Table III only*; Table II is unaffected.

Reference: F. Caupin and M. A. Anisimov, J. Chem. Phys. 151, 034503 (2019).
"""

# ── Physical constants ────────────────────────────────────────────────────
R = 8.314462          # J/(mol·K), gas constant
M_H2O = 0.018015268   # kg/mol, molar mass of water

# ── LLCP coordinates (Table II) ───────────────────────────────────────────
Tc = 219.4717          # K
Pc = 58.73897          # MPa
Vc = 18.74395e-6       # m³/mol  (rho_c = M/Vc = 961.12 kg/m³)

# ── Derived pressure scale ────────────────────────────────────────────────
P_scale_Pa = R * Tc / Vc
P_scale_MPa = P_scale_Pa / 1e6
Phc = Pc / P_scale_MPa

# ── Interaction parameter ─────────────────────────────────────────────────
omega0 = 0.2931166

# ── State B−A Gibbs difference (Eq. 7) ────────────────────────────────────
lam = 2.701194         # λ
a = 0.06424121
b = -0.05480673
d = -0.007881021
f = 0.5910234

# ── Spinodal Gibbs contribution Â(T) (Eq. 8) ─────────────────────────────
A0 = -0.08666044       # Â₀  (< 0 ensures κT divergence)
A1 = 0.1116703         # Â₁

# ── Polynomial coefficients c_mn for Ĝ^A (Eq. 6, Table II as printed) ────
# c_mn: m = ΔT̂ power, n = ΔP̂ power
c01 = 1.113806         # constrained by Eq. 9
c02 = 0.01422090
c11 = -0.3451214
c20 = -1.040621
c03 = -0.001587632
c12 = -0.03211844
c21 = 0.2203273
c30 = -1.290837
c04 = 0.0001027115
c13 = 0.003822330
c22 = -0.01169451
c40 = 0.5408265
c14 = -0.0002424047

# ── Liquid-vapor spinodal (Eq. 2, from TIP4P/2005) ───────────────────────
# Identical to the without-Kim fit — the spinodal is not fitted, it is
# fixed from TIP4P/2005 molecular dynamics.
ps_a = -462.0          # MPa
ps_b = 2.61            # MPa/K
ps_c = -0.0065         # MPa/K²
ps_T0 = 182.0          # K, reference temperature

# ── IAPWS-95 reference state alignment ─────────────────────────────────
# Offsets calibrated at T_ref=273.15 K, P_ref=0.1 MPa so that
# S_aligned = S_raw + S_OFFSET matches IAPWS-95 entropy and
# G_aligned = G_raw + H_OFFSET - T * S_OFFSET matches IAPWS-95 Gibbs.
# Recomputed for the Table II parameter set (see calibration in the
# tests / scripts) — the without-Kim offsets do NOT transfer because the
# absolute raw G and S differ between the two fits.
T_REF = 273.15                  # K
S_OFFSET = 294.5902568285       # J/(kg·K)
H_OFFSET = 201073.8069705671    # J/kg
