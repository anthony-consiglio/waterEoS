"""
Shi & Tanaka (2020) hierarchical two-state EoS parameters.

Reference: R. Shi and H. Tanaka, PNAS 117, 26591 (2020). Table S2 of the
supporting information. All parameters refer to real (experimental) liquid
water unless noted.

Scaled (dimensionless) variables used internally
------------------------------------------------
    Tr = 308.15 K, Pr = 200 MPa,  Vr = kB * Tr / Pr  (per local-structural unit)
    T_hat = T / Tr,   P_hat = P / Pr,
    dT    = T_hat - 1,  dP   = P_hat - 1,
    G_hat = G / (kB * Tr).

Parameter conventions (from Table S2 caption)
---------------------------------------------
- DELTA_E_K  : ΔE / kB  in K  (the table reports -1952). To get the
  dimensionless ΔÊ that enters ΔĜ = ΔÊ - T̂·Δσ̂ + P̂·ΔV̂ (Eq. S2), divide by Tr.
- DELTA_SIGMA: Δσ / kB  dimensionless  (-8.317). Already the value of Δσ̂
  that enters Eq. S2 directly.
- DELTA_V_MK : ΔV listed in Table S2 with unit annotation "[MPa⁻¹·K]"
  (= K/MPa). This annotation is MISLEADING — the listed numerical value
  1.593 is already the dimensionless ΔV̂ that enters Eq. S2 directly, not
  a quantity in K/MPa requiring further Pr/Tr scaling.

  This was verified empirically (G:/.../waterEoS/_hyp_dV.py): using
  ΔV̂ = 1.593, the model reproduces (a) IAPWS-95 density to ~0.05% across
  255-320 K at 0.1 MPa, (b) TMD at 278.3 K matching real water (277.13 K),
  (c) κ_T at the reference state to 1.2%, and (d) the published Fig. 2
  curves to within figure-digitization precision. Applying the
  dimensionally-natural-looking ΔV·Pr/Tr = 1.034 instead gives a 17 K
  cold-shifted TMD, κ_T 17% too low at ambient, and α off by 54% — none
  of which match the published figures or IAPWS-95.

  Physical interpretation: ΔV̂ = 1.593 corresponds to ΔV per structural
  unit ≈ 3.4×10⁻²⁹ m³, i.e., a ~4 Å³/molecule LFTS-DNLS volume difference
  (n = 7.888 molecules per unit), consistent with the ice-like vs liquid-
  like picture of the two states.

  The transport-module parameters (ΔV^D, ΔV_a) carry the same "[MPa⁻¹·K]"
  annotation in Table S3 and are used with the same convention — see
  shi_tanaka_transport/params.py.

The polynomial background G_ρ(T̂, P̂) (Eq. S4) uses the specific c_{mn}
terms present in Table S2 — c_22, c_10, c_00 are explicitly absent.
"""

# ── Physical constants ────────────────────────────────────────────────────
R = 8.314462          # J/(mol·K), gas constant
kB = 1.380649e-23     # J/K, Boltzmann constant
NA = 6.02214076e23    # 1/mol, Avogadro constant
M_H2O = 0.018015268   # kg/mol, molar mass of water

# ── Reference state (Shi-Tanaka convention) ───────────────────────────────
Tr = 308.15           # K, reference temperature
Pr = 200.0            # MPa, reference pressure
Pr_Pa = Pr * 1e6      # Pa

# Per-structural-unit volume scale  Vr = kB Tr / Pr   [m³ per unit]
Vr = kB * Tr / Pr_Pa  # ≈ 2.126e-29 m³

# ── Two-state parameters (Table S2, H₂O column) ───────────────────────────
DELTA_E_K   = -1952.0     # ΔE / kB  [K]
DELTA_SIGMA = -8.317      # Δσ / kB  [dimensionless]
DELTA_V_MK  = 1.593       # ΔV       [MPa⁻¹·K]
N_UNIT      = 7.888       # avg molecules per local structural unit (n > 4)

# Dimensionless forms used in the equations
DELTA_E_HAT     = DELTA_E_K / Tr                # ≈ -6.334
DELTA_SIGMA_HAT = DELTA_SIGMA                   # ≈ -8.317  (already /kB)
DELTA_V_HAT     = DELTA_V_MK                    #  = 1.593  (table value is
                                                # already the dimensionless
                                                # ΔV̂; see docstring)

# ── Polynomial background coefficients for G_ρ (Table S2) ────────────────
# Term layout:  G_ρ = Σ c_{mn} ΔT̂^m ΔP̂^n  +  c_1 · T̂ · (ln T̂ - 1)
# Note: c_22, c_10, c_00 are explicitly absent (not free parameters).
c01 = 10.34
c02 = -0.2629
c03 = 0.03432
c11 = 1.309
c12 = -0.3383
c13 = 0.1090
c20 = -13.73
c21 = -0.7274
c23 = 0.7443
c30 = -0.3602
c31 = 2.062
c1  = -39.91             # coefficient of T̂·(ln T̂ - 1)

# ── Mass-density prefactor ────────────────────────────────────────────────
# ρ = (n · M_water_per_molecule) / V_per_unit
#   = (n · M_H2O / NA) / (Vr · V̂)
#   = (n · M_H2O / (NA · Vr)) / V̂
RHO_PREFACTOR = N_UNIT * M_H2O / (NA * Vr)      # kg/m³ when divided by V̂

# ── IAPWS-95 reference state alignment ────────────────────────────────────
# Placeholder offsets so S and G are returned as raw values from the model
# (zero of entropy / Gibbs is arbitrary in this model). Set to nonzero
# values after calibrating against IAPWS-95 at (T_REF, P_REF) if needed.
T_REF = 273.15           # K
S_OFFSET = 0.0           # J/(kg·K)
H_OFFSET = 0.0           # J/kg

# ── Validity range (Fig. 2 & S2 of the paper) ─────────────────────────────
# Density data covers ~240–320 K at 0.1 MPa with extrapolation to ~190 K
# under pressure; negative-pressure density validated to -110 MPa.
T_MIN = 180.0            # K  (LFTS dominates strongly below this)
T_MAX = 320.0            # K
P_MIN = -110.0           # MPa
P_MAX = 200.0            # MPa
