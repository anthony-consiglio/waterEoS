"""
Shi & Tanaka (2020) hierarchical-two-state transport-property parameters.

Reference: R. Shi and H. Tanaka, PNAS 117, 26591 (2020).
Parameters from Table S3 of the supporting information (H₂O column).

Three transport quantities are computed:

    eta   — dynamic viscosity                  [Pa·s]
    D     — self-diffusion coefficient         [m²/s]
    tau_r — rotational relaxation time         [s]

All three share the dynamic order parameter s^D from the hierarchical
extension of the two-state model (Eq. 6), but differ in their generalized
Arrhenius prefactor (x_0), high-T activation (E_a^0), activation volume
(ΔV_a), slow-fast activation difference (ΔE_a), Stokes-Einstein exponent λ,
and sign convention (X is viscosity or 1/D — diffusion is INVERSELY related
to the Arrhenius exponent).

All "energy" parameters are stored as E/k_B in K, matching the table
convention; pressures are in MPa.
"""

# ── Dynamic order parameter (s^D) — Eq. 6 ────────────────────────────────
# Shared across all three transport quantities.
DELTA_E_D_K     = -2356.0     # ΔE^D / k_B  [K]
DELTA_SIGMA_D   = -11.40      # Δσ^D / k_B  [dimensionless]
DELTA_V_D_MK    = 2.054       # ΔV^D        [MPa⁻¹·K]
B_M2K           = -0.00316    # quadratic-P curvature term [MPa⁻²·K]

# ── Viscosity η  (X = η, λ = 1, sign +) ─────────────────────────────────
# log x_0 reported in mPa·s; convert to Pa·s when applying.
LOG_X0_ETA      = -2.950      # log10(η_0 / mPa·s)
E_A0_ETA_K      = 1896.0      # E_a^0 / k_B  [K]
DV_A_ETA_MK     = 0.2867      # ΔV_a         [MPa⁻¹·K]
DE_A_ETA_K      = 2093.0      # ΔE_a / k_B   [K]
LAMBDA_ETA      = 1.0
X0_ETA_PAS      = (10.0 ** LOG_X0_ETA) * 1e-3       # Pa·s

# ── Self-diffusion D  (X = 1/D, λ = 0, sign +) ─────────────────────────
# log x_0 reported in 10⁻¹⁰ m²/s for D itself.
LOG_X0_D        = 3.954       # log10(D_0 / [10⁻¹⁰ m²/s])
E_A0_D_K        = 1645.0
DV_A_D_MK       = 0.4452
DE_A_D_K        = 2725.0
LAMBDA_D        = 0.0
D0_M2_S         = (10.0 ** LOG_X0_D) * 1e-10         # m²/s

# ── Rotational relaxation τ_R  (X = τ, λ = 0, sign +) ─────────────────
# log x_0 reported in ps.
LOG_X0_TAU      = -2.116      # log10(τ_0 / ps)
E_A0_TAU_K      = 1650.0
DV_A_TAU_MK     = 0.1837
DE_A_TAU_K      = 2586.0
LAMBDA_TAU      = 0.0
TAU0_S          = (10.0 ** LOG_X0_TAU) * 1e-12       # s

# Reference temperature for the (T/T_r)^λ prefactor — Tr from the
# thermodynamic model (Eq. S5 reference state).
T_REF = 308.15
