# Caupin & Anisimov (2019) Two-State EoS

> Extension of the two-state framework to negative pressures via a liquid-vapor spinodal Gibbs contribution, enabling modeling of stretched (metastable) water.

**Reference:** F. Caupin and M. A. Anisimov, "Thermodynamics of supercooled and stretched water: Unifying two-structure description and liquid-vapor spinodal," *J. Chem. Phys.* **151**, 034503 (2019).
**Erratum:** *J. Chem. Phys.* **163**, 039902 (2025). Corrected signs in polynomial coefficients c11, c03, c22, c14.

## Overview

This model extends the Holten et al. (2014) two-state framework to negative pressures where water is mechanically stretched. The key innovation is a spinodal Gibbs energy contribution `G^sigma` that ensures the isothermal compressibility diverges properly at the liquid-vapor spinodal boundary, which is critical for describing water under tension.

The total Gibbs energy of state A (HDL) is decomposed as:

```
G^A = G^sigma + G^poly
```

where `G^sigma` handles the spinodal divergence and `G^poly` is a polynomial background. The state B-A difference `G^BA`, interaction parameter `omega`, and equilibrium solver follow the same two-state mixing framework as Holten, but with molar (not per-kg) reduced variables.

Unlike Holten's 20-term background, the polynomial `G^poly` uses only ~13 coefficients `c_mn` in powers of `DeltaT_hat` and `DeltaP_hat`, making the model more compact.

## Liquid-Liquid Critical Point

| Parameter | Value |
|-----------|-------|
| T_c | 218.1348 K |
| P_c | 71.94655 MPa |
| V_c | 18.22426 cm^3/mol |

## Valid Range

- **Temperature:** 200-360 K
- **Pressure:** -160 to 400 MPa (extends into the negative-pressure regime)

## Key Equations

### Reduced variables

```
DeltaT_hat = (T - T_c) / T_c
DeltaP_hat = (P - P_c) * V_c / (R * T_c)
T_hat      = 1 + DeltaT_hat = T / T_c
```

where `R = 8.314462 J/(mol*K)` and `V_c = 18.22426e-6 m^3/mol`. The pressure scale is `P_scale = R*T_c/V_c ~ 99.53 MPa`.

### Liquid-vapor spinodal (Eq. 2)

```
P_s(T) = p_a + p_b * (T - 182) + p_c * (T - 182)^2   [MPa]
```

Parameterized from TIP4P/2005 molecular dynamics data. Coefficients: `p_a = -462 MPa`, `p_b = 2.61 MPa/K`, `p_c = -0.0065 MPa/K^2`.

### Spinodal Gibbs contribution (Eqs. 1, 8)

```
G^sigma = A_hat(T) * [P_hat - P_hat_s(T)]^{3/2}
```

where `A_hat(T) = A0 + A1 * DeltaT_hat`. The 3/2-power ensures that the isothermal compressibility (`kappa_T ~ -d^2G/dP^2`) diverges as `(P - P_s)^{-1/2}` at the spinodal. `A0 < 0` guarantees the correct sign of the divergence.

### Polynomial G^A (Eq. 6)

```
G^poly = sum_{m,n} c_mn * DeltaT_hat^m * DeltaP_hat^n
```

A 4th-degree polynomial with 13 coefficients (`c01` through `c14`). Replaces Holten's 20-term exponential background with a simpler algebraic form.

### State B-A Gibbs difference (Eq. 7)

```
G^BA = lambda * (DeltaT_hat + a*DeltaP_hat + b*DeltaT_hat*DeltaP_hat
                 + d*DeltaP_hat^2 + f*DeltaT_hat^2)
```

with `lambda = 1.653737`, encoding the free-energy difference between LDL and HDL states.

### Interaction parameter (Eq. 5)

```
omega_hat = (2 + omega0 * DeltaP_hat) / T_hat
```

with `omega0 = 0.1854443`. Note the explicit temperature dependence (divided by `T_hat`), unlike Holten where `omega` depends only on pressure.

### Equilibrium condition

```
G^BA + T_hat * [ln(x/(1-x)) + omega_hat * (1 - 2x)] = 0
```

Solved via Newton-Raphson from multiple starting points, with the globally stable root selected by minimum Gibbs energy of mixing.

### Property conversion

All thermodynamic properties are derived from the total reduced Gibbs energy derivatives, converted from molar reduced units to per-kg physical units via the molar mass `M_H2O = 0.018015268 kg/mol`.

## Parameters

### Set 1 (Table II / corrected Table III from 2025 erratum)

The implementation uses the corrected parameter set. Key parameters:

| Parameter | Value |
|-----------|-------|
| omega0 | 0.1854443 |
| lambda | 1.653737 |
| A0 | -0.08118730 |
| A1 | 0.05070641 |

Polynomial coefficients `c_mn` — see `params.py` for the full set (13 terms, with 4 signs corrected per the erratum).

## Properties Computed

| Key | Property | Unit |
|-----|----------|------|
| rho | Density | kg/m^3 |
| V | Specific volume | m^3/kg |
| S | Specific entropy | J/(kg*K) |
| G | Specific Gibbs energy | J/kg |
| H | Specific enthalpy | J/kg |
| U | Specific internal energy | J/kg |
| A | Specific Helmholtz energy | J/kg |
| Cp | Isobaric heat capacity | J/(kg*K) |
| Cv | Isochoric heat capacity | J/(kg*K) |
| Kt | Isothermal bulk modulus | MPa |
| Ks | Isentropic bulk modulus | MPa |
| alpha | Isobaric expansivity | 1/K |
| vel | Speed of sound | m/s |
| x | LDL fraction (state B) | - |
| Kp | dKt/dP (opt-in) | - |

State A and state B properties are also available with `_A` and `_B` suffixes.

## Implementation Notes

- **`params.py`** — All parameters with corrected signs from the 2025 erratum. Spinodal coefficients from TIP4P/2005.
- **`core.py`** — Scalar and vectorized (`compute_batch`) property computation. Uses molar reduced variables internally.
- **`core_ad.py`** — JAX-based autodiff version.
- **`phase_diagram.py`** — Spinodal and binodal curves, including the liquid-vapor spinodal boundary.
- **Reference state:** Aligned to IAPWS-95 at T = 273.15 K, P = 0.1 MPa.
- **Key difference from Holten:** Molar units (R in J/(mol*K), V_c in m^3/mol) vs. Holten's per-kg units.
