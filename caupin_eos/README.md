# Caupin & Anisimov (2019) Two-State EoS

> Extension of the two-state framework to negative pressures via a liquid-vapor spinodal Gibbs contribution, enabling modeling of stretched (metastable) water.

**Reference:** F. Caupin and M. A. Anisimov, "Thermodynamics of supercooled and stretched water: Unifying two-structure description and liquid-vapor spinodal," *J. Chem. Phys.* **151**, 034503 (2019).
**Erratum:** *J. Chem. Phys.* **163**, 039902 (2025). Corrected signs in polynomial coefficients c11, c03, c22, c14.

## Overview

This model extends the two-state framework (Holten et al., 2014) to negative pressures where water is mechanically stretched. It adds a spinodal Gibbs energy contribution `G^sigma` that ensures the isothermal compressibility diverges properly at the liquid-vapor spinodal boundary, which is required for describing water under tension.

The total Gibbs energy of state A (HDL) is decomposed as:

```
G^A = G^sigma + G^poly
```

where `G^sigma` handles the spinodal divergence and `G^poly` is a polynomial background. The state B-A difference `G^BA`, interaction parameter `omega`, and equilibrium solver follow the standard two-state mixing framework, formulated in molar reduced variables.

The polynomial background `G^poly` uses ~13 coefficients `c_mn` in powers of `DeltaT_hat` and `DeltaP_hat`, giving a compact algebraic form.

## Two Parameter Sets (with / without Kim et al. data)

The reference reports two fits that differ by whether the isothermal
compressibility data of Kim et al. (*Science* **358**, 1589, 2017) are
included. Both are available:

| | `caupin2019` (default, Table III) | `caupin2019_kim` (Table II) |
|---|---|---|
| Kim et al. κ_T data | excluded | included |
| LLCP T_c | 218.1348 K | 219.4717 K |
| LLCP P_c | 71.94655 MPa | 58.73897 MPa |
| V_c | 18.22426 cm³/mol | 18.74395 cm³/mol |
| ρ_c | 988.53 kg/m³ | 961.12 kg/m³ |
| overall reduced χ² | 0.97 | 1.23 |

The without-Kim fit (`caupin2019`) is the paper's preferred result and the
default here; the reported fit quality is better, particularly for κ_T and
sound velocity. The with-Kim fit (`caupin2019_kim`) reproduces the sharp
isothermal-compressibility peak (≈1 GPa⁻¹ near 226 K at saturated vapour
pressure) and a non-monotonic pressure dependence of the low-density
fraction that follow from including the Kim et al. data.

Table III parameters carry four sign corrections from the 2025 erratum
(c11, c03, c22, c14); Table II is used as printed in the 2019 paper (the
erratum corrects Table III only).

The tables and equations below describe the default `caupin2019`
(Table III) fit; `caupin2019_kim` uses the same equations with the
Table II parameters in `params_kim.py`.

## Liquid-Liquid Critical Point

| Parameter | Value |
|-----------|-------|
| T_c | 218.1348 K |
| P_c | 71.94655 MPa |
| V_c | 18.22426 cm^3/mol |

## Valid Range

- **Temperature:** 200-300 K
- **Pressure:** -140 to 400 MPa (extends into the stretched / negative-pressure regime)

These bounds follow Caupin & Anisimov (2019), which reproduces the experimentally observed anomalies in metastable water up to 400 MPa and down to -140 MPa. The underlying liquid-vapor spinodal lies far lower (`P_s` is about -462 MPa near 182 K), enabling physically based extrapolation toward the spinodal where data are unavailable.

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

A 4th-degree polynomial with 13 coefficients (`c01` through `c14`) providing the algebraic background contribution.

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

with `omega0 = 0.1854443`. The interaction parameter carries an explicit temperature dependence through the division by `T_hat`, in addition to its linear pressure term.

### Equilibrium condition

```
G^BA + T_hat * [ln(x/(1-x)) + omega_hat * (1 - 2x)] = 0
```

Solved via Newton-Raphson from multiple starting points, with the globally stable root selected by minimum Gibbs energy of mixing.

### Property conversion

All thermodynamic properties are derived from the total reduced Gibbs energy derivatives, converted from molar reduced units to per-kg physical units via the molar mass `M_H2O = 0.018015268 kg/mol`.

## Parameters

### Default fit — Table III (without Kim et al. data, erratum-corrected)

Key parameters:

| Parameter | Value |
|-----------|-------|
| omega0 | 0.1854443 |
| lambda | 1.653737 |
| A0 | -0.08118730 |
| A1 | 0.05070641 |

Polynomial coefficients `c_mn` — see `params.py` for the full set (13 terms,
with 4 signs corrected per the 2025 erratum: c11, c03, c22, c14).

### With-Kim fit — Table II

The `caupin2019_kim` variant uses Table II (`params_kim.py`), as printed in
the 2019 paper. Key parameters:

| Parameter | Value |
|-----------|-------|
| omega0 | 0.2931166 |
| lambda | 2.701194 |
| A0 | -0.08666044 |
| A1 | 0.1116703 |

Both parameter sets are aligned to IAPWS-95 at (273.15 K, 0.1 MPa) via their
own entropy/enthalpy offsets (the offsets differ because the absolute raw
G and S differ between the two fits).

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
- **Units:** Molar reduced variables internally (R in J/(mol*K), V_c in m^3/mol), converted to per-kg specific quantities for output.
