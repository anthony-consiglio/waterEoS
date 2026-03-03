# Grenke & Elliott (2025) Tait-Tammann EoS

> Direct empirical Tait-Tammann correlation for liquid water — no two-state decomposition, purely analytic with 17 parameters.

**Reference:** J. H. Grenke and J. A. W. Elliott, "Analytic Correlation for the Thermodynamic Properties of Water at Low Temperatures (200-300 K) and High Pressures (0.1-400 MPa)," *J. Phys. Chem. B* **129**, 1997-2012 (2025).
**Correction:** *J. Phys. Chem. B* **129**, 9850-9853 (2025). Corrected sign error in Eq. 14 (Q function) and Table 12 parameter values.

## Overview

Unlike the two-state models (Holten, Caupin, Duska), this model takes a purely empirical approach: a modified Tait-Tammann equation directly correlates the specific volume as a function of temperature and pressure, with no underlying physical model of HDL/LDL structural states. All thermodynamic properties are obtained analytically from the specific volume equation and a reference heat capacity, via exact analytical derivatives and path integration.

The model has only 17 adjustable parameters (compared to ~28-70 for the two-state models) and achieves an overall R^2 = 0.9991 against experimental data. The trade-off is that extrapolation beyond the fitted range is less reliable, and the model does not predict phase behavior (no LLCP, spinodals, or LDL fraction).

## Valid Range

- **Temperature:** 200-300 K
- **Pressure:** 0.1-400 MPa

Note the narrower temperature range compared to the two-state models (which extend to 360 K).

## Key Equations

### Tait-Tammann specific volume (Eq. 1)

```
v(T, P) = v0(T) * [1 - C(T) * ln((B(T) + P) / (B(T) + P0))]
```

where `v0(T)` is the specific volume at reference pressure `P0 = 101325 Pa`, and `B(T)` and `C(T)` are temperature-dependent functions.

### Base functions

**Reference volume** (Eq. 20):
```
v0(T) = a1*exp(a2*T) + a3*exp(a4*T) + a5
```
5 parameters (Table 3): `a1 = 68.4089 m^3/kg`, `a2 = -0.0611145 K^{-1}`, etc.

**Tait parameter B** (Eq. 23):
```
B(T) = b1 / (1 + (T/b2)^b3)^b4 * 1e8   [Pa]
```
4 parameters (Table 8): a generalized logistic function.

**Tait parameter C** (Eq. 24):
```
C(T) = c1 / (1 + (T/c2)^c3)^c4 + c5
```
5 parameters (Table 8): a generalized logistic with offset. Dimensionless.

**Reference heat capacity** (Eq. 29):
```
cp0(T) = d1*exp(d2*T) + d3
```
3 parameters (Table 6): `d1 = 4.44575e12 J/(kg*K)`, `d2 = -0.0928377 K^{-1}`, `d3 = 4172.09 J/(kg*K)`.

### Derivative expressions

All derivatives of `B(T)` and `C(T)` are computed analytically:

- `B_T`, `B_TT` (Eqs. 25-26): first and second temperature derivatives of B
- `C_T`, `C_TT` (Eqs. 27-28): first and second temperature derivatives of C
- `v_T = dv/dT` and `v_P = dv/dP` follow from the chain rule applied to Eq. 1

### Thermodynamic properties

**Isothermal compressibility** (Eq. 2):
```
kappa_T = v0 * C / (v * (B + P))
```

**Isobaric expansivity** (Eq. 3):
```
alpha_P = v_T / v
```

**Isobaric heat capacity** (Eq. 4):
```
cp(T, P) = cp0(T) - T * integral_{P0}^{P} v_TT dP'
```

The integral is evaluated analytically using auxiliary functions.

**Speed of sound** (Eq. 5):
```
w = sqrt(v / kappa_S)
```

### Path integration (Eqs. 10-15)

The pressure integral of `v_TT` is decomposed into auxiliary functions:

| Function | Eq. | Role |
|----------|-----|------|
| J(T,P) | 13 | Volume integral: `(B+P)[ln((B+P)/(B+P0)) - 1] + B+P0` |
| K(T,P) | 12 | `B_T * ln((B+P)/(B+P0))` |
| L(T,P) | 10 | `B_TT * ln((B+P)/(B+P0))` |
| M(T,P) | 11 | `-B_T^2 * [1/(B+P) - 1/(B+P0)]` |
| R(T,P) | 15 | `(P - P0) - C*J` |
| **Q(T,P)** | **14** | **`-C*K + C*(B_T/(B+P0))*(P-P0) - C_T*J`** (corrected) |
| N(T,P) | 9 | Full `v_TT` integral combining L, M, K, J and derivatives |

The Q function had a sign error in the original paper (`C_T*J` was `+` instead of `-`), corrected in the 2025 correction paper and in this implementation.

The full heat capacity integral is:
```
integral v_TT dP = v0*N + 2*v0_T*Q + v0_TT*R
```

### Entropy and enthalpy (path integration)

From the reference state `(T0, P0) = (273.15 K, 0.1 MPa)`:

**Temperature leg** (at P0):
```
dh_T = (d1/d2)*(exp(d2*T) - exp(d2*T0)) + d3*(T - T0)
ds_T = d1*(Ei(d2*T) - Ei(d2*T0)) + d3*ln(T/T0)
```
where `Ei` is the exponential integral arising from `integral exp(d2*T)/T dT`.

**Pressure leg** (at T):
```
dh_P = v0*R - T*(v0*Q + v0_T*R)
ds_P = -(v0*Q + v0_T*R)
```

## Parameters

17 total parameters in 4 groups:

| Group | Parameters | Count | Source |
|-------|-----------|-------|--------|
| v0(T) | a1-a5 | 5 | Table 3 |
| B(T) | b1-b4 | 4 | Table 8 |
| C(T) | c1-c5 | 5 | Table 8 |
| cp0(T) | d1-d3 | 3 | Table 6 |

See `params.py` for exact values.

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
| Kp | dKt/dP (always NaN) | - |

No LDL fraction, state A/B decomposition, or phase diagram is available.

## Implementation Notes

- **`params.py`** — All 17 parameters plus IAPWS-95 alignment offsets.
- **`core.py`** — Scalar (`compute_properties`) and vectorized (`compute_batch`) computation. All derivatives are fully analytical; the exponential integral `Ei` (from `scipy.special.expi`) is used for the entropy temperature leg.
- **Reference state:** Aligned to IAPWS-95 at T = 273.15 K, P = 0.1 MPa.
- **Correction applied:** The Q function (Eq. 14 / A.44) uses the corrected sign from the 2025 correction paper.
- **No JAX or phase diagram modules** — the model is purely empirical with no two-state structure.
