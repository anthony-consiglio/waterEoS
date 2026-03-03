# Duska (2020) Volume-Additive Two-State EoS (EOS-VaT)

> Volume-additive two-state equation of state that mixes at the volume level rather than the Gibbs energy level, using the vapor-liquid critical point as the natural reference state.

**Reference:** M. Duska, "Two-state equation of state of water above the spinodal," *J. Chem. Phys.* **152**, 174501 (2020).

## Overview

While Holten (2014) and Caupin (2019) use pressure-additive (Gibbs-level) mixing of two structural states, this model takes the alternative approach of volume-additive mixing:

```
V(T, P) = (1 - x) V_A(T, P) + x V_B(T, P)
```

where `V_A` and `V_B` are the specific volumes of the HDL and LDL states, and `x` is the equilibrium LDL fraction. This volume-additive formulation naturally captures the density anomaly and leads to a different mathematical structure for the state A properties.

The model is formulated entirely in reduced variables relative to the vapor-liquid critical point (VLCP) of water, not the liquid-liquid critical point. The LLCP location emerges as a prediction rather than an input. Spinodal curves for both states are parameterized as polynomial functions of reduced temperature, and state A properties (volume, entropy) are derived analytically from these spinodal curves via a modified Tait-like equation.

## Critical Points

### Vapor-Liquid Critical Point (reference state)

| Parameter | Value |
|-----------|-------|
| T_VLCP | 647.096 K |
| p_VLCP | 22.064 MPa |

### Predicted Liquid-Liquid Critical Point

| Parameter | Value |
|-----------|-------|
| T* | 220.9 K |
| p* | 54.2 MPa |
| rho* | 1022 kg/m^3 |

## Valid Range

- **Temperature:** 200-360 K
- **Pressure:** 0-400 MPa

## Key Equations

### Reduced variables

```
T_hat = T / T_VLCP
p_hat = P / p_VLCP
```

### Spinodal polynomials (Eqs. 6a-c)

The mechanical stability limit (spinodal) for state A is defined by three polynomial functions of `T_hat`:

**Spinodal pressure** (Eq. 6a):
```
p_S(T_hat) = 1 + d1*(T_hat - 1) + d2*(T_hat - 1)^2 + d3*(T_hat - 1)^3
```

**Spinodal volume** (Eq. 6b):
```
V_S(T_hat) = c0 + c1*T_hat + c2*T_hat^2 + c3*T_hat^3 + c4*T_hat^4
```

**Second pressure derivative** (Eq. 6c):
```
phi(T_hat) = b0 + b1*T_hat + b2*T_hat^2 + b3*T_hat^3 + b4*T_hat^4
```

These polynomials encode the shape of the equation of state near the spinodal, including the reentrant behavior of the state A spinodal at low temperatures.

### Auxiliary parameter B (Eqs. 14-16)

```
B = sqrt(-phi / (2 * p_S * V_S^2))
```

Derived from the spinodal properties. Controls the curvature of the pressure-density relation near the spinodal.

### State A volume (Eqs. 17-18)

```
V_A = V_S * B / (u + B)
```

where `u = sqrt(1 - p_hat / p_S)`. This is a modified Tait-like equation that smoothly connects the spinodal volume `V_S` to the compressed liquid volume.

### State A entropy (Eqs. 19-26)

The entropy is obtained by integrating the Maxwell relation `(dS/dP)_T = -(dV/dT)_P` along the spinodal, combined with a reference entropy `S_S(T_hat)` at the spinodal:

```
S_A = dpS*(V_A - V_S) + A_coeff*(...) + C_coeff*(...) + S_S
```

where `A_coeff` and `C_coeff` are combinations of spinodal derivatives, and `S_S` is computed from a polynomial (Eq. 25).

### Gibbs energy of state A (numerical integration)

The Gibbs energy `G_A` is obtained via 2D path integration from the VLCP `(p_hat=0, T_hat=1)` to `(p_hat, T_hat)` using 16-point Gauss-Legendre quadrature:

```
g_A = integral_1^{T_hat} (-S_A(0, T')) dT' + integral_0^{p_hat} V_A(p', T_hat) dp'
```

### Gibbs difference G^B - G^A (Eq. 5)

```
DeltaG = a0 + a1*p_hat*T_hat + a2*p_hat + a3*T_hat + a4*T_hat^2 + a5*p_hat^2 + a6*p_hat^3
```

A 7-coefficient polynomial. State B properties are derived analytically: `V_B = V_A + dDeltaG/dp`, `S_B = S_A - dDeltaG/dT`.

### Cooperativity (Eq. 33)

```
omega = w0 * (1 + w1*p_hat + w2*T_hat + w3*T_hat*p_hat)
```

A bilinear function of both temperature and pressure (4 parameters).

### Equilibrium condition (Eqs. 7-9)

```
DeltaG + T_hat * ln(x / (1-x)) + omega * (1 - 2x) = 0
```

Solved via Newton-Raphson from multiple starting points with globally stable root selection.

## Parameters

All 28 parameters are from Table I of the reference (with corrections for coefficient ordering confirmed by the author):

- **DeltaG polynomial:** 7 coefficients `a0`-`a6`
- **Spinodal phi:** 5 coefficients `b0`-`b4` (reversed order from Table I)
- **Spinodal volume:** 5 coefficients `c0`-`c4` (reversed order from Table I)
- **Spinodal pressure:** 3 coefficients `d1`-`d3` (d2/d3 swapped from Table I)
- **Cooperativity:** 4 coefficients `w0`-`w3`
- **Spinodal entropy:** 4 coefficients `s0`-`s3` (reversed order from Table I)

See `params.py` for corrected values with detailed notes on each correction.

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

- **`params.py`** — All 28 parameters with corrections for coefficient ordering (confirmed against the author's MATLAB implementation).
- **`core.py`** — Scalar (`compute_properties`) and vectorized (`compute_batch`) computation. Uses finite-difference for `dS_A/dT_hat` and Gauss-Legendre quadrature for `G_A`.
- **`core_ad.py`** — JAX-based autodiff version (avoids finite differences entirely).
- **`phase_diagram.py`** — Spinodal and binodal curves.
- **Reference state:** Aligned to IAPWS-95 at T = 273.15 K, P = 0.1 MPa.
- **Key difference from Holten/Caupin:** Volume-additive mixing rather than Gibbs-additive. Reduced variables referenced to VLCP (647 K, 22 MPa), not LLCP.
