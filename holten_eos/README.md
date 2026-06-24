# Holten, Sengers & Anisimov (2014) Two-State EoS

> Gibbs-energy-based two-state model treating supercooled liquid water as a pressure-additive mixture of high-density (HDL) and low-density (LDL) structural states.

**Reference:** V. Holten, J. V. Sengers, M. A. Anisimov, "Equation of state for supercooled water at pressures up to 400 MPa," *J. Phys. Chem. Ref. Data* **43**, 014101 (2014).

## Overview

This model describes the anomalous thermodynamic behavior of supercooled water using a two-state framework rooted in the liquid-liquid critical point (LLCP) hypothesis. Water is treated as a mixture of two interconvertible local structures:

- **State A (HDL):** High-density, disordered liquid — dominant at high temperatures and pressures.
- **State B (LDL):** Low-density, tetrahedrally coordinated liquid — dominant at low temperatures.

The total Gibbs energy of the mixture is:

```
G = (1 - x) G_A + x G_B + G_mix
```

where `x` is the equilibrium fraction of state B (LDL), determined by minimizing the total Gibbs energy. The mixing contribution includes ideal mixing entropy and a mean-field interaction term controlled by the cooperativity parameter `omega`.

The model uses two sets of reduced variables: `(tau, pi)` for the 20-term background `B`, and `(t, p_red)` for the hyperbolic field `L` and cooperativity `omega`. This dual-variable scheme follows the original MATLAB reference implementation.

## Liquid-Liquid Critical Point

| Parameter | Value |
|-----------|-------|
| T_c | 228.2 K |
| P_c | 0 MPa |

The LLCP is placed at zero pressure in this mean-field formulation.

## Valid Range

- **Temperature:** 200-360 K
- **Pressure:** 0-400 MPa

## Key Equations

### Reduced variables

```
tau = T / T_c                           (background B)
pi  = (P - P0) / (rho0 * R * T_c)      (background B)
t   = (T - T_c) / T_c = tau - 1        (field L, omega)
p   = (P - P_c) / (rho0 * R * T_c)     (field L, omega)
```

where `R = 461.523087 J/(kg*K)` is the specific gas constant, `rho0 = 1081.6482 kg/m^3`, and `P0 = -300 MPa`.

### Background Gibbs energy B (Eq. 12 / Table 7)

```
B(tau, pi) = sum_{i=1}^{20} c_i * tau^{a_i} * pi^{b_i} * exp(-d_i * pi)
```

A 20-term expansion with coefficients `(c_i, a_i, b_i, d_i)` from Table 7. All first and second partial derivatives with respect to `tau` and `pi` are computed analytically in a single pass.

### Hyperbolic field L (Eq. 14)

The field `L(t, p)` encodes the Gibbs energy difference between states A and B as a smooth function that correctly maps the mixed-field character of the liquid-liquid transition. It involves parameters `L0`, `k0`, `k1`, `k2` from Table 6.

### Cooperativity parameter omega (Eq. 15)

```
omega = 2 + omega0 * p
```

where `omega0 = 0.52122690` (Table 6). When `omega > 2`, the system can exhibit a first-order liquid-liquid transition.

### Equilibrium condition (Eq. 10)

The equilibrium fraction `x` satisfies:

```
L + ln(x / (1 - x)) + omega * (1 - 2x) = 0
```

Solved via bisection with the "L < 0 flip trick" from the reference MATLAB code: when `L < 0`, negate `L`, solve for the guaranteed-small root, then return `1 - x`.

### Thermodynamic properties (Eqs. 23-27)

Properties are derived from the order parameter `f = 2x - 1` and the susceptibility `chi = 1 / [2/(1 - f^2) - omega]`:

- **Entropy:** `S = R * [-0.5*(f+1)*L_t*tau - g0 - B_t]`
- **Volume:** `V = (rho0)^{-1} * [0.5*tau*(omega0/2*(1-f^2) + L_p*(f+1)) + B_p]`
- **Compressibility:** from `-(1/V) d^2G/dP^2` involving `chi*(L_p - omega0*f)^2`
- **Expansivity:** from `(1/V) d^2G/dPdT`
- **Heat capacity:** from `-T d^2G/dT^2` involving `L_t^2 * chi`

The chi-dependent terms produce the characteristic anomalous divergence near the LLCP.

## Parameters

### Mixing and field parameters (Table 6)

| Parameter | Value |
|-----------|-------|
| omega0 | 0.52122690 |
| L0 | 0.76317954 |
| k0 | 0.072158686 |
| k1 | -0.31569232 |
| k2 | 5.2992608 |

### Background coefficients (Table 7)

20 sets of `(c_i, a_i, b_i, d_i)` — see `params.py` for the full arrays.

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

All properties are also computed individually for state A (`_A` suffix) and state B (`_B` suffix).

## Implementation Notes

- **`params.py`** — All model parameters, reference constants, and IAPWS-95 alignment offsets.
- **`core.py`** — Scalar and vectorized (`compute_batch`) property computation. Follows the MATLAB reference implementation exactly.
- **`core_ad.py`** — JAX-based autodiff version for gradient-aware applications.
- **`phase_diagram.py`** — Spinodal and binodal curve computation via Newton-Raphson root finding.
- **Reference state:** Aligned to IAPWS-95 at T = 273.15 K, P = 0.1 MPa via entropy and enthalpy offsets.
- **Units:** All internal calculations use per-kg specific quantities (no molar mass conversion needed).
