# Singh, Issenmann & Caupin (2017) Two-State Transport Model

> Two-state transport model for viscosity, self-diffusion, and rotational correlation time, using the Holten (2014) EoS as a thermodynamic backbone.

**Reference:** L. P. Singh, B. Issenmann, F. Caupin, "Pressure dependence of viscosity in supercooled water and a unified approach for thermodynamic and dynamic anomalies of water," *PNAS* **114**, 4312-4317 (2017).

## Overview

This model provides a unified description of water's transport anomalies by connecting them to the same two-state thermodynamic framework used for density anomalies. The key insight is that the LDL fraction `f(T, P)` from the Holten (2014) EoS — which explains the density anomaly — also explains the anomalous pressure dependence of viscosity, diffusion, and rotational dynamics in supercooled water.

The model describes a fragile-to-strong crossover in transport behavior:

- **High-density state (HDS):** Fragile liquid behavior with a Vogel-Fulcher-Tammann (VFT) divergence at `T_0 = 147.75 K`. The activation energy `E_HDS` enters the VFT term `E_HDS/(T - T_0)`, which diverges as T approaches the glass transition. An explicit pressure dependence via `dv_HDS` accounts for the pressure-volume work of molecular rearrangement.
- **Low-density state (LDS):** Strong liquid behavior with simple Arrhenius kinetics (`E_LDS / T`). The activation energy `E_LDS` is larger in magnitude but enters a non-divergent `1/T` term. No pressure dependence in the LDS term.

As water cools, the LDL fraction `f` increases, and transport properties smoothly transition from HDS-dominated (fragile) to LDS-dominated (strong) behavior. This explains why viscosity anomalously decreases with increasing pressure in the supercooled regime.

## Valid Range

- **Temperature:** ~244-298 K (limited by homogeneous ice nucleation at low T)
- **Pressure:** 0.1-400 MPa

## Key Equation

### Transport property (Eq. 1)

```
A(T, P) = A0 * (T / T_ref)^nu * exp{ eps * [(1-f) * (E_HDS/kB + dv_HDS*P/kB) / (T - T0)
                                              + f * E_LDS/kB / T] }
```

where:
- `A` is the transport property (eta, D, or tau_r)
- `A0` is a prefactor with property-specific units
- `T_ref = 273.15 K` (reference temperature)
- `nu` is a power-law exponent accounting for average molecular speed
- `eps = +1` for properties that increase with ordering (eta, tau_r), `-1` for those that decrease (D)
- `f = x` is the LDL fraction from the Holten (2014) model — **not fitted, computed from thermodynamics**
- `E_HDS/kB` and `E_LDS/kB` are activation energies in units of K
- `dv_HDS` is the HDS activation volume (m^3)
- `T_0 = 147.75 K` is the VFT singularity temperature (shared across all properties)
- `kB = 1.380649e-23 J/K` is the Boltzmann constant

The HDS term has the VFT form `1/(T - T_0)` which diverges as T approaches the glass transition, while the LDS term has the simple Arrhenius form `1/T`.

## Parameters (Table 1)

| Parameter | Viscosity (eta) | Self-diffusion (D) | Correlation time (tau_r) |
|-----------|----------------|--------------------|-----------------------|
| A0 | 38.75 uPa*s | 40330 um^2/s | 86.2 fs |
| E_HDS/kB | 421.9 K | 402.2 K | 395.0 K |
| E_LDS/kB | 2262 K | 1984 K | 2585 K |
| dv_HDS | 2.44e-30 m^3 | 1.79e-30 m^3 | 1.62e-30 m^3 |
| nu | 0.5 | 0.5 | -0.5 |
| eps | +1 | -1 | +1 |

**Common parameters:** `T_ref = 273.15 K`, `T_0 = 147.75 K`.

Note: The paper's Table 1 labels the larger activation energies as `E_HDS` and the smaller as `E_LDS`. The convention used here and in `params.py` follows the code implementation, where `E_HDS` (smaller) enters the VFT term `1/(T - T_0)` and `E_LDS` (larger) enters the Arrhenius term `1/T`.

## Properties Computed

| Key | Property | Unit |
|-----|----------|------|
| eta | Dynamic viscosity | Pa*s |
| D | Self-diffusion coefficient | m^2/s |
| tau_r | Rotational correlation time | s |
| f | LDL fraction (= x from Holten) | - |

When called via `compute_batch`, the full set of Holten thermodynamic properties (rho, V, S, G, H, Cp, etc.) is also included in the output.

## Implementation Notes

- **`params.py`** — All transport parameters from Table 1. Physical constants `kB` and `T_ref`.
- **`core.py`** — Scalar (`compute_properties`) and vectorized (`compute_batch`) computation. Internally calls `holten_eos.core` to obtain the LDL fraction `f = x`.
- **Backbone dependency:** This module imports from `holten_eos` at runtime. The Holten model provides all thermodynamic properties; this module adds the three transport properties on top.
- **No phase diagram or JAX module** — transport properties are computed as a simple post-processing step on top of the Holten EoS output.
