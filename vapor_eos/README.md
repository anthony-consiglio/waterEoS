# Water Vapor

> Water vapor as a Gibbs function of temperature and pressure, on the IAPWS-95 reference state shared by the liquids and ices of this package: the IAPWS-95 ideal gas with the second and third virial coefficients of the IAPWS-95 residual part.

**References:**
W. Wagner and A. Pruß, "The IAPWS Formulation 1995 for the Thermodynamic Properties of Ordinary Water Substance for General and Scientific Use," *J. Phys. Chem. Ref. Data* **31**, 387 (2002); IAPWS R6-95(2018).
W. Wagner, T. Riethmann, R. Feistel, and A. H. Harvey, *J. Phys. Chem. Ref. Data* **40**, 043103 (2011); IAPWS R14-08(2011) (sublimation pressure).
W. Wagner and A. Pruß, *J. Phys. Chem. Ref. Data* **22**, 783 (1993); IAPWS SR1-86(1992) (saturation pressure).

## Overview

The vapor is a first-class phase: it is evaluated like the liquids and the SeaFreeze ices, returns the same property set in the same units, and its Gibbs energy can be compared with theirs directly. Equating it with the Gibbs energy of ice Ih gives the sublimation pressure; equating it with a liquid model gives that model's saturation pressure, also for supercooled water.

It is one phase, independent of the liquid model it is paired with. Nothing is fitted or anchored to a vapor-pressure curve.

## Formulation

IAPWS-95 is a Helmholtz function `phi0(tau, delta) + phir(tau, delta)` with `tau = T_c / T` and `delta = rho / rho_c`. At low density the residual part is a power series

```
phir = b1(tau) delta + b2(tau) delta^2 + O(delta^3)
```

whose coefficients are the second and third virial coefficients, `B = b1 / rho_c` and `C = 2 b2 / rho_c^2`. They follow in closed form from the IAPWS-95 coefficients (terms 1-5, 8-12 and 23-26; see `params.py`). Reverting the series from density to pressure gives a Gibbs function that is explicit in (T, p):

```
g / (R T) = 1 + phi0(tau, q) + b1 q + (b2 - b1^2 / 2) q^2

q = p / (rho_c R T)
```

where `q` is the reduced density an ideal gas would have at (T, p). There is no density iteration, and every property is an analytic derivative of this one function. With

```
z = 1 + b1 q + 2 k q^2        k = b2 - b1^2 / 2
e = 1 - 2 k q^2
a = 1 - (tau b1') q - 2 (tau k' + k) q^2
```

the properties are

```
V     = z R T / p              (z is the compressibility factor)
Kt    = p z / e
alpha = a / (T z)
Cp    = -R tau^2 (d^2/dtau^2)(g / R T)
Cv    = Cp - R a^2 / e
Ks    = Kt Cp / Cv,   vel = sqrt(Ks V)
Kp    = 2 z / e^2 - 1
```

The ideal-gas part carries the IAPWS-95 reference state (zero internal energy and entropy of the saturated liquid at the triple point). The liquid models of this package are aligned to the same state at 273.15 K and 0.1 MPa, and the SeaFreeze ice Ih (Feistel & Wagner 2006) is on it by construction.

### Why the series stops at the third virial coefficient

Between the triple point and 100 °C the terms of IAPWS-95 beyond `delta^2` change the Gibbs energy of saturated vapor by at most 1.4e-5 R T (see the table below), so they do not matter for phase equilibria.

They are also not usable outside the stable vapor. IAPWS-95 has a term in `delta^3 tau^50` (term 51) that is negligible on the saturation curve but grows steeply in supersaturated vapor at low temperature: at 229 K and 1 kPa, 126 times the sublimation pressure, it alone contributes -0.51 to `phir`, where the second virial term contributes -0.0034. A phase-equilibrium solver evaluates the vapor at such states on its way to a solution, and the truncated form stays well behaved there.

## Valid Range

- **Temperature:** 130 K (the limit of the IAPWS-95 ideal-gas part) to 400 K
- **Pressure:** above zero, up to the saturation or sublimation pressure and into moderately supersaturated vapor (0.25 MPa at 400 K)

Outside the suggested range the functions still evaluate, and `getProp` warns. Points where the truncated series has no meaning (`V <= 0` or `dV/dp >= 0`, far beyond saturation), and non-positive T or p, return NaN.

## Accuracy

### Vapor pressures from Gibbs-energy equality

`vapor_pressure(T, phase)` solves `G_vapor(T, p) = G_phase(T, p)`.

**Ice Ih (SeaFreeze)** against the IAPWS R14-08 sublimation pressure:

| T / K | IAPWS R14-08 / Pa | from G equality / Pa | deviation |
|------:|------------------:|---------------------:|----------:|
| 130.00 | 1.20017e-08 | 1.20038e-08 | +0.017 % |
| 150.00 | 6.09572e-06 | 6.09568e-06 | -0.001 % |
| 170.00 | 7.30159e-04 | 7.30073e-04 | -0.012 % |
| 190.00 | 3.23554e-02 | 3.23522e-02 | -0.010 % |
| 210.00 | 7.01727e-01 | 7.01725e-01 | -0.000 % |
| 230.00 | 8.94735 | 8.94794 | +0.007 % |
| 250.00 | 76.0127 | 76.0161 | +0.004 % |
| 270.00 | 470.062 | 470.055 | -0.002 % |
| 273.16 | 611.657 | 611.649 | -0.001 % |

The largest deviation is 0.010 % between 190 K and 273.16 K and 0.017 % down to 130 K.

**Liquid models** against the saturation pressure of IAPWS-95 (deviation in percent):

| T / K | p_sat / Pa | `IAPWS95` | `water1` | `holten2014` | `caupin2019` | `caupin2019_kim` | `duska2020` | `grenke2025` |
|------:|-----------:|----------:|---------:|-------------:|-------------:|-----------------:|------------:|-------------:|
| 273.16 | 611.65 | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 |
| 280.00 | 991.82 | -0.001 | -0.002 | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 |
| 290.00 | 1920.00 | -0.001 | -0.003 | -0.001 | -0.001 | -0.001 | -0.002 | +0.001 |
| 300.00 | 3536.81 | -0.001 | -0.003 | -0.001 | -0.000 | -0.000 | -0.002 | +0.004 |
| 320.00 | 10545.98 | -0.002 | -0.003 | -0.002 | -0.001 | +0.002 | -0.002 | +0.013 |
| 340.00 | 27187.81 | -0.001 | -0.002 | -0.002 | -0.008 | +0.018 | -0.002 | +0.028 |
| 360.00 | 62193.57 | -0.002 | -0.002 | +0.002 | -0.026 | +0.089 | -0.001 | +0.053 |
| 373.15 | 101418.00 | -0.002 | -0.003 | +0.008 | -0.044 | +0.199 | -0.002 | +0.078 |

The deviations above 300 K for `holten2014`, `caupin2019`, `caupin2019_kim` and `grenke2025` are outside the range those liquids were fitted to; they measure the liquid, not the vapor. The transport models `singh2017*` inherit the Gibbs energy of their thermodynamic backbone.

`shi_tanaka2020` is not on the IAPWS-95 reference state (its alignment offsets are placeholders), so its Gibbs energy cannot be equated with the vapor's, or with an ice's.

**Supercooled water** (`holten2014`) against Murphy & Koop (2005): 286.44, 125.46, 50.88 and 18.88 Pa at 263.15, 253.15, 243.15 and 233.15 K, against 286.6, 125.6, 51.0 and 18.9 Pa.

### Properties against the full IAPWS-95 formulation

Saturated vapor (the least favorable state; the deviations fall with the third power of the pressure below saturation):

| T / K | p_sat / MPa | Z | G / (R T) | V | S / R | Cp | Cv | vel | ideal gas: G / (R T) | ideal gas: V |
|------:|------------:|--:|----------:|--:|------:|---:|---:|----:|---------------------:|-------------:|
| 273.16 | 0.000612 | 0.99942 | +1.0e-05 | +3.1e-05 | +5.3e-04 | -0.69 % | -0.81 % | +0.07 % | +5.6e-04 | +5.8e-04 |
| 300.00 | 0.003537 | 0.99824 | +1.4e-05 | +4.2e-05 | +7.2e-04 | -0.92 % | -1.1 % | +0.09 % | +1.7e-03 | +1.8e-03 |
| 323.15 | 0.012352 | 0.99609 | +1.2e-05 | +3.5e-05 | +6.0e-04 | -0.76 % | -0.89 % | +0.07 % | +3.9e-03 | +3.9e-03 |
| 348.15 | 0.038595 | 0.99179 | +8.5e-06 | +2.8e-05 | +4.0e-04 | -0.48 % | -0.56 % | +0.05 % | +8.1e-03 | +8.3e-03 |
| 373.15 | 0.101418 | 0.98451 | +1.4e-05 | +5.3e-05 | +4.7e-04 | -0.49 % | -0.53 % | +0.03 % | +1.5e-02 | +1.6e-02 |
| 400.00 | 0.245769 | 0.97218 | +4.8e-05 | +2.0e-04 | +1.4e-03 | -1.2 % | -1.2 % | +0.07 % | +2.7e-02 | +2.9e-02 |
| 450.00 | 0.932204 | 0.93279 | +3.7e-04 | +1.5e-03 | +8.4e-03 | -5.1 % | -5.0 % | +0.29 % | +6.3e-02 | +7.2e-02 |

G, V and S columns are absolute deviations in the stated units (V relative). At 100 °C the vapor is within 1.4e-5 R T in G and 5.3e-5 in V of IAPWS-95, where an ideal gas is off by 1.5 % and 1.6 %. At half the saturation pressure the deviations at 273.16, 300 and 373.15 K are below 2e-6 R T in G, 6e-6 in V and 0.12 % in Cp.

## Usage

```python
import numpy as np
from watereos import getProp, compute

# Like any other model
out = compute(T_K=[250.0, 273.16, 300.0], P_MPa=[1e-5, 6e-4], model='vapor')
out.G          # shape (2, 3), J/kg

# Like an ice, by phase name (P first, as in SeaFreeze)
from watereos._seafreeze_native import getProp_scatter
P = np.array([1e-4, 6.11657e-4])
T = np.array([250.0, 273.16])
vap = getProp_scatter('vapor', P, T)
ice = getProp_scatter('Ih', P, T)
vap.G - ice.G  # J/kg, same reference state

# Batched evaluation without the PT packing (T first, as in the liquids'
# compute_batch): dict with G, V, rho, S, H, U, A, Cp, Cv, Kt, Ks, Kp,
# alpha, vel
from vapor_eos import compute_batch, compute_grid
out = compute_batch(T, P)        # paired points
out = compute_grid(T, P)         # every P with every T, shape (len(P), len(T))

# Vapor pressure implied by the equations of state
from vapor_eos import vapor_pressure, sublimation_pressure, saturation_pressure
vapor_pressure(253.15, 'Ih')          # ice Ih:            1.0324e-4 MPa
vapor_pressure(253.15, 'holten2014')  # supercooled water: 1.2546e-4 MPa
sublimation_pressure(253.15)          # IAPWS R14-08 correlation
saturation_pressure(300.0)            # Wagner & Pruss (1993) correlation
```

## Backends and Cost

`watereos_rs.compute_batch_vapor` and `compute_grid_vapor` (Rust) are used when the extension is available; `vapor_eos.core` (NumPy) is the fallback and the reference the Rust port is tested against. The two agree to rounding.

The evaluation is closed-form, so a call is cheap at any size. Measured on an Apple Silicon workstation with four worker threads (`RAYON_NUM_THREADS=4`), per call of `compute_batch`:

| points | Rust | NumPy |
|-------:|-----:|------:|
| 1 | 1.4 µs | 45 µs |
| 16 | 2.2 µs | 47 µs |
| 256 | 14 µs | 63 µs |
| 1 024 | 52 µs | 114 µs |
| 100 000 | 1.4 ms | 6.9 ms |
| 1 000 000 | 13 ms | 69 ms |

A 1000 x 1000 grid through `compute_grid` takes 11 ms. Calls below 8192 points stay on the calling thread, so small calls never pay for waking the thread pool; larger ones are split across it. `compute_grid` evaluates the temperature-only terms once per temperature.

## Files

| File | Contents |
|------|----------|
| `params.py` | IAPWS-95 coefficients, the two vapor-pressure correlations, suggested range |
| `core.py` | NumPy implementation and the derivation |
| `vapor_eos.py` | `getProp`, backend dispatch, `vapor_pressure` |
| `../src/vapor.rs` | Rust implementation |
