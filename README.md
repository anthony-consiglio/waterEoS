<p align="center">
  <img src="waterEoS_header.png" alt="waterEoS" width="700">
</p>

<p align="center">
  <a href="https://pypi.org/project/waterEoS/"><img src="https://img.shields.io/pypi/v/waterEoS.svg" alt="PyPI version"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <a href="https://pypi.org/project/waterEoS/"><img src="https://img.shields.io/pypi/pyversions/waterEoS.svg" alt="Python versions"></a>
</p>

## Overview

**waterEoS** provides Python implementations of three two-state equations of state (EOS) for supercooled water, unified under a single [SeaFreeze](https://github.com/Bjorn-Elsrud/SeaFreeze)-compatible API. Each model captures the thermodynamic anomalies of water by treating it as a mixture of two interconvertible local structures (low-density/tetrahedral and high-density/disordered), predicting a liquid-liquid critical point (LLCP) in the deeply supercooled regime.

## Installation

```bash
pip install waterEoS
```

## Quick Start

```python
import numpy as np
from watereos import getProp

# Single point: 0.1 MPa, 300 K
PT = np.array([[0.1], [300.0]], dtype=object)
out = getProp(PT, 'duska2020')
print(f"Density: {out.rho[0,0]:.2f} kg/m³")
print(f"Cp:      {out.Cp[0,0]:.1f} J/(kg·K)")
print(f"x:       {out.x[0,0]:.4f}")
```

## Available Models

| Model key | Reference | LLCP (T, P) | Valid range |
|-----------|-----------|-------------|-------------|
| `'holten2014'` | Holten, Sengers & Anisimov, J. Phys. Chem. Ref. Data **43**, 014101 (2014) | 228.2 K, 0 MPa | 200-360 K, -400-400 MPa |
| `'caupin2019'` | Caupin & Anisimov, J. Chem. Phys. **151**, 034503 (2019) | 218.1 K, 72.0 MPa | 200-360 K, -200-400 MPa |
| `'duska2020'` | Duska, J. Chem. Phys. **152**, 174501 (2020) | 220.9 K, 54.2 MPa | 235-360 K, -100-400 MPa |
| `'water1'` | SeaFreeze water1 (pass-through) | -- | 240-501 K, 0-2300 MPa |
| `'IAPWS95'` | SeaFreeze IAPWS-95 (pass-through) | -- | 240-501 K, 0-2300 MPa |

## Usage

### Grid Mode

Evaluate on a pressure x temperature grid (like SeaFreeze):

```python
import numpy as np
from watereos import getProp

P = np.arange(0.1, 200, 10)    # pressures in MPa
T = np.arange(250, 370, 1)     # temperatures in K
PT = np.array([P, T], dtype=object)

out = getProp(PT, 'holten2014')
# out.rho has shape (len(P), len(T))
```

### Scatter Mode

Evaluate at specific (P, T) pairs:

```python
import numpy as np
from watereos import getProp

PT = np.empty(3, dtype=object)
PT[0] = (0.1, 273.15)    # 0.1 MPa, 273.15 K
PT[1] = (0.1, 298.15)    # 0.1 MPa, 298.15 K
PT[2] = (100.0, 250.0)   # 100 MPa, 250 K

out = getProp(PT, 'caupin2019')
# out.rho has shape (3,)
```

### Individual Model Access

Each model can also be imported directly:

```python
from duska_eos import getProp
from caupin_eos import getProp
from holten_eos import getProp
```

### List Available Models

```python
from watereos import list_models
print(list_models())
# ['water1', 'IAPWS95', 'holten2014', 'caupin2019', 'duska2020']
```

## Output Properties

All three two-state models return an object with the following attributes:

### Mixture (equilibrium) properties

| Attribute | Property | Units |
|-----------|----------|-------|
| `rho` | Density | kg/m³ |
| `V` | Specific volume | m³/kg |
| `Cp` | Isobaric heat capacity | J/(kg·K) |
| `Cv` | Isochoric heat capacity | J/(kg·K) |
| `Kt` | Isothermal bulk modulus | MPa |
| `Ks` | Adiabatic bulk modulus | MPa |
| `Kp` | Pressure derivative of bulk modulus | -- |
| `alpha` | Thermal expansivity | 1/K |
| `vel` | Speed of sound | m/s |
| `S` | Specific entropy | J/(kg·K) |
| `G` | Specific Gibbs energy | J/kg |
| `H` | Specific enthalpy | J/kg |
| `U` | Specific internal energy | J/kg |
| `A` | Specific Helmholtz energy | J/kg |
| `x` | Tetrahedral (LDL) fraction | -- |

### Per-state properties

Each property above (except `x`) is also available for the individual states with `_A` and `_B` suffixes:
- `rho_A`, `Cp_A`, `vel_A`, ... (State A: high-density / disordered)
- `rho_B`, `Cp_B`, `vel_B`, ... (State B: low-density / tetrahedral)

**Total: 43 output properties** (15 mixture + 14 state A + 14 state B).

All thermodynamic potentials (S, G, H, U, A) are aligned to the IAPWS-95 reference state.

## Phase Diagram

Each model provides functions to compute the liquid-liquid phase diagram:

```python
from duska_eos import compute_phase_diagram

result = compute_phase_diagram()
# result contains: T_LLCP, p_LLCP, T_spin_upper, p_spin_upper,
#                  T_spin_lower, p_spin_lower, T_binodal, p_binodal, ...
```

Available functions: `find_LLCP()`, `compute_spinodal_curve()`, `compute_binodal_curve()`, `compute_phase_diagram()`.

## Performance

Throughput on a 100x100 = 10,000-point grid:

| Model | Time | Throughput |
|-------|------|-----------|
| Holten (2014) | 32 ms | 317k pts/s |
| Caupin (2019) | 18 ms | 563k pts/s |
| Duska (2020) | 49 ms | 203k pts/s |

## References

1. V. Holten, J. V. Sengers, and M. A. Anisimov, "Equation of state for supercooled water at pressures up to 400 MPa," *J. Phys. Chem. Ref. Data* **43**, 014101 (2014). [doi:10.1063/1.4895593](https://doi.org/10.1063/1.4895593)

2. F. Caupin and M. A. Anisimov, "Thermodynamics of supercooled and stretched water: Unifying two-structure description and liquid-vapor spinodal," *J. Chem. Phys.* **151**, 034503 (2019). [doi:10.1063/1.5100228](https://doi.org/10.1063/1.5100228)
   - Erratum: *J. Chem. Phys.* **163**, 039902 (2025). [doi:10.1063/5.0239673](https://doi.org/10.1063/5.0239673)

3. M. Duska, "Water above the spinodal," *J. Chem. Phys.* **152**, 174501 (2020). [doi:10.1063/5.0006431](https://doi.org/10.1063/5.0006431)

## Authors

- Anthony Consiglio

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
