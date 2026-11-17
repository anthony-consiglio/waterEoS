# Pure-water PC-SAFT

The two Novak papers supply three pure-water parameter sets for two-state
PC-SAFT. The package also includes the single-state PC-SAFT reference set
reported in both papers. Parameters were transcribed from the published tables
without refitting. The binary-mixture fits in the 2025 paper require component
and cross-association models beyond this pure-water API.

## Parameters

For each pair, the first value is the association energy divided by Boltzmann's
constant in K, and the second is the dimensionless association volume.
High-density water uses the 4C scheme with two donor and two acceptor sites.
Low-density water uses the 2B scheme with one donor and one acceptor site.

| Key | Source | Segment count | Diameter (Å) | Segment energy (K) | 4C association | 2B association | 2B–4C association |
|---|---|---:|---:|---:|---|---|---|
| `pcsaft_water` | 2024 Table II, Diamantonis–Economou reference | 2.1945 | 2.229 | 141.66 | 1804.2, 0.2039 | — | — |
| `novak2024_pcsaft_ts_tv` | 2024 Table II | 2.1145 | 2.266 | 122.33 | 2373.7, 0.0910 | 5723.2, 0.00009538 | Combining rules |
| `novak2024_pcsaft_ts_t` | 2025 Table 1, printed precision | 2.2235 | 2.207 | 130.81 | 1811.2, 0.2324 | 4045.9, 0.1030 | Combining rules |
| `novak2025_pcsaft_ts_caf` | 2025 Table 3 | 2.3147 | 2.182 | 143.03 | 1746.0, 0.2388 | 3736.7, 0.06212 | 2453.6, 0.4560 |

The 2024 fits take the arithmetic mean of the association energies and the
geometric mean of the association volumes. The temperature and volume fit
uses (ΔH/R, ΔS/R, ΔP/R) = (−928.9 K, −5.79352, 2.7836 × 10⁷ mol K/m³).
The temperature-only fits use (−2615.2 K, −10.769, 0) for TS-T and
(−2967.0 K, −12.037, 0) for TS-CAF.

Eqs. 11–12 of the 2024 paper give the low-density fraction as
`1 / (1 + exp((ΔH/R − Vₘ ΔP/R) / T − ΔS/R))` with molar volume
`Vₘ` in m³/mol. The printed Eq. 13 reverses the sign in the exponential.
Eqs. 11–12 reproduce the 2025 Fig. 12 curves, including the increase in the
low-density fraction on cooling. The exported `x` is this fraction of
2B-type molecules. It characterizes local association within a single fluid.

## Evaluation

Use `watereos.compute(T_K, P_MPa, model)` for density, specific volume, heat
capacities, bulk moduli, thermal expansivity, sound speed, entropy, and
energies. The two-state fits also return `x`. The association equations
include temperature and volume derivatives of `x` where applicable.
The ideal-gas contribution uses the IAPWS-95 water function, while the
published residual heat-capacity fits used DIPPR ideal-gas heat capacities
to construct the target data. Energies retain the resulting PC-SAFT model
reference. Direct evaluation at a known mass density is available as
`pcsaft_eos.pressure(T_K, rho, model)`.

The benchmark isobars cover 273.16–372.76 K at 0.1 MPa and 273.16–703.16 K
at 100, 200, and 400 MPa. The registry uses their rectangular envelope.
At fixed T and P the solver selects the densest mechanically stable root
in the packing-fraction interval 0.12–0.68. An unavailable root yields
NaN. Extrapolations into stretched or deeply supercooled water require
caution. The package interface provides the dense branch and direct
pressure evaluation. Saturation and vapor–liquid flash calculations
are outside this implementation.

## Checks against the published curves

These are figure readings at the precision shown, compared with values
computed using the printed parameters.

| Check | Paper (approximately) | Computed |
|---|---:|---:|
| TS-TV low-density fraction at 300 K and 0.1 MPa, 2025 Fig. 12 | 0.26 | 0.266 |
| TS-CAF low-density fraction at 300 K and 0.1 MPa, 2025 Fig. 12 | 0.105 | 0.105 |
| TS-CAF low-density fraction at 250 K and 0.1 MPa, 2025 Fig. 12 | 0.46 | 0.458 |
| TS-TV / TS-T heat capacity at 273.16 K and 0.1 MPa, 2025 Fig. 10 | 5.8 / 11.3 kJ/(kg·K) | 5.75 / 11.34 kJ/(kg·K) |
| TS-CAF / PC-SAFT heat capacity at 273.16 K and 0.1 MPa, 2025 Fig. 10 | 3.8 / 3.3 kJ/(kg·K) | 3.80 / 3.27 kJ/(kg·K) |
| TS-TV sound speed at 350 K and 0.1 MPa, 2024 Fig. 3 | 1555 m/s | 1554.1 m/s |
| Density at 600 K and 400 MPa, 2025 Fig. 8, all four curves | Around 52.5 mol/L | 52.19–52.87 mol/L |
| Ambient-pressure density maximum for two-state fits | Above 276.15 K, fitted constraint | 276.32 K (TS-TV), 276.17 K (TS-T), 276.16 K (TS-CAF) |

The conventional PC-SAFT reference gives 1028.6 kg/m³ at 273.16 K and
0.1 MPa, about 2.9% above the experimental liquid density. It has no
density maximum in the tested ambient-pressure liquid range. The
temperature-only TS-T fit overpredicts cold-water heat capacity, as
reported in both papers.

[Novak et al. (2024)](https://doi.org/10.1063/5.0186752) ·
[Novak et al. (2025)](https://doi.org/10.1016/j.fluid.2025.114475) ·
[Gross and Sadowski (2001)](https://doi.org/10.1021/ie0003887)
