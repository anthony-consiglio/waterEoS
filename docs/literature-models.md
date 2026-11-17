# Literature models

Use the keys below with `watereos.compute(T_K, P_MPa, model)` or
`watereos.getProp(PT, model)`. The registry lists each model's properties and
suggested plotting bounds. Rectangular bounds can include extrapolated states.
All output pressures and bulk moduli are in MPa. Compressibility in MPa⁻¹ is
the reciprocal of `Kt`.

## Moynihan (1996)

`moynihan1996species` implements the H₂O regular-solution model with
ΔV° = −4.5 cm³/mol, ΔH° = 1940 J/mol, ΔS° = 0.98 R (about 8.1 J/(mol K)),
and W = 3610 J/mol. It returns the low-density species fraction `x` and the
excess responses from Eqs. 17–19. `Cp_ex` is in J/(kg·K), `V_alpha_ex` in
m³/(kg·K), and `V_kappa_ex` in m³/(kg·MPa). Approximate `V`, `rho`, `alpha`,
and `Kt` use the constant background response coefficients discussed on
p. 421 and the critical specific volume of 1.01 cm³/g from Fig. 7.
The excess responses are independent of this volume reconstruction.
The model leaves the caloric background unspecified.

The model supports `branch='stable'`, `'hdl'`, and `'ldl'`. The
`moynihan_eos` package exposes `critical_point()`,
`coexistence_pressure(T_K)`, `spinodal_pressures(T_K)`, and
`coexistence_volumes(T_K)`. The named liquid branches represent extensions
of the regular-solution model toward the amorphous-ice region. The fit
uses supercooled-liquid response data around 236–277 K at pressures up to
100 MPa and incorporates the 135 K transformation pressures as stability
limits.

| Check | Paper | This implementation |
|---|---:|---:|
| Critical temperature | 217 K | 217.092 K |
| Critical pressure | 380 atm | 376.25 atm |
| Coexistence pressure at 135 K | 1850 atm | 1843.25 atm |
| HDL stability limit at 135 K | 500 atm | 503.83 atm |
| LDL stability limit at 135 K | 3200 atm | 3182.67 atm |

The rounded parameters give the small pressure differences above. The
reconstructed Fig. 7 coexistence volumes at 135 K are 0.838 and 1.061 cm³/g.
The 1 atm excess heat-capacity difference relative to 277 K peaks at
37.94 J/(mol·K) and 230.9 K, close to Fig. 5's approximately 37.5 J/(mol·K)
and 231.5 K. [Moynihan (1996)](https://doi.org/10.1557/PROC-455-411)

## Mishima & Sumita (2023)

`mishima2023equation` uses the case 3, 7 × 7 polynomial from Table S1.
This is the fit incorporating liquid, LDA, and HDA volumetric data. It
returns `V`, `rho`, `alpha`, and `Kt`. At a given temperature and pressure
the implementation finds mechanically stable roots in 0.70–1.25 cm³/g.
Where there are several, it selects the root with the lowest relative
Gibbs energy along that isotherm. `mishima_eos.volume_roots(T_K, P_MPa)`
exposes the stable roots in cm³/g. An unavailable root is represented
by NaN.

The approximate plotting envelope is 135–500 K and −100–1000 MPa.
Actual fit points cover a narrower and irregular region, including
amorphous-ice volumes near 135–152 K. The polynomial can oscillate under
extrapolation and misses the measured compressibility near 220 K and
atmospheric pressure. At 135 K and 1 atm it has only a dense stable root
near 0.873 cm³/g. The low-density branch reappears at 150 K and 1 atm,
with a root near 1.071 cm³/g. These are roots of the model polynomial.

| Check | Paper | This implementation |
|---|---:|---:|
| Table 1 critical point | 204 K, 114 MPa, 0.982 cm³/g | 203.704 K, 113.706 MPa, 0.982486 cm³/g |
| Table 2 HDL spinodal saddle | 194 K, 122 MPa, 0.951 cm³/g | 193.489 K, 122.026 MPa, 0.951067 cm³/g |

The saddle temperature differs from the printed integer by about 0.51 K.
[Mishima & Sumita (2023)](https://doi.org/10.1021/acs.jpcb.2c08342)

## Fits to simulations of model waters

Each fit describes the named potential. Use these models to compare
simulation-fitted equations, with energy and entropy zeros specific to the
source. The Gibbs fits generally return mixture properties, per-structure outputs,
and the low-density fraction. The Yu fit is limited to volumetric responses
and dynamics because the source leaves its caloric background unspecified. `Kp` and its per-structure counterparts are
NaN placeholders on the full two-state batch paths. Yu also returns the structural fraction `x` and
the slow-cluster fraction `f` used for `D` and `tau_r`. The `hdl` and `ldl`
branches exist only where allowed by the two-state equilibrium.

| Key | Model water | Fit or plotting range | Published-value check |
|---|---|---|---|
| `singh2016_tip4p2005` | TIP4P/2005 | 180–300 K, roughly −80 to 400 MPa, density ≥960 kg/m³. The validity contour in Fig. 6 is narrower. | 182 K, 170 MPa, 1017 kg/m³ in Table I. At 100 MPa, Fig. 3(b) shows a compressibility maximum near 0.00124 MPa⁻¹ and 205 K. Computed 0.001238 MPa⁻¹ at 204.8 K. |
| `biddle2017_tip4p2005` | TIP4P/2005 | 180–320 K, roughly −275 to 400 MPa, restricted by the liquid–vapor spinodal and the lower-left limit in Fig. 1. | Supplement Fig. S2 gives a liquid–vapor spinodal of −351 MPa at 230 K. The 2018 erratum coefficients give −351.7 MPa. |
| `gartner2020_scan` | SCAN-DNN | 240–330 K, 0.1–300 MPa | Table 1 gives 224 K, 268.7 MPa, 1072 ± 16 kg/m³. Computed critical density 1073.24 kg/m³. |
| `holten2014_st2_i_mf` | ST2(I), mean field | 240–322 K, 100–250 MPa | Tables I and III: 253.5 K, 160 MPa and 52.478 kmol/m³. |
| `holten2014_st2_ii_mf` | ST2(II), mean field | 240–315 K, 100–300 MPa | Tables I and III: 249.0 K, 146 MPa and 53.261 kmol/m³. |
| `holten2014_st2_i_crossover` | ST2(I), crossover | 240–322 K, 100–250 MPa | 252.0 K, 168 MPa and 52.739 kmol/m³ from Table II. At 240 K, coexistence densities in Fig. 5 are about 1086 and 918 kg/m³. |
| `holten2014_st2_ii_crossover` | ST2(II), crossover | 240–315 K, 100–300 MPa | 246.75 K, 155 MPa and 53.645 kmol/m³ from Table IV. On the critical isobar, calculated `Cv` at 246.65 and 246.85 K is 7.443 and 5.995 kJ/(kg·K), versus approximately 7.44 and 6.00 in Fig. 10. |
| `holten2013_mw`, `holten2013_mw_n6` | mW, unclustered and six-site cluster | Plotted simulation envelope 170–280 K, approximately 0.1–1013 MPa | The fits have a single composition minimum. At 220 K and 0.1 MPa, Fig. 2 gives `x` ≈ 0.28 and 0.18. Computed 0.2800 and 0.1803. |
| `holten2013_mw_weak_crystallization` | mW, empirical weak-crystallization fit | Same simulation envelope. Returns NaN for T ≤ Tₛ(P) or Tₛ(P) ≤ 0. | At 220 K and 0.1 MPa, Fig. 7 gives about 998 kg/m³ and 3.2 kJ/(kg·K). Computed 998.33 and 3.201. |
| `weis2022_wail` | WAIL, printed Table III | 215–260 K, 10–80 MPa | At 200 K, Fig. 1A gives coexistence densities around 1048 and 908 kg/m³. Computed 1050.47 and 909.21 kg/m³. |
| `eltareb2022_qtip4pf_h2o_classical` | q-TIP4P/F, classical H₂O | 180–325 K, −50 to 350 MPa | Critical density 1028.41 kg/m³ vs 1030 kg/m³. At 300 K and 0.1 MPa, 1001.61 vs about 1002 kg/m³. |
| `eltareb2022_qtip4pf_h2o_quantum` | q-TIP4P/F, quantum H₂O | 180–325 K, −50 to 350 MPa | Critical density 1018.14 kg/m³ vs 1020 kg/m³. At 300 K and 0.1 MPa, 995.04 vs about 995 kg/m³. |
| `eltareb2022_qtip4pf_d2o_quantum` | q-TIP4P/F, quantum D₂O | 190–325 K, −50 to 350 MPa | Critical density 1132.27 kg/m³ vs 1130 kg/m³. At 300 K and 0.1 MPa, 1107.86 vs about 1110 kg/m³. |
| `yu2023_tip4p2005` | TIP4P/2005, hierarchical two-state | About 170–350 K, 0.1–180 MPa | Table 1 gives 175.9 K and 177.2 MPa. At 200 K, Fig. 3 gives log₁₀(τᵣ/ps) about 4.9 and 2.3 at 0.1 and 180 MPa. Computed 4.945 and 2.306. |
| `zorzi2024_mlbop` | ML-BOP | 100–270 K, 0.1–700 MPa, with low-temperature data primarily on the dense side | At 100 K and 700 MPa, Figs. 2 and 4 give about 1256 kg/m³ and −38.05 kJ/mol. Computed 1257.8 kg/m³ and −38.039 kJ/mol. |
| `handle2018_tip4p2005` | TIP4P/2005, Gaussian landscape | 200–270 K, 0.90–1.42 g/cm³, approximately −142 to 2411 MPa over that envelope | At 250 K and 1.10 g/cm³, Helmholtz energy −55.9765 kJ/mol vs −55.95 in Table I. Extrapolated critical point 175.469 K, 175.099 MPa and 997.036 kg/m³, within the Fig. 12 estimates. |
| `eltareb2024_qtip4pf_classical` | q-TIP4P/F, classical Gaussian landscape | 200–280 K, 0.94–1.40 g/cm³, approximately −161 to 2579 MPa over that envelope | At 0.88 cm³/g, the Kauzmann temperature is 137.2 K from Eq. 29 vs about 137 K in Fig. 7(b). At 200 K and 1.20 g/cm³, pressure is 448.42 MPa vs about 435 MPa in Fig. 8(a). |


Biddle uses the corrected spinodal coefficients from the 2018 erratum and
reconstructs the unlisted linear pressure coefficient from the stated
critical density. The supplementary density curves check the reconstruction.
Gartner's printed parameters reproduce the critical density and the Fig. 2A
density jump to figure-reading accuracy. At the text's reduced coexistence
temperature of 0.965 they give 1129.4 and 1037.6 kg/m³ versus the quoted 1102
and 1047 kg/m³. The regression against those quoted values is an expected
failure. [Singh et al. (2016)](https://doi.org/10.1063/1.4944986) ·
[Biddle et al. (2017)](https://doi.org/10.1063/1.4973546) ·
[Biddle et al. (2018)](https://doi.org/10.1063/1.5024509) ·
[Gartner et al. (2020)](https://doi.org/10.1073/pnas.2015440117)

### Reproduction limits

- **ST2(I) and mW.** The ST2(I) and unclustered mW fits can give negative
  adiabatic moduli and undefined sound speeds in cold, low-pressure states.
  For the athermal mW fits, the named branches select halves of the
  composition range. Each fit has a single equilibrium composition.
- **WAIL.** The printed Table III fit gives 998.21 kg/m³ at 250 K and
  80 MPa, versus approximately 1079 kg/m³ from Fig. 1A. The printed
  coefficients can give a negative bulk modulus within the nominal fit box.
  The existing source does not provide a verified coefficient correction.
- **q-TIP4P/F two-state.** To recover the critical volumes and density curves,
  the 325 K columns in Tables S1–S3 are read with the background values
  rotated from the printed row labels. The quantum H₂O heat-capacity peak
  at 0.1 MPa is 211.87 K versus 216 K in the text. The D₂O table appears to
  use the H₂O mass basis for the Gibbs normalization. The physical
  D₂O molar conversion gives a peak of 106.9 J/(mol·K), compared with
  around 96 in Fig. S7(b). Use mass-specific units for comparisons.
- **ML-BOP.** The printed Table A1 coefficients give 1022.2 kg/m³ at
  270 K and 0.1 MPa, versus approximately 997 kg/m³ in Fig. 2(a).
  They also give 1040.4 kg/m³ at the critical point and 986.6/1149.7 kg/m³
  at 150 K coexistence, versus approximately 1020 and 974/1132 kg/m³ in
  Fig. 1. These checks remain expected failures.
- **Classical q-TIP4P/F landscape.** Direct substitution of the printed
  polynomials gives 28.91 MPa and −37.2881 kJ/mol at 260 K and 1 cm³/g,
  versus about 0.1 MPa and −37.219 kJ/mol in SI Table S1. It also fails
  to reproduce the Table S3 critical point. The module retains the
  printed polynomials and documents the inferred units.

### Candidates without a selectable implementation

| Catalog key | Reason |
|---|---|
| `neophytou2024potential` | Landscape coefficients are available graphically. |
| `szukalo2026energy` | The local deposit now includes coefficient functions on a 100-point volume grid. Reproducing those tables would require choosing interpolation rules. The paper and supplement do not print a complete analytic parameterization. |
| `verma2025statistical` | The supplement is on disk with all 14 parameters. Ambiguous equations and rounded constants prevent reproduction of the reported critical point with the interpretations checked so far. |
| `eltareb2026nuclear` | The quantum landscape table fails key pressure, critical-point, and free-energy checks. A separate classical parameter table is unavailable. |
