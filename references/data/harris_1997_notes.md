# Notes on harris_1997_D.csv

Source: K. R. Harris and P. J. Newitt, "Self-Diffusion of Water at Low Temperatures and
High Pressure", J. Chem. Eng. Data 1997, 42 (2), 346-348 (JE9602935).
PDF: `references/harris_1997.pdf` (3 pages).

## Table extracted

- **Table 1, "Self-Diffusion Coefficients of Water", p. 347** — the only numerical D
  table in the paper (Figures 1-4 are plots only, not digitized).
- Table 1 is printed as two side-by-side three-column blocks with columns
  `t/°C`, `p/MPa`, `D/10^-9 m^2 s^-1`; 46 printed rows in total.
- Extraction method: values taken from the PDF's embedded text layer (PyMuPDF span
  extraction, including the small-font subscript glyphs), then cross-verified twice
  against a 5.5-6x-zoom render of the table region and against an independent manual
  transcription. Zero discrepancies; 16 additional spot checks passed.

## Original units and conversions

| Quantity | Printed unit | CSV unit | Conversion |
|---|---|---|---|
| t | °C | K (`T_K`) | T_K = t + 273.15 (exact decimal addition) |
| p | MPa | MPa (`P_MPa`) | none (copied verbatim) |
| D | 10^-9 m^2 s^-1 | m^2 s^-1 (`D_m2s`) | mantissa kept verbatim as printed, suffixed `e-09` |

- The last digit of each subzero-block D value is printed as a **subscript**
  (journal convention for a digit of reduced significance), e.g. 0.56₄, 1.05₀.
  These are transcribed as ordinary final digits: 0.564, 1.050. The trailing zero of
  1.05₀ is retained (`1.050e-09`) to preserve the printed significant figures.
- Above-zero-block values (0, 5, 25 °C isotherms) are printed with three plain digits
  (e.g. 1.13, 2.34) and are transcribed as such.

## Raw vs. smoothed / excluded rows

- The paper tabulates only **raw (measured) values**; there is no smoothed-value table.
  Each tabulated point is itself the average of repeat runs at that state point:
  "At each state point, values obtained for constant magnetic-field gradients over a
  range of rf-pulse separations and at constant rf-pulse separation over a range of
  gradients were averaged."
- **Excluded (2 rows):** the two 0.1 MPa entries flagged with footnote *a*
  ("Atmospheric pressure values from Mills (1973).") are literature reference values,
  not measurements of this work, and were omitted from the CSV to avoid mixing in
  Mills (1973) data:
  - 5.00 °C, 0.1 MPa, D = 1.313 x 10^-9 m^2/s
  - 25.00 °C, 0.1 MPa, D = 2.299 x 10^-9 m^2/s
  Add them back manually if literature anchors at atmospheric pressure are wanted.
- The two ~0 °C atmospheric points (0.06 °C and 0.02 °C at 0.1 MPa, both 1.13) carry
  no footnote and ARE this work's measurements; both are kept.
- Genuine printed repeats kept as printed: -10.00 °C/200.5 MPa and -9.99 °C/200.5 MPa
  (both 0.872), and the two 0.1 MPa points at ~0 °C.

## Stated uncertainty / precision

The paper states **no single overall percentage uncertainty for D**. The relevant
statements (quoted verbatim) are:

1. Gradient-coil calibration (the dominant systematic for spin-echo D):
   > "The gradient coil was calibrated using the reference values for the
   > self-diffusion coefficient of water established at 0.1 MPa by Mills (1973). The
   > coil constant, averaged over 13 points, was (0.5017 ± 0.0015) T/(A m rad): the
   > maximum deviation was 0.8%."
   (±0.0015/0.5017 corresponds to ±0.3% in the coil constant; maximum deviation 0.8%.)
2. Indirect statement of the experimental error, from the comparison with Prielmeier
   et al.:
   > "Good agreement was obtained with the pulsed gradient emulsion results of
   > Prielmeier et al. (1987, 1988) (Figure 4), except above 200 MPa at -10 °C, where
   > their data lie about 7-9% below ours, twice the sum of the estimated
   > experimental errors."
3. State-variable accuracies:
   > "Pressures (accuracy, ±0.5 MPa) were measured with a Heise Bourdon gauge
   > calibrated against a Budenberg 283 dead-weight piston gauge."
   > "Temperatures (accuracy, ±0.02 K) were measured with a calibrated four-lead Pt
   > resistance thermometer (Leeds and Northrup) inserted in the bottom closure of
   > the pressure vessel"
   (Calibration reference fluids in general are described as "accurately known
   (± 0.1-0.2%)"; the ±(3 to 6)% and ±(3 to 7)% figures on p. 346 refer to the older
   Angell et al. (1976) work, not to this paper.)

## Dataset summary

- **Points written: 44** (46 printed rows minus the 2 Mills footnote rows).
- Per-isotherm counts (as written):
  ~-20 °C group (t = -21.50 to -19.89): 4; -14.99 °C: 5; ~-10 °C: 7; ~-5 °C: 6;
  ~0 °C: 10; ~5 °C: 5; ~25 °C: 7.
- **T range:** 251.65 K to 298.22 K (t = -21.50 °C to +25.07 °C).
- **P range:** 0.1 MPa to 350.5 MPa.
- **D range:** 0.564 x 10^-9 to 2.35 x 10^-9 m^2/s.
- Row order in the CSV follows the printed order (left column block top-to-bottom,
  then right column block), i.e. isotherms in ascending temperature.
- **Illegible entries: none.** No NaN values were required.
- The subzero data lie in the liquid region bounded by ice I and ice III (Figure 1);
  per the paper, some points near -21 °C may lie just inside the ice III boundary of
  Henderson and Speedy (1987) / Bridgman (1912). No emulsions were used, and the
  authors did not attempt measurements deep in the supercooled region.
