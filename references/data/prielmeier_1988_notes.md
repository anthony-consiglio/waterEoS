# Transcription notes: prielmeier_1988_D.csv

**Source:** F. X. Prielmeier, E. W. Lang, R. J. Speedy, H.-D. Lüdemann,
"The Pressure Dependence of Self Diffusion in Supercooled Light and Heavy Water",
Ber. Bunsenges. Phys. Chem. 92, 1111-1117 (1988).
PDF: `references/prielmeier_1988.pdf`

## What was transcribed

- **Table 1** (p. 1114): "Self diffusion coefficient D (10^-10 m^2 s^-1) in H2O".
  This is the only H2O data table in the paper. All 110 printed cells were transcribed.
- **Table 2** (D2O) was deliberately ignored, as were Tables 3-6 (VTF / power-law fit
  parameters, not raw data).

## Original units and conversions

| Quantity | As printed | CSV column | Conversion |
|---|---|---|---|
| Temperature | K (e.g. 273.0, 208.5) | `T_K` | none |
| Pressure | MPa (columns 0.1, 50, 100, 150, 200, 250, 300, 350, 400) | `P_MPa` | none |
| Diffusion coefficient | units of 10^-10 m^2 s^-1 | `D_m2s` | multiplied by 1e-10 (exact decimal shift) |

CSV values are written with 4-significant-figure mantissas (e.g. printed `10.9` ->
`1.090e-9`). The trailing digits come from the exact decimal shift only; **printed
precision is 3 significant figures** (2 s.f. for the smallest entry, `0.075`). No
rounding, interpolation, or added precision.

## Point count and ranges

- **Total H2O points: 110**
- **T range: 203.5 - 273.0 K** (16 isotherms: 273.0, 268.0, 263.0, 258.0, 255.0,
  252.0, 248.0, 243.0, 238.0, 233.0, 228.0, 223.0, 218.0, 212.0, 208.5, 203.5)
- **P range: 0.1 - 400 MPa** (9 isobars)
- The table is triangular: low-T isotherms exist only at high P (homogeneous
  nucleation limit). Points per isobar: 0.1 MPa: 6; 50 MPa: 8; 100 MPa: 9;
  150 MPa: 11; 200 MPa: 13; 250 MPa: 15; 300, 350, 400 MPa: 16 each.
  First printed pressure per isotherm: 273.0-252.0 K start at 0.1 MPa; 248.0 and
  243.0 K at 50 MPa; 238.0 K at 100 MPa; 233.0 and 228.0 K at 150 MPa; 223.0 and
  218.0 K at 200 MPa; 212.0 and 208.5 K at 250 MPa; 203.5 K at 300 MPa.

## Stated experimental uncertainty (H2O)

From Section 2 (Measurements), p. 1112:

> "The self diffusion coefficients for H2O are judged reliable to ±3%. Their
> reproducability was ±1-2% except for the temperatures below 210 K where the strong
> temperature dependence together with the very short T2 increases the maximal error
> to ±10%."

("reproducability" is the paper's spelling.) So: **±3% overall accuracy, up to ±10%
maximum error below 210 K** (i.e. the 208.5 K and 203.5 K isotherms).

## Data provenance caveats (from Section 3, p. 1113)

- "Data for T > 273 K were taken from the Refs. [16,17]" — the higher-temperature
  isotherms shown in Fig. 2 (277-363 K) are literature values, appear **only in the
  figure**, and were NOT digitized (per instructions, no figure digitization).
- "The ambient pressure data points of the two lowest complete isotherms of Fig. 2
  (open triangles) were thus taken from Gillens data and multiplied by a factor 1.07."
  The two lowest complete isotherms are 255.0 and 252.0 K, so the **0.1 MPa entries at
  255.0 K (4.94) and 252.0 K (4.29)** are corrected Gillen et al. (1972) values, not
  original Prielmeier measurements. They are printed in Table 1 and are included in
  the CSV as printed.
- Fig. 4 implies a 0.1 MPa reference value at 243 K exists (used for the reduced
  isotherm), but no such value is printed in Table 1; it was not transcribed.

## Ambiguities / discrepancies

- **None illegible.** All 110 cells were read cleanly at 6x magnification; the
  embedded OCR text layer, coordinate-based column mapping, and two independent
  visual reads all agree. No NaN entries were needed.
- **Fig. 2 vs Table 1 label:** Fig. 2 labels one isotherm "212.5 K" but Table 1
  prints the row as "212.0". The CSV follows the printed table (212.0 K). Keep this
  internal inconsistency of the paper in mind when fitting (a 0.5 K ambiguity on that
  one isotherm).

## Verification performed

1. Values extracted from the PDF text layer with word x-coordinates; each value
   assigned to the nearest pressure-column center (all assignments within ~7 pt of
   column centers spaced ~22-25 pt apart, so unambiguous), with assertions on row
   counts, duplicate cells, and blank cells.
2. Table 1 re-rendered at 6x resolution and re-read visually; all 16 rows compared
   cell-by-cell against the CSV — full match, including the partial-row alignments
   (e.g. 228.0 K starting at 150 MPa, 218.0 K starting at 200 MPa).
3. Programmatic second pass: row count (110) and per-isobar counts verified; 19
   spot-check cells spanning all 9 isobars and both temperature extremes re-typed
   independently from the high-resolution image matched the CSV exactly; 10
   blank-cell absence checks passed.
4. Sanity check: D(273.0 K, 0.1 MPa) = 1.090e-9 m^2/s agrees with the accepted
   ambient-pressure value near 0 C (~1.1e-9 m^2/s).
