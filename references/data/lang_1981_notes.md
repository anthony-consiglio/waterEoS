# Notes: lang_1981_T1.csv

## Source

E. W. Lang and H.-D. Lüdemann, "High Pressure O-17 Longitudinal Relaxation Time
Studies in Supercooled H2O and D2O", Ber. Bunsenges. Phys. Chem. 85, 603-611 (1981).
DOI: 10.1002/bbpc.19810850716. PDF: `references/lang_1981.pdf` (9 PDF pages; the
paper occupies PDF pages 1-8 = journal pages 603-611; PDF page 9 is the start of an
unrelated paper).

Measurements: 17O spin-lattice relaxation times T1 at 13.56 MHz (Varian XL-100-15 FT
NMR), water-in-cycloalkane emulsions, water enriched to 25% in 17O.

## Data extracted

**Table 1** (journal p. 605, PDF page 3): "Spin-lattice relaxation times T1 (ms) of
the oxygen-17 nucleus in H2-17O."

- 17 temperatures (K): 457, 423, 403, 383, 353, 323, 309, 299, 283, 273, 268, 263,
  258, 253, 248, 243, 238
- 6 measured pressures (MPa): 5, 50, 100, 150, 200, 250
- **Total: 102 points** written to `lang_1981_T1.csv` (columns `T_K,P_MPa,T1_s`),
  ordered as printed (T descending, P ascending within each T).
- T range: 238-457 K. P range: 5-250 MPa.
- No missing or illegible cells among the measured columns; no NaN entries.

Table 2 (same page) is D2-17O — deliberately NOT extracted (D2O out of scope).

## Original units and conversions

| Quantity | Printed unit | CSV unit | Conversion |
|---|---|---|---|
| T | K | K | none |
| p | MPa | MPa | none |
| T1 | ms (stated in Table 1 title) | s | exact decimal shift /1000 |

All printed significant figures kept, including trailing zeros (e.g. printed
"29.70" ms -> `0.02970` s; "1.00" ms -> `0.00100` s; "2.50" ms -> `0.00250` s).
Integer-precision entries appear only in the two hottest isotherm rows and the 403 K
row (e.g. "59" ms -> `0.059` s, "36" ms -> `0.036` s).

## Columns of Table 1 deliberately excluded from the CSV

Table 1 has four additional columns at p = 0.1 MPa, none of which are measurements
from this work:

1. "0.1 (extrapolated)^a" — footnote a: "Data obtained by extrapolation of the
   isotherms measured from 5 MPa to 0.1 MPa." (paper's own extrapolation, not
   measured; excluded to keep the CSV strictly experimental). For reference, the
   printed extrapolated values (ms) at T = 457...238 K are: 59, 45, 36, 30.5, 21,
   12.5, 9.2, 7.4, 4.6, 3.4, 2.4, 1.93, 1.55, 1.23, 0.83, 0.57, 0.40.
2. "0.1 (Ref. [12])^b", "0.1 (Ref. [11])^b", "0.1 (Ref. [13])^b" — footnote b:
   "Data calculated with the resp. fit-equations published by Hindman et al.
   (Ref. [11-13])." (literature comparison values, not this paper's data; excluded).

## Stated experimental uncertainty

**The paper states no T1-specific percentage uncertainty anywhere** (verified by
full-text search of all pages for %, error, accuracy, reproducibility, precision,
uncertainty, ±). What it does state:

- Experimental section (p. 603): "They were measured by a precision Bourdon gauge
  (Heise, Newton, CT, USA) to ±0.5 MPa and generated with standard (1/8)" equipment
  (HIP, Erie PA, USA). The temperatures were determined to ±0.5 K by a
  chromel-alumel thermocouple."
- The only experimental-error percentage in the paper (p. 609) refers to the
  correlation-time preexponential factor tau_s of the Speedy-Angell fit (Eq. 2),
  NOT directly to T1: "The absence of an isotope effect in tau_s larger than the
  limits of experimental error (<=20%) is then readily explained."

If a T1 error bar is needed, it must be taken from a companion publication (e.g.
Lang & Lüdemann 1977/1980 on 1H/2H T1) or assumed; this paper does not give one.

## Quadrupole coupling constant (for tau_r conversion)

Given in the paper. Eq. (1), p. 604 (fast-motional limit, I = 5/2):

  1/T1 = (3/125) * (e^2 qQ / hbar)^2 * (1 + eta_Q^2 / 3) * tau_theta

Adopted values, p. 606: "We therefore choose C_17O-QC = 6.6 ± 0.1 MHz and
eta_Q = 0.93 ± 0.01 observed in ice Ih as temperature- and pressure independent
parameters for light and heavy water to calculate the orientational correlation
times tau_theta via Eq. (1) from the experimental 17O-T1."

Table 3 (p. 605) lists for supercooled liquid H2O (and D2O): C_17O-QC = 6600 ± 100
(kHz; units inferred from the text's 6.6 MHz), eta = 0.93 ± 0.01, chosen per its
footnote as the mean of the experimental results in D2-17O ice Ih and H2-17O ice Ih.
So e^2qQ/h = 6.6 MHz with asymmetry parameter included via the (1 + eta^2/3) factor;
omega_Q itself is not quoted as a single number.

## T1 data in figures

- Fig. 2 (p. 604) plots the H2-17O T1 isotherms; the Results text states: "Figs. 2
  and 3 contain the spin-lattice relaxation times T1 of the 17O-nucleus between
  457 K and 238 K and pressures up to 250 MPa in H2-17O and D2-17O. The data are
  also compiled in Tables 1 and 2." Inspection of Fig. 2 at high zoom shows each
  isotherm carries exactly the six tabulated pressure points (5, 50, 100, 150, 200,
  250 MPa) — **no H2O T1 data exist only in figures; nothing was digitized.**
- Fig. 6 shows only T1(H2O)/T1(D2O) ratios; Figs. 4, 5, 8 show derived correlation
  times tau_theta, not T1.

## Discrepancy noted (paper-internal, does not affect CSV)

The 4th isotherm label in Fig. 2 reads "393K", while Table 1 prints 383 (and the
D2O figure/table both use 383 K). The Hindman fit-equation comparison values printed
in that Table 1 row (29.2, 28.7 ms) are consistent with 383 K, not 393 K, so the
figure label appears to be a typo. The CSV follows the table: 383 K.

## Transcription incident log

- Cell T = 238 K, p = 50 MPa: initially read as "0.46" ms from a low-resolution
  full-page render; the embedded OCR text layer gave "0.66". Resolved by rendering
  the cell at 10x zoom: the printed value is unambiguously **0.66 ms** (also the
  physically smooth value along the isotherm). CSV contains 0.00066 s.

## Verification performed

1. Table transcribed independently three ways: (a) visual read of full-page render,
   (b) PyMuPDF text-layer word extraction with x-coordinate column assignment,
   (c) cell-by-cell visual read of three 8x-zoom strips covering all 17 rows.
   All 102 measured values agree across methods (after resolving the 0.66 incident).
2. Post-write check: PDF page 3 re-read after writing; CSV re-read and confirmed to
   hold exactly 102 data rows; all 102 rows compared programmatically against an
   independent text-layer extraction (0 mismatches, trailing-zero fidelity included);
   16 explicit spot checks spanning every temperature row and all pressure columns —
   16/16 passed. No discrepancies remain.
