# Notes: arnold_2002_T1.csv

Source: M. R. Arnold and H.-D. Ludemann, "The pressure dependence of self-diffusion
and spin-lattice relaxation in cold and supercooled H2O and D2O",
Phys. Chem. Chem. Phys. 4 (2002) 1581-1586, DOI 10.1039/b110639m.
PDF: `references/arnold_2002.pdf`

## What was extracted

- **Table 4** (p. 1583): "Oxygen-17 spin-lattice relaxation times T1 (ms) of H2(17)O at 67.8 MHz".
  This is the ONLY table of 17O T1 for light water in the paper.
- Ignored per task scope: Table 1 (D of D2O), Table 3 (2H T1 of D2O), Table 5 (17O T1 of D2O),
  Tables 6-8 (dimensionless ratios), all self-diffusion data.

## Original units and conversions

| Quantity | Printed as | CSV column | Conversion |
|---|---|---|---|
| Temperature | T/K (already kelvin) | T_K | none |
| Pressure | p/MPa (already MPa) | P_MPa | none |
| T1 | ms | T1_s | divided by 1000 (decimal shift only; all printed significant figures preserved) |

Significant figures preserved exactly as printed, e.g. `2.1` ms -> `0.0021` s (2 s.f., at
253 K / 350 and 400 MPa), `1.20` ms -> `0.00120` s, `0.70` ms -> `0.00070` s.

## Coverage

- **41 data points** (CSV has 41 data rows + 1 header row).
- T range: **222-273 K** (11 isotherms: 273, 268, 263, 258, 253, 248, 243, 238, 233, 229, 222 K).
- P range: **250-400 MPa** (columns 250, 300, 350, 400 MPa).
- Blank (unmeasured) cells in the printed table -- not transcription problems, simply absent:
  - 233 K at 400 MPa
  - 222 K at 350 and 400 MPa
- No illegible entries; column assignment of partial rows was verified from word x-coordinates
  in the PDF text layer (233 K row occupies the 250/300/350 columns; 222 K row the 250/300 columns).

## Stated experimental uncertainty

- T1 (and D): the paper states these are "judged reliable to +/-5%" (Arnold & Ludemann 2002,
  p. 1582, left column, last sentence before Results and discussion; the sentence covers both the
  self-diffusion coefficients and the spin-lattice relaxation times reported in the paper).
- Temperature: stated reliable to +/-0.5 K (same paragraph; Bruker variable-temperature unit
  checked against the methanol standard and a metal-sheathed thermocouple).
- Pressure: measured with class-0.1 Heise Bourdon gauges to +/-0.6 MPa (same paragraph).

## Quadrupole coupling information in the paper (for tau_r conversion)

The paper does NOT quote a liquid-phase omega_Q or QCC for 17O in H2O, and the authors
explicitly decline to compute rotational correlation times tau_2 from the T1 data, on the
grounds that literature QCC determinations for the liquid are indirect/approximate and any
T,p-dependence of the QCC is smaller than experimental error (p. 1583, left column).
What it does give:

- **Table 2** (p. 1583): QCC compilation for the 17O nucleus in D2O:
  - gas phase: **10.18 MHz** (ref. 39 = Verhoeven, Dynamus, Bluyssen, J. Chem. Phys. 50 (1969) 3330)
  - solid (ice): **6.41 MHz** (ref. 40 = Brosnan & Edmonds, J. Mol. Struct. 58 (1980) 23)
- p. 1583, right column: the 17O QCC of H2(17)O and D2(17)O are identical within data precision
  in both the solid state and the gas phase (refs. 29, 38, 40) -- i.e. the Table 2 D2O values
  can be carried over to H2O.
- Asymmetry parameter (p. 1582, right column): literature values (refs. 39-41) span
  **0.72 < eta_Q < 0.94** for oxygen-17; (for 2H, the eta_Q contribution can be neglected).
- Relaxation equation, eqn (2) (p. 1582): 1/T1 = (3/200) * (2I+3)/(I^2(2I-1)) *
  (e^2 qQ / hbar)^2 * (1 + eta_Q^2/3) * [tau_2/(1+omega^2 tau_2^2) + 4 tau_2/(1+4 omega^2 tau_2^2)],
  with I = 5/2 for 17O; valid in the extreme narrowing limit (omega_0 tau_2)^2 < 1, and the paper
  states (citing ref. 11) that all data discussed are well within that regime
  (extreme narrowing: the bracket -> 5 tau_2).
- Larmor (resonance) frequency, not omega_Q: 17O measured at **67.78 MHz** (Bruker Avance 500;
  p. 1581, Experimental; table header rounds to 67.8 MHz).

Caveat for downstream tau_r = 1/(T1 * omega_Q^2)-type conversions: choose the QCC and eta_Q
convention consistently with eqn (2); the paper itself provides only the gas/ice QCC bounds above.

## Data that exist only in figures (NOT digitized)

- **Fig. 3** (p. 1584): isotherms of 17O T1 of H2(17)O from ~0.1 MPa to 400 MPa. Open symbols
  are older literature data (ref. 14 = Lang & Ludemann, Ber. Bunsen-Ges. Phys. Chem. 85 (1981) 603);
  full symbols are the new data. The text (p. 1583) states that Tables 3-5 compile only the NEW T1
  values -- hence Table 4 starts at 250 MPa, and all lower-pressure H2O 17O T1 points
  (including parts of the 233/229/222 K isotherms shown in Fig. 3) exist only as figure symbols.
  Per task instructions these were not digitized.
- Figs. 5 and 6 contain only reduced/ratio quantities derived from T1, no additional raw T1.

## Sample note

The Experimental section (p. 1581) states the H2(17)O sample was 25% oxygen-17 enriched
(GFK Isotopenstelle, Karlsruhe); the Fig. 3 caption says "enriched with 50% oxygen-17".
This is an internal inconsistency of the paper (its ref. 11 used both 25% and 50% enriched
samples). Enrichment level does not affect the T1 values used here.

## Verification

- Values transcribed from the rendered page image, then independently cross-checked against the
  PDF embedded text layer (pdfplumber), including word x-coordinates to resolve column membership
  of the two incomplete rows. Row count (41) and all values matched on a second pass.
