# Notes: dehaoui_2015_eta.csv

Source: Dehaoui, Issenmann, Caupin (2015), "Viscosity of deeply supercooled water and its
coupling to molecular diffusion", PNAS 112 (39), 12020-12025, doi:10.1073/pnas.1508996112,
and its Supporting Information (SI, 8 pages).
Local files: `references/dehaoui_2015.pdf`, `references/dehaoui_2015_SI.pdf`.

## 1. Where the eta data came from

- **CSV data = Table S2, "Smoothed values of the viscosity" (SI p. 7 of 8).**
  55 points, T = 239.15 to 293.15 K in exact 1 K steps, eta in mPa s, at atmospheric
  pressure (P_MPa written as 0.1 for all rows; the paper reports ambient-pressure
  measurements only).
- Table S2 is the authors' recommended representation of their own measurement
  ("our data"): SI text states the smoothed data "were used in Fig. 1", and for the
  combined power-law fits "The viscosity data below 250 K are the smoothed values from
  the present work (Table S2)". Table S3 lists "Dehaoui 2015" primary data as 55 points.
- **Table S1** (SI pp. 5-6) contains the underlying raw data: 369 individual measurements
  ("A total of 369 viscosity measurements were collected from four runs with four
  independent capillaries"), tabulated per run (Run 1-4) with 1-11 repeats per
  temperature and NO per-point uncertainties. Not used for the CSV (Table S2 preferred
  per the authors' own usage).
- Significant figures kept exactly as printed: 4 sig figs (e.g., 14.22, 9.354, 1.794);
  the last entry 0.994 is printed with 3 sig figs.

## 2. Uncertainty convention (deta_mPas column) — DERIVED, not printed

No table in the paper or SI lists per-point uncertainties. The 1-SD uncertainty is
specified by formula in the main paper, Materials and Methods, "Measurement Uncertainty"
(p. 12024), quoted verbatim:

> "We take for the intrinsic relative uncertainty (1 SD) on eta the 2.3% SD of the
> measurements on 12 independent capillaries at T0 = 293.15 K. Because the viscosity is
> well described by a power law [eta(T) = eta0(T/Ts - 1)^-gamma], the effect of the
> 0.15 K temperature uncertainty was taken into account to give the total relative
> uncertainty (1 SD) at temperature T using {(0.023)^2 + [0.15 gamma/(T - Ts)]^2}^(1/2).
> The resulting uncertainty ranges from 2.3% at the highest temperature to 2.9% at the
> lowest temperature."

- deta_mPas = eta_mPas * sqrt(0.023^2 + (0.15*gamma/(T - Ts))^2), i.e., **relative 1 SD**
  applied to the printed eta. Rounded to 4 significant figures.
- Parameters used: the power-law fit of the authors' own raw data (SI, "Viscosity
  Values" section, unnumbered equation): **Ts = 224.80 K, gamma = 1.7044** (see Sec. 3.1).
  Endpoint check: 2.909% at 239.15 K and 2.330% at 293.15 K — reproduces the stated
  "2.3% ... to 2.9%" range.
- The M&M text does not state numerically which (Ts, gamma) pair enters the formula. The
  alternative (combined-fit Table S5 values Ts = 225.66 K, gamma = 1.6438) gives 2.938% /
  2.329% at the endpoints — also consistent with the stated range; the resulting deta
  differs by <= 1% (relative) from the values written. Flagged as the only interpretive
  choice in this CSV.
- Table S3 footnote confirms the convention: "The given accuracy value corresponds to
  1 SD" (Dehaoui 2015 row: accuracy "2.9-2.3" %).
- Temperature uncertainty: 0.15 K (M&M, "Experimental Setup").
- Calibration anchor: eta(T0 = 293.15 K) = 1.0016 +/- 0.0017 mPa s from ref. 67
  (Huber et al., J Phys Chem Ref Data 2009); the smoothed Table S2 value there is 0.994.

## 3. Power-law fits (Task B), verbatim parameter values

### 3.1 Smoothing fit to the authors' own raw viscosity data
SI, "Viscosity Values" section (unnumbered inline equation, SI p. 1):

> "the best fit with the lowest number of parameters was found to be given by a power
> law: eta(T) = eta0(T/Ts - 1)^-gamma, with eta0 = 1.3069x10^-4 Pa s, Ts = 224.80 K,
> and gamma = 1.7044, reduced chi^2 = 0.79."

Fit to the 369 raw points of Table S1 (their data alone, 239.15-293.15 K). No parameter
error bars are given for this fit. This fit generated the smoothed Table S2 values.

### 3.2 Combined-data power-law fits — Table S5 (SI p. 8)
Form (Table S5 title): **A0 (T/Ts - 1)^-gamma**; data selection in Table S3; procedure in
SI section "Power Law Fits to the Data on Self-Diffusion Coefficient and Rotational
Correlation Time": "The minimum of the temperature interval was fixed at the lowest
available temperature, and its maximum was adjusted until the fit quality was
satisfactory." Footnote: "The reported parameter errors correspond to a 68.3% confidence
interval."

| Quantity | T range, K | N | Reduced chi^2 | Ts, K | gamma | A0 |
|---|---|---|---|---|---|---|
| Viscosity eta | 239.15-373.15 | 49 | 0.91 | 225.66 +/- 0.18 | 1.6438 +/- 0.0052 | 137.88 +/- 0.26 uPa s |
| Self-diffusion coefficient Dt | 237.8-498.2 | 36 | 1.62 | 213.96 +/- 0.35 | -2.0801 +/- 0.0086 | 16,077 +/- 78 um^2 s^-1 |
| Rotational relaxation time tau_r | 236.18-451.63 | 51 | 0.61 | 223.05 +/- 0.14 | 1.8760 +/- 0.0065 | 217.89 +/- 0.90 fs |

- The eta fit is the same one shown in Fig. 2 (main paper) whose caption gives:
  "(Right) Power law representation, with best fit (chi^2 = 0.91) parameters
  Ts = 225.66 +/- 0.18 K, eta0 = (1.3788 +/- 0.0026)x10^-4 Pa s, and
  gamma = 1.6438 +/- 0.0052." (137.88 uPa s = 1.3788x10^-4 Pa s; consistent.)
  Main text: "The power law gives an excellent fit over the whole 134-K interval. It
  would extrapolate to a singular temperature Ts = 225.66 +/- 0.18 K."
- **Sign convention for Dt:** Table S5 uses A0(T/Ts-1)^-gamma with gamma NEGATIVE
  (-2.0801), so Dt INCREASES with T: Dt = A0 (T/Ts - 1)^{+2.0801}. In the "usual" form
  quoted in the SI text ("the usual power law fit [Dt(T) = D0(T/Ts - 1)^gamma]"):
  **Dt(T) = D0 (T/Ts - 1)^gamma with D0 = 16,077 um^2 s^-1 = 1.6077x10^-8 m^2 s^-1,
  Ts = 213.96 +/- 0.35 K, gamma = 2.0801 +/- 0.0086; validity 237.8-498.2 K.**
  This is the Dt power law later used by Singh et al. (2017) to generate D values at
  238, 243, 248 K.
- The fits are made directly to eta, Dt, tau_r. **No eta/T or D/T power-law variant is
  fitted anywhere** in the paper/SI (eta/T appears only inside the fractional
  Stokes-Einstein tests, Dt and tau_r proportional to (eta/T)^zeta, Fig. 4).
- Main text (p. 12022): "A power law also gives the best fit to Dt up to 500 K and to
  tau_r up to 450 K but with lower values of Ts" (rounded versions of 498.2/451.63 K).
  "Based on the error bars on the fit parameters, the Ts values for the three quantities
  are not consistent with each other" (SI).
- Viscosity fit input data (Table S3, "Selected data"): Dehaoui 2015 smoothed
  239.15-249.15 K (11 pts) + Hallett 1963, 250.15-273.15 K (24 pts) + Collings 1983,
  274.15-343.15 K (12 pts) + Kestin 1985 subset of 343.35-491.95 K. 11+24+12 = 47, so
  N = 49 implies 2 Kestin points inside the fit interval (inference, not printed). The
  SI text says the viscosity selection covers "the full temperature range between 239.15
  and 362.25 K", and Figs. 3/S5 normalize data "by their value at 362.25 K", so the
  highest eta datum in the fit is 362.25 K even though Table S5 prints the interval as
  239.15-373.15 K.

### 3.3 Non-power-law fits of eta (for completeness, Fig. 2 caption, main paper)
- Arrhenius eta(T) = eta0 exp[Ea/(kB T)]: apparent activation energy increases from
  1,560 to 6,410 K upon cooling (no single fit).
- Parabolic law (Eqs. S2-S3: eta = eta0 exp[J^2(1/T - 1/T0)^2 + Ea/(kB T)] for T < T0,
  Arrhenius above): chi^2 = 14.2, T0 = 305.15 K, eta0 = 2.323x10^-6 Pa s, J = 1,112 K,
  Ea = 1,769 K.
- VFT eta(T) = eta0 exp[B T0/(T - T0)]: chi^2 = 10.5, T0 = 168.9 K,
  eta0 = 4.442x10^-5 Pa s, B = 2.288.

## 4. Ambiguities / problem entries in the source

1. **Table S3 typo:** Dehaoui 2015 primary data printed as "239.15-298.15 K, 55 data".
   55 points at 1 K steps starting 239.15 ends at 293.15 K (as in Table S2); 298.15 K
   would give 60 points. 298.15 is evidently a typo for 293.15. CSV uses Table S2 as
   printed (239.15-293.15 K).
2. **Table S1 spurious entry:** Run 1, row 244.45 K contains a printed value "0.000"
   (impossible viscosity). Counting all printed values in Table S1 gives 370 including
   it; excluding it gives exactly the stated 369 measurements. Does not affect the CSV
   (built from Table S2).
3. **deta is derived** from the paper's printed 1-SD formula, not transcribed from a
   table (Sec. 2); parameter-set choice (Ts = 224.80 K, gamma = 1.7044) documented, with
   the alternative differing by <= 1% relative.
4. Scientific-notation values print without multiplication sign in the PDF text layer
   (e.g., "1.306910-4 Pa s" = 1.3069x10^-4 Pa s; "4.44210-5" = 4.442x10^-5); rendered
   page images confirm the readings.
5. Table S2 carries no pressure column; ambient/atmospheric pressure (0.1 MPa) is
   implied by the experiment (open capillary at atmospheric pressure) and stated in the
   paper's framing ("at ambient pressure"). P_MPa = 0.1 was added per extraction spec.

## 5. Verification performed

- Three independent channels agree on all 55 (T, eta) pairs: (a) rendered page image of
  SI p. 7 (read twice, before and after writing the CSV), (b) PyMuPDF text-layer
  extraction, (c) pdfplumber word extraction. Diff vs CSV: 0 mismatches.
- Row count: 55 data rows + header; T from 239.15 to 293.15 K in exact 1 K steps.
- 12 spot checks against the freshly re-rendered source table (T_K -> eta_mPas):
  239.15 -> 14.22, 243.15 -> 9.354, 248.15 -> 6.203, 253.15 -> 4.456, 258.15 -> 3.379,
  263.15 -> 2.663, 268.15 -> 2.161, 273.15 -> 1.794, 278.15 -> 1.517, 283.15 -> 1.302,
  288.15 -> 1.132, 293.15 -> 0.994. All match.
- deta column recomputed independently from the formula and compared: max deviation
  within rounding (<0.05%).
- No illegible values; no NaN entries were needed.
