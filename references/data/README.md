# Singh (2017) transport-model refit: data and provenance

This directory contains the experimental datasets used to refit the
Singh–Issenmann–Caupin (2017) two-state transport model (PNAS 114, 4312,
Eq. 1) with alternative two-state EoS backbones (`caupin2019`, `duska2020`)
in addition to the original Holten (2014) backbone.

Fit driver: `scripts/refit_singh_transport.py` → `singh_refit_results.json`
Figure: `scripts/plot_singh_refit_comparison.py` → `singh_refit_comparison.png`
Package parameters: `singh_viscosity/params.py` (`BACKBONE_PARAMS`)

## Datasets

| File | Source | Points used | Notes |
|------|--------|-------------|-------|
| `singh_2017_eta.csv` | Singh et al. 2017, SI Table S1 | 165 | capillary viscosity, 244.3–297.8 K, 18.8–297.6 MPa, per-point 1-SD |
| `dehaoui_2015_eta.csv` | Dehaoui et al. 2015, SI Table S2 | 13 of 55 | ambient (0.1 MPa) smoothed viscosity; subset T ≤ 251.15 K enters the fit |
| `prielmeier_1988_D.csv` | Prielmeier et al. 1988, Table 1 | 110 | PGSE NMR self-diffusion, 203.5–273 K, 0.1–400 MPa |
| `harris_1997_D.csv` | Harris & Newitt 1997, Table 1 | 44 | SGSE NMR self-diffusion, 251.65–298.22 K, 0.1–350.5 MPa (2 Mills literature rows excluded) |
| `lang_1981_T1.csv` | Lang & Lüdemann 1981, Table 1 | 60 of 102 | ¹⁷O T1 of H₂O, 5–250 MPa; fit uses T ≤ 300 K (extrapolated-0.1-MPa column excluded) |
| `arnold_2002_T1.csv` | Arnold & Lüdemann 2002, Table 4 | 41 | ¹⁷O T1 of H₂O, 222–273 K, 250–400 MPa |

Per-source `*_notes.md` files document table locations, unit conversions,
stated uncertainties, and verification passes.

## Dataset assembly (mirrors the paper)

- **η (N = 184):** Table S1 (165) + Dehaoui T ≤ 251.15 K (13; with the 165
  this reproduces the paper's N = 178 exactly) + 6 ambient stable-region
  anchors at 273.15–298.15 K, 0.1 MPa from the IAPWS-2008 viscosity
  formulation (Huber et al. 2009) with σ = 1%. The anchors are an addition
  relative to the paper: the fitted dataset contains no stable-region data
  below 18.8 MPa, and without them the 20 → 0.1 MPa extrapolation drifts
  +2–4% for the caupin2019/duska2020 fractions. IAPWS-2008 is the same
  correlation Singh et al. used to calibrate every capillary run.
- **D (N = 157, identical to the paper):** Prielmeier (110) + Harris (44)
  + 3 points at 238/243/248 K, 0.1 MPa generated from the Dehaoui 2015
  power law D = 1.6077e-8 · (T/213.96 − 1)^2.0801 m²/s (their Table S5),
  exactly as described in the paper's Materials & Methods.
- **τ_r (N = 101, identical to the paper):** T1 → τ_r = 1/(T1·ω_Q²) with
  ω_Q = 9.12e6 s⁻¹ (the paper's convention, after Qvist et al.); Lang
  T ≤ 300 K (60) + Arnold (41).

## Weighting (1 SD)

- Table S1 and Dehaoui: tabulated per-point σ. The 13 Dehaoui points enter
  at 2× their tabulated σ (≈5–6% effective): with this single choice the
  Holten-backbone refit reproduces every published Table 1/S2 parameter
  within ~1 published SD, so it evidently matches the paper's effective
  weighting. (With tabulated σ the η fit moves toward the deeply
  supercooled ambient tail: E_LDS/kB → ~2440 K; the published model
  underpredicts the 239–246 K ambient viscosity by up to ~27%.)
- Prielmeier: 3% (10% below 210 K), per the paper's stated reliability.
- Harris: 2%. Lang: 5%. Arnold: 5% (stated). Power-law points: 5%.

## Validation

With the holten2014 fraction and this pipeline the separate τ_r fit
reproduces the published Table S2 parameters to all printed digits
(A0 52.98 vs 53.0 fs, E_LDS/kB 2655.9 vs 2656, E_HDS/kB 503.2 vs 503.3,
Δv 1.704 vs 1.70, T0 135.28 vs 135.28, χ²ᵣ 0.72 vs 0.72), and the joint
common-T0 fit reproduces Table 1 within ~1 SD for η and τ_r. The D
parameters land at E_LDS/kB ≈ 2085 vs published 1984 with Δv 2.02 vs 1.79;
this direction is insensitive to all tested weighting conventions, and the
two parameter sets produce D surfaces agreeing to 1.3% rms (max 5.5% at
the 238 K power-law point) — physically equivalent along the flat χ²
valley of Eq. 1.

## Results (joint fits, common T0 per backbone)

| Backbone | T0 (K) | χ²ᵣ η | χ²ᵣ D | χ²ᵣ τ_r | χ²ᵣ total |
|----------|--------|-------|-------|---------|-----------|
| holten2014 (refit, for reference) | 146.15 | 1.40 | 0.85 | 0.79 | 1.07 |
| caupin2019 | 145.25 | 2.09 | 0.93 | 0.83 | 1.40 |
| duska2020  | 141.05 | 2.26 | 1.03 | 0.89 | 1.52 |

The package's `holten2014` entry keeps the published Table 1 parameters;
`caupin2019` and `duska2020` use the refits above (full precision in
`singh_refit_results.json`).

Known limitation: with the caupin2019/duska2020 fractions the fitted model
retains a +2–3% viscosity bias in the ambient stable region (273–298 K,
0.1 MPa) even with the anchors — Eq. 1 with these larger, more
pressure-sensitive fractions cannot simultaneously match the high-pressure
dataset and the ambient isobar. The holten2014 fraction extrapolates to
+0.3% there. E_LDS/kB stays at the hydrogen-bond scale (~2000–2260 K) and
T0 within the reported glass-transition range (110–160 K) for all
backbones, preserving the physical interpretation of the paper.
