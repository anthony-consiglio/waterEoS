"""Tab 0: Info — layout definition."""

from dash import html, dcc

_CONTENT = r"""
## About

**waterEoS** provides Python implementations of three two-state equations of state (EOS), one empirical Tait–Tammann EOS, and a two-state transport properties model for supercooled water, unified under a single [SeaFreeze](https://github.com/Bjournaux/SeaFreeze)-compatible API. This web app lets you interactively explore, compare, and compute thermodynamic properties across all models.

---

## Web App Guide

### Property Explorer

Plot any thermodynamic property as **isobars or isotherms** (curves mode), a **2D heatmap** with contour overlay, or a **rotatable 3D surface**. Select a model and property, set temperature/pressure ranges, then click **Update Plot**. Use the **Z/Color Axis** dropdown in surface modes to rearrange axes. Enable **Show phase boundaries** for two-state models to overlay the spinodal, binodal, and LLCP. Download curve data as CSV via the sidebar button.

### Phase Diagram

Compute and display the **liquid–liquid phase diagram** (T–P plane) for two-state models (Duska 2020, Holten 2014, Caupin 2019). Toggle binodal, spinodal, and LLCP visibility without recomputing. Click any point on the diagram to send its (T, P) coordinates to the Point Calculator tab.

### Model Comparison

Compare **two or more models** side-by-side or overlaid on a single plot. Select models and a shared property, then choose Overlay (model-colored curves on one axes) or Side by Side (one subplot per model with a shared y-axis).

### Point Calculator

Enter a temperature and pressure (or receive them from a Phase Diagram click) and select one or more models. Click **Calculate** to get a table of all thermodynamic properties at that state point, formatted to 6 significant figures.

### Settings

Customize curve palette, surface colormap, phase boundary colors, line widths, font size, grid visibility, and plot background. Changes apply to all tabs in real time. Settings are saved in your browser and persist across sessions. Click **Reset to Defaults** to restore the original appearance.

---

## Available Models

| Model key | Reference | LLCP (T, P) |
|-----------|-----------|-------------|
| `'holten2014'` | Holten, Sengers & Anisimov, J. Phys. Chem. Ref. Data **43**, 014101 (2014) | 228.2 K, 0 MPa |
| `'caupin2019'` | Caupin & Anisimov, J. Chem. Phys. **151**, 034503 (2019) | 218.1 K, 72.0 MPa |
| `'duska2020'` | Duska, J. Chem. Phys. **152**, 174501 (2020) | 220.9 K, 54.2 MPa |
| `'grenke2025'` | Grenke & Elliott, J. Phys. Chem. B **129**, 1997 (2025) | -- (empirical) |
| `'singh2017'` | Singh, Issenmann & Caupin, PNAS **114**, 4312 (2017) | -- (transport) |
| `'water1'` | SeaFreeze water1 (pass-through) | -- |
| `'IAPWS95'` | SeaFreeze IAPWS-95 (pass-through) | -- |

### Validity Ranges

The three two-state models accept **any** (T, P) input without raising errors, but results are only physically meaningful within the ranges below. The "paper-stated" range is where each model was validated by its authors; the "code-accessible" range is where the code runs without numerical failure (though results outside the paper range may be unphysical).

| Model | Paper-stated validity | Code-accessible range |
|-------|----------------------|-----------------------|
| `'holten2014'` | T_H(P)–300 K, 0–400 MPa (extrap. to 1000 MPa) | Unbounded (any T, P) |
| `'caupin2019'` | ~200–300 K, -140–400 MPa | Unbounded (any T, P) |
| `'duska2020'` | ~200–370 K, 0–100 MPa (extrap. to 200 MPa) | Unbounded (any T, P) |
| `'grenke2025'` | 200–300 K, 0.1–400 MPa | Unbounded (any T, P) |
| `'water1'` | 240–501 K, 0–2300 MPa | Enforced by SeaFreeze |
| `'IAPWS95'` | 240–501 K, 0–2300 MPa | Enforced by SeaFreeze |

**Notes:**
- T_H(P) is the homogeneous ice nucleation temperature (~235 K at 0.1 MPa, ~181 K at 200 MPa).
- Duska (2020) was fitted to data at positive pressures only; negative-pressure extrapolation is unvalidated.
- Caupin (2019) is the only model explicitly validated at negative pressures (stretched water).
- Grenke (2025) is a direct empirical Tait-Tammann correlation, not a two-state model. It has no `x`, `_A`, or `_B` outputs.
- Singh (2017) is a transport properties model that uses Holten (2014) as its thermodynamic backbone. It returns all Holten thermodynamic properties plus `eta`, `D`, and `tau_r`. Its validity range matches Holten (2014).
- Outside the paper-stated ranges, models may return unphysical values (e.g., negative compressibility or heat capacity) without warning.

---

## Output Properties

All models return the following mixture (equilibrium) properties. Two-state models also provide per-state values with `_A` (high-density) and `_B` (low-density) suffixes — **43 properties total**.

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

### Transport properties (`singh2017` only)

| Attribute | Property | Units |
|-----------|----------|-------|
| `eta` | Dynamic viscosity | Pa·s |
| `D` | Self-diffusion coefficient | m²/s |
| `tau_r` | Rotational correlation time | s |

---

## References

1. V. Holten, J. V. Sengers, and M. A. Anisimov, "Equation of state for supercooled water at pressures up to 400 MPa," *J. Phys. Chem. Ref. Data* **43**, 014101 (2014). [doi:10.1063/1.4895593](https://doi.org/10.1063/1.4895593)

2. F. Caupin and M. A. Anisimov, "Thermodynamics of supercooled and stretched water: Unifying two-structure description and liquid-vapor spinodal," *J. Chem. Phys.* **151**, 034503 (2019). [doi:10.1063/1.5100228](https://doi.org/10.1063/1.5100228)
   - Erratum: *J. Chem. Phys.* **163**, 039902 (2025). [doi:10.1063/5.0239673](https://doi.org/10.1063/5.0239673)

3. M. Duska, "Water above the spinodal," *J. Chem. Phys.* **152**, 174501 (2020). [doi:10.1063/5.0006431](https://doi.org/10.1063/5.0006431)

4. J. C. Grenke and J. R. Elliott, "Empirical fundamental equation of state for the metastable state of water based on the Tait-Tammann equation," *J. Phys. Chem. B* **129**, 1997-2012 (2025). [doi:10.1021/acs.jpcb.4c06847](https://doi.org/10.1021/acs.jpcb.4c06847)
   - Correction: *J. Phys. Chem. B* **129**, 9850-9853 (2025). [doi:10.1021/acs.jpcb.5c04618](https://doi.org/10.1021/acs.jpcb.5c04618)

5. L. P. Singh, B. Issenmann, and F. Caupin, "Pressure dependence of viscosity in supercooled water and a unified approach for thermodynamic and dynamic anomalies of water," *Proc. Natl. Acad. Sci. U.S.A.* **114**, 4312-4317 (2017). [doi:10.1073/pnas.1619501114](https://doi.org/10.1073/pnas.1619501114)
"""


def layout():
    return html.Div(
        style={
            'maxWidth': '900px',
            'margin': '0 auto',
            'padding': '24px 32px',
            'overflowY': 'auto',
            'height': 'calc(100vh - 60px)',
        },
        children=[
            # Header image
            html.Div(
                style={'textAlign': 'center', 'marginBottom': '8px'},
                children=[
                    html.Img(
                        src='/assets/waterEoS_header.png',
                        style={'maxWidth': '700px', 'width': '100%'},
                    ),
                ],
            ),
            # Badges
            html.Div(
                style={'textAlign': 'center', 'marginBottom': '16px'},
                children=[
                    html.A(
                        html.Img(src='https://img.shields.io/pypi/v/waterEoS.svg',
                                 alt='PyPI version'),
                        href='https://pypi.org/project/waterEoS/',
                        style={'marginRight': '6px'},
                    ),
                    html.A(
                        html.Img(src='https://img.shields.io/badge/License-GPLv3-blue.svg',
                                 alt='License: GPL v3'),
                        href='https://www.gnu.org/licenses/gpl-3.0',
                        style={'marginRight': '6px'},
                    ),
                    html.A(
                        html.Img(src='https://img.shields.io/pypi/pyversions/waterEoS.svg',
                                 alt='Python versions'),
                        href='https://pypi.org/project/waterEoS/',
                    ),
                ],
            ),
            # Body
            dcc.Markdown(
                _CONTENT,
                style={'color': '#e2e8f0', 'lineHeight': '1.6'},
                className='info-markdown',
            ),
            # Footer
            html.Div(
                style={
                    'marginTop': '24px', 'paddingTop': '16px',
                    'borderTop': '1px solid #1e3a5f',
                    'fontSize': '13px', 'color': '#94a3b8',
                    'paddingBottom': '32px',
                },
                children=[
                    html.P(['Author: Anthony Consiglio']),
                    html.P([
                        'Source: ',
                        html.A('github.com/anthony-consiglio/waterEoS',
                               href='https://github.com/anthony-consiglio/waterEoS',
                               style={'color': '#3b82f6'}),
                    ]),
                    html.P([
                        'License: ',
                        html.A('GNU General Public License v3.0',
                               href='https://www.gnu.org/licenses/gpl-3.0',
                               style={'color': '#3b82f6'}),
                    ]),
                ],
            ),
        ],
    )
