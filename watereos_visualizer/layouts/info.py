"""Tab 0: Info — layout definition."""

from dash import html, dcc
import dash_mantine_components as dmc

_CONTENT = r"""
## About

**waterEoS** is a Python package for computing thermodynamic and transport properties of supercooled water. It unifies five equation-of-state models under a single [SeaFreeze](https://github.com/Bjournaux/SeaFreeze)-compatible API. This web app lets you interactively explore, compare, and compute thermodynamic properties across all models. For installation, usage, and full documentation see the [GitHub repository](https://github.com/anthony-consiglio/waterEoS).

---

## Web App Guide

### Property Explorer

Plot any thermodynamic property as **isobars or isotherms** (curves mode), a **2D heatmap** with contour overlay, or a **rotatable 3D surface**. Select a model and property from the sidebar, set temperature and pressure ranges, then click **Update Plot**.

In **Curves** mode, each curve corresponds to a fixed pressure (isobars) or temperature (isotherms). For two-state models, enable **Show phase boundaries** to overlay the LL spinodal, binodal, and LLCP on the plot. The example below shows specific volume isobars for Duska (2020) with phase boundaries — the liquid–liquid transition is visible as a sharp step at low temperatures.

![Property Explorer — Curves mode](/assets/duska_volume_curves.png)

Switch to **3D** mode for an interactive surface that you can rotate, zoom, and pan. The **Z / Color Axis** dropdown lets you choose which variable maps to the vertical axis vs. the color scale. Hover over the surface to read exact values. Below, the Caupin (2019) density surface shows the characteristic low-density region in the deeply supercooled regime.

![Property Explorer — 3D surface mode](/assets/caupin_density_surface.png)

Download the underlying curve data as a CSV file via the **Download CSV** button in the sidebar.

### H2O Phase Diagram

View the **multi-phase water phase diagram** computed via isothermal convex-hull analysis using [SeaFreeze](https://github.com/Bjournaux/SeaFreeze). Shows stability fields and coexistence boundaries for liquid water, Ice Ih, Ice II, Ice III, Ice V, and Ice VI. Choose between **T–V**, **T–P**, or **3D P–T–V** projections using the sidebar radio buttons, and adjust display limits to zoom into regions of interest. Hover over any phase region to see the phase identity and state variables.

![H2O Phase Diagram — 3D P–T–V projection](/assets/ptv_phase_diagram.png)

### EoS Phase Diagram

Compute and display the **liquid–liquid phase diagram** (T–P plane) for two-state models (Holten 2014, Caupin 2019, Duska 2020). The diagram includes the LL binodal, HDL and LDL spinodals, LLCP, temperature of maximum density (TMD), Widom line (Cp max), Kauzmann temperature (where liquid entropy equals ice entropy), and ice Ih/III liquidus and nucleation curves. Use the checkboxes to toggle individual curves on or off without recomputing. Click any point on the diagram to send its (T, P) coordinates directly to the Point Calculator tab.

![EoS Phase Diagram — Duska (2020)](/assets/duska_eos_phase_diagram.png)

### Model Comparison

Compare **two or more models** on the same property over a shared temperature and pressure range. Select models and a property from the sidebar, configure the axis ranges and number of curves, then choose **Overlay** (model-colored curves on a single axes) or **Side by Side** (one subplot per model with a shared y-axis). This is useful for identifying where models agree and where they diverge — especially in the deeply supercooled regime near the liquid–liquid critical point.

![Model Comparison — Duska vs. Holten density overlay](/assets/duska_holten_model_comparison.png)

### Point Calculator

Enter a temperature and pressure (or receive them from a Phase Diagram click) and select one or more models. Click **Calculate** to get a table of all thermodynamic properties at that state point, formatted to 6 significant figures. The table displays one column per model, making it easy to compare exact values. Units reflect the current selection in the Settings tab.

![Point Calculator — Duska & Holten at 273.15 K, 0.1 MPa](/assets/duska_holten_point_calculator.png)

### Settings

Customize the **appearance** of all plots: curve palette, surface colormap, phase boundary colors (binodal, spinodal, LLCP), line widths, font size, grid visibility, and background color. A live preview on the right updates as you adjust each control.

The **Units** section lets you change display units for density, specific volume, energy, entropy/heat capacity, bulk modulus, and viscosity. Unit conversions are applied at display time across all tabs — the underlying computation always runs in native SI units.

All settings are saved in your browser's local storage and persist across sessions. Click **Reset to Defaults** to restore the original configuration.

![Settings — appearance and unit controls](/assets/app_settings.png)

---

## Key Concepts

**Two-state model** — A framework that treats liquid water as a mixture of two local structures: **State A (HDL)**, a high-density disordered network, and **State B (LDL)**, a low-density tetrahedral network. The equilibrium LDL fraction is **x** (0 to 1). Properties suffixed `_A` and `_B` are per-state values; unsuffixed properties are the equilibrium mixture.

**Liquid--liquid critical point (LLCP)** — The predicted critical point at the top of the HDL/LDL coexistence dome. Each two-state model places it at a different (T, P). It has not been directly observed experimentally.

**Spinodal** — The stability limit where compressibility diverges. Beyond the spinodal, the liquid is mechanically unstable and must phase-separate. Two-state models have both an HDL spinodal and an LDL spinodal.

**Binodal (LLTL)** — The liquid--liquid coexistence curve where HDL and LDL have equal Gibbs energy. Lies between the two spinodals.

**Temperature of maximum density (TMD)** — Where water's density peaks at a given pressure (thermal expansivity = 0). About 277 K (4 C) at 0.1 MPa; shifts to lower T at higher pressures.

**Widom line** — A line of Cp maxima extending from the LLCP into the one-phase region, marking the crossover between HDL-like and LDL-like behavior.

**Kauzmann temperature** — Where the liquid's entropy equals that of ice Ih. Serves as a thermodynamic lower bound on the metastable liquid.

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
| `'singh2017'` | 200–300 K, 0–400 MPa (matches Holten backbone) | Unbounded (any T, P) |
| `'water1'` | 240–501 K, 0–2300 MPa | Enforced by SeaFreeze |
| `'IAPWS95'` | 240–501 K, 0–2300 MPa | Enforced by SeaFreeze |

**Notes:**
- T_H(P) is the homogeneous ice nucleation temperature (~235 K at 0.1 MPa, ~181 K at 200 MPa).
- Duska (2020) was fitted to data at positive pressures only; negative-pressure extrapolation is unvalidated.
- Caupin (2019) is the only model explicitly validated at negative pressures (stretched water).
- Grenke (2025) is a direct empirical Tait-Tammann correlation, not a two-state model. It has no `x`, `_A`, or `_B` outputs.
- Singh (2017) is a transport properties model that uses Holten (2014) as its thermodynamic backbone. It returns all Holten thermodynamic properties plus `eta`, `D`, and `tau_r`.
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
    return dmc.Paper(
        shadow="xs",
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
                style={'lineHeight': '1.6'},
                className='info-markdown',
            ),
            # Footer
            dmc.Divider(my="md"),
            dmc.Stack(gap="xs", children=[
                dmc.Text('Author: Anthony Consiglio', size="sm", c="dimmed"),
                dmc.Text([
                    'Source: ',
                    html.A('github.com/anthony-consiglio/waterEoS',
                           href='https://github.com/anthony-consiglio/waterEoS',
                           style={'color': 'var(--mantine-color-blue-5)'}),
                ], size="sm", c="dimmed"),
                dmc.Text([
                    'License: ',
                    html.A('GNU General Public License v3.0',
                           href='https://www.gnu.org/licenses/gpl-3.0',
                           style={'color': 'var(--mantine-color-blue-5)'}),
                ], size="sm", c="dimmed"),
            ], style={'paddingBottom': '32px'}),
        ],
    )
