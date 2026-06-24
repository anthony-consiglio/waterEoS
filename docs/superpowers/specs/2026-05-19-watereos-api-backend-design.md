# waterEoS API Backend (sub-project A) — Design

**Date:** 2026-05-19
**Status:** Approved (design); pending spec review → implementation plan
**Part of:** waterEoS visualizer re-architecture (replaces the Dash app with a React front-end + Python API). This spec covers **sub-project A only — the FastAPI backend**. Sub-project B (Vite/React front-end) and sub-project C (deployment + retire Dash) are separate spec→plan→build cycles.

## 1. Summary

Build `watereos_api`, a FastAPI service that exposes the existing, validated
`watereos` thermodynamics package over HTTP as JSON. Plot endpoints return
**ready-to-render Plotly figure JSON** (built by reusing the proven figure
construction from the current Dash app, made Dash-independent and themed to the
new prototype design tokens). The Point Calculator endpoint and metadata
endpoints return raw structured data. No thermodynamic science is rewritten —
only wrapped, refactored for reuse, and served.

## 2. Goals / Non-goals

**Goals**
- A self-contained `watereos_api` package depending only on `watereos`.
- Endpoints covering every data need of the 7 visualizer tabs at **full parity
  with the current Dash app** (2D curves, 2D heatmap/contour, 3D surface,
  model comparison, EoS LL phase diagram, H₂O T-V/T-P/P-T-V phase diagram,
  point calculator, metadata, units).
- Plot endpoints return Plotly figure JSON with fidelity to the validated Dash
  plots, themed (dark/light) to match the prototype.
- Independently verifiable by automated tests with no browser and no front-end.

**Non-goals (handled in later sub-projects)**
- The React/Vite front-end (sub-project B).
- Render deployment config, static-asset serving, retiring the Dash package
  (sub-project C).
- Any change to thermodynamic algorithms or model science.

## 3. Reused `watereos` surface (no science changes)

Data computations (used as-is):
- `watereos/computation.py`: `compute_property_curves(model_key, prop_key,
  T_range, P_range, n_curves, n_points, isobar_mode)`,
  `compute_property_surface(model_key, prop_key, T_range, P_range, n_points)`,
  `compute_multi_model_curves(model_keys, prop_key, T_range, P_range,
  n_curves, n_points, isobar_mode)`,
  `compute_point_properties(model_keys, T_K, P_MPa)`,
  `compute_phase_diagram_data(model_key, …)`,
  `compute_property_at_forced_x(model_key, prop_key, T_arr, P_arr, x_arr)`.
- `watereos/model_registry.py`: `MODEL_REGISTRY`, `MODEL_ORDER`,
  `PROPERTY_LABELS`, `PROPERTY_UNITS`, `ModelInfo`, `get_display_label`,
  `get_common_properties`, `models_with_phase_diagram`.
- `watereos/tv_phase_diagram.py`: `compute_tv_phase_diagram(…)` and the
  `plot_tv_phase_diagram_plotly` / `plot_tp_phase_diagram_plotly` /
  `plot_ptv_phase_diagram_plotly` figure builders (these already return
  Plotly figures and are Dash-independent — reused directly).

## 4. Architecture

### 4.1 New package

```
watereos_api/
├── __init__.py
├── app.py            # FastAPI app, router registration, CORS, error handlers
├── schemas.py        # Pydantic request/response models
├── figures.py        # Dash-independent Plotly figure builders (extracted)
├── theming.py        # Plotly template from prototype tokens (dark/light)
├── cache.py          # keyed in-process cache for slow computes
├── serialization.py  # numpy/NaN/Inf-safe Plotly JSON helper
└── routes/
    ├── metadata.py   # GET /api/metadata, GET /api/health
    ├── figures.py    # POST /api/figures/*
    └── point.py      # POST /api/point
```

`watereos_api` imports only `watereos` (+ stdlib, fastapi, pydantic, plotly,
numpy). It must not import `watereos_visualizer`.

### 4.2 Required in-scope refactors

These are necessary because the API cannot depend on the to-be-retired Dash
package, and the figure logic currently lives inside Dash callbacks:

1. **Relocate units:** move `watereos_visualizer/units.py` →
   `watereos/units.py` (single source of truth: `UNIT_OPTIONS`,
   `UNIT_DEFAULTS`, `CATEGORY_LABELS`, `get_factor`, `display_label`,
   `convert_array`, `get_unit_string`). The Dash package (still present until
   sub-project C) is updated to import from the new location so it keeps
   working during the transition.
2. **Port figure builders → `watereos_api/figures.py` (copy, do not move):**
   re-implement the figure-construction logic from
   `watereos_visualizer/callbacks/property_explorer.py` (`_render_curves`,
   `_render_surface_2d`, `_render_surface_3d`, `_add_phase_traces_2d/3d`),
   `watereos_visualizer/callbacks/model_comparison.py` (`_build_overlay`,
   `_build_sidebyside`), `watereos_visualizer/callbacks/phase_diagram.py`
   (figure assembly + `_CURVE_STYLES`), and the layout/palette helpers in
   `watereos_visualizer/style.py` (`make_layout`, `make_layout_3d`,
   `get_phase_traces`, `get_palette`, `get_model_colors`). The new functions
   take **explicit parameters** (computed data, `theme`, palette, per-curve
   semantic colors, line widths, unit settings) — not a Dash `settings-store`
   dict. This is a **copy/port, not a move**: the original Dash callback and
   `style.py` code is left untouched so the transitional Dash app keeps
   working until sub-project C deletes it. Semantic colors and plot
   structure are reproduced verbatim so scientific fidelity is identical.
   (Accepted: temporary, intentional duplication of figure logic for the
   lifetime of the transition; sub-project C removes the Dash copy.)
3. **Theming:** `watereos_api/theming.py` defines `watereos_dark` /
   `watereos_light` Plotly templates whose colors/fonts derive from the
   prototype's `tokens.css` palette (not the old Mantine palette). Figure
   builders apply the template selected by the request `theme` field.

### 4.3 Request flow

`route → validate (Pydantic) → compute (watereos.computation) → optional unit
conversion (watereos.units) → build figure (figures.py + theming) →
serialize (serialization.py) → JSON response`. Point and metadata routes skip
the figure/serialize steps.

## 5. Endpoints

All POST bodies are Pydantic models; unknown fields rejected. All figure
responses are `{"figure": <plotly-figure-json>}` plus a `warnings` array
(validity-range messages, possibly empty). Numeric NaN/Inf serialize to JSON
`null`.

### 5.1 `GET /api/health`
→ `{"status": "ok", "version": <watereos version>}`.

### 5.2 `GET /api/metadata`
→ `{ models: [{ key, display_name, is_two_state, has_phase_diagram,
has_transport, T_min, T_max, P_min, P_max, properties: [key…] }] (in
MODEL_ORDER), properties: { key: { label, unit } }, units: { options:
UNIT_OPTIONS, defaults: UNIT_DEFAULTS, category_labels: CATEGORY_LABELS } }`.
Bootstraps all dropdowns and the ⌘K palette in one request.

### 5.3 `POST /api/figures/curves`
Body: `{ model, property, T_range:[min,max], P_range:[min,max], n_curves,
n_points, isobar_mode:bool, show_phase_boundaries:bool, theme:"dark"|"light",
units?:{…} }`. → curves figure JSON (isobars/isotherms; optional
spinodal/binodal/LLCP traces when `show_phase_boundaries` and model supports
it). Wraps `compute_property_curves` (+ phase traces via
`get_phase_traces`/`compute_property_at_forced_x`).

### 5.4 `POST /api/figures/surface2d`
Body: curves-style + `colormap`. → heatmap+contour figure JSON. Wraps
`compute_property_surface`.

### 5.5 `POST /api/figures/surface3d`
Body: same as surface2d. → 3D surface figure JSON (`make_layout_3d` scene).
Wraps `compute_property_surface`.

### 5.6 `POST /api/figures/compare`
Body: `{ model_keys:[…], property, T_range, P_range, n_curves, n_points,
isobar_mode, layout:"overlay"|"sidebyside", theme, units? }`. → comparison
figure JSON (overlay = single axes; sidebyside = subplots). Wraps
`compute_multi_model_curves`.

### 5.7 `POST /api/figures/eos-phase-diagram`
Body: `{ model, show:[curve keys…], auto_limits:bool, T_range?, P_range?,
theme }`. → EoS liquid–liquid phase-diagram figure JSON (binodal, HDL/LDL
spinodals, LLCP, TMD, Widom, ice Ih/III liquidus, Ih/III nucleation,
Kauzmann, triple point — per `show`). Wraps `compute_phase_diagram_data`.
**Cached** (see §6.4).

### 5.8 `POST /api/figures/h2o-phase-diagram`
Body: `{ projection:"tv"|"tp"|"ptv", V_range?, T_range?, P_range?, theme }`.
→ figure JSON from `compute_tv_phase_diagram` + the matching
`plot_{tv,tp,ptv}_phase_diagram_plotly`. **Heavy** — cached/precomputed
(§6.4).

### 5.9 `POST /api/point`
Body: `{ model_keys:[…], T_K, P_MPa, units?:{…} }`. → `{ results: { model_key:
{ prop_key: value|null } }, warnings: [validity messages] }`. Raw values
(rendered as a table by the front-end). Wraps `compute_point_properties`;
unit conversion via `watereos.units`.

## 6. Cross-cutting

### 6.1 Serialization
`watereos_api/serialization.py` wraps `plotly.io.to_json` (handles numpy);
post-process so non-finite floats (NaN/Inf, e.g. out-of-domain property
values) become JSON `null`. Endpoints return parsed JSON objects (FastAPI
serializes), media type `application/json`. A unit test asserts the helper
produces strictly valid JSON for a figure containing NaN.

### 6.2 Units
Requests may include a `units` object (subset of `UNIT_DEFAULTS` keys);
absent keys fall back to defaults. Conversion uses `watereos.units` before
figure building / point assembly. Axis titles reflect the chosen unit
(`display_label`). Single source of truth — never duplicated in JS.

### 6.3 Validation & warnings
Pydantic enforces types/required fields and rejects unknown models/properties
(404/422). Inputs are additionally checked against `ModelInfo` validity
ranges; out-of-range requests still compute (parity with Dash behavior) but
the response `warnings` array carries structured messages
(`{model, message}`) equivalent to the Dash `range_warning_banner` text.

### 6.4 Caching
`watereos_api/cache.py`: process-local cache keyed by (endpoint, normalized
params). `eos-phase-diagram` reuses precomputed JSON in `watereos/data/`
where available; `h2o-phase-diagram` results are cached after first compute.
First uncached call latency (~1–10 s for phase/H₂O) is accepted and
documented; all other endpoints are sub-second. No external cache/store.

### 6.5 CORS & config
CORS allows the Vite dev origin (`http://localhost:5173`) and configurable
allowed origins via env var; same-origin in production. App is runnable via
`uvicorn watereos_api.app:app --reload` for development. (Production
process/Render config is sub-project C.)

### 6.6 Error handling
Validation errors → 422 with field detail. Unknown model/property → 404.
Unexpected compute failures → 500 with a safe message and server-side log;
never leak stack traces in the body.

## 7. Testing (independently verifiable — no browser, no front-end)

`tests/test_watereos_api.py` using `fastapi.testclient.TestClient`:
- `/api/health` → 200, version present.
- `/api/metadata` → models/properties/units exactly match
  `model_registry` + `watereos.units` (order = `MODEL_ORDER`).
- Each `POST /api/figures/*` for a representative model/property: 200;
  response has `figure` with a non-empty `data` array and expected trace
  count/structure (e.g., `n_curves` line traces; surface3d has a `surface`
  trace; eos-phase-diagram includes requested curve traces); `layout`
  reflects requested `theme`.
- **Ground-truth parity:** for curves and point, assert returned numeric
  values equal calling the underlying `watereos.computation` function
  directly (within float tolerance) — guards scientific fidelity.
- `/api/point` matches `compute_point_properties`; `null` for unsupported
  props; `warnings` populated when (T,P) out of a model's validity range.
- Unit conversion: a property requested in a non-SI unit returns values
  scaled by the expected `watereos.units` factor and a matching axis label.
- Serialization: a figure with injected NaN serializes to valid JSON with
  `null`.
- Caching: second identical phase-diagram call is served from cache
  (faster / cache hit observable) and byte-identical.
- Regression guard: `pytest tests/` (existing backend suite) still green;
  the units relocation keeps `watereos_visualizer` importable (shim import).

## 8. Risks & mitigations

- **Figure-builder extraction drift.** The builders are entangled with Dash
  `style.py`/settings. Mitigation: port logic with explicit params, preserve
  semantic colors/structure verbatim, and add parity tests comparing key
  numeric series to direct `watereos.computation` output.
- **`plotly.io.to_json` NaN handling.** Verify it emits `null` (not `NaN`);
  the serialization helper + its unit test enforce strictly valid JSON.
- **Phase/H₂O latency.** Mitigated by precomputed JSON reuse + cache;
  documented first-call cost; not a correctness issue.
- **Units relocation breaking the still-present Dash app.** Keep a
  re-export shim at `watereos_visualizer/units.py` (imports from
  `watereos.units`) so the transitional Dash app and existing tests stay
  green until sub-project C removes Dash.

## 9. Rollback

All work is on branch `claude/cool-blackwell-d6413d` in an isolated worktree;
`watereos_api/` is additive and the units move ships with a back-compat shim,
so reverting the branch fully restores prior behavior. No published package
surface changes in sub-project A.
