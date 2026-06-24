# waterEoS API Backend (sub-project A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `watereos_api`, a FastAPI service that serves the existing `watereos` thermodynamics package over HTTP — metadata + point values as raw JSON, all plots as ready-to-render Plotly figure JSON at full parity with the Dash app.

**Architecture:** New `watereos_api/` package depending only on `watereos`. The validated `watereos.computation` data functions are reused untouched. Plotly figure-construction logic is **ported (copied)** out of the Dash callbacks/`style.py` into a Dash-independent `watereos_api/figures.py`, themed from the React prototype's design tokens. `watereos_visualizer/units.py` is moved to `watereos/units.py` with a back-compat shim so the transitional Dash app keeps working. Independently verified via FastAPI `TestClient` with ground-truth parity checks against `watereos.computation`.

**Tech Stack:** Python 3.14, FastAPI 0.135, Pydantic 2.12, uvicorn 0.42, httpx 0.28 (TestClient), Plotly 6.5, NumPy 2.4, pytest. All already installed.

**Spec:** `docs/superpowers/specs/2026-05-19-watereos-api-backend-design.md`

**Environment:** worktree `G:\My Drive\Isochoric\python_packages\waterEoS\.claude\worktrees\cool-blackwell-d6413d`; Python `C:/Python314/python.exe`; run commands via `cd "<worktree>" && <cmd>`; branch `claude/cool-blackwell-d6413d`. Commit normally (hooks ON, never `--no-verify`); **never** add AI/Claude/`Co-Authored-By` attribution to commits or files. Do NOT `git add -A` (untracked scratch: `_redesign_handoff/`, `design-preview.html`, `ref-standalone-1.png`, `.playwright-mcp/`, `_redesign_handoff`); add only the files each task names.

---

## File Structure

| File | Responsibility |
|---|---|
| `watereos/units.py` | **Moved** from `watereos_visualizer/units.py` (verbatim, import path only changes context) — unit conversion, single source of truth |
| `watereos_visualizer/units.py` | Becomes a back-compat shim re-exporting from `watereos.units` |
| `watereos_api/__init__.py` | Package marker + version |
| `watereos_api/app.py` | FastAPI app: CORS, exception handlers, router registration, `create_app()` |
| `watereos_api/schemas.py` | Pydantic request/response models |
| `watereos_api/serialization.py` | Plotly figure → JSON-safe dict (NaN/Inf → null) |
| `watereos_api/theming.py` | `watereos_dark`/`watereos_light` Plotly templates + curve palette from prototype tokens; `apply_theme(fig, theme)` |
| `watereos_api/figures.py` | Dash-independent figure builders ported from `watereos_visualizer` |
| `watereos_api/cache.py` | Process-local keyed cache for slow phase/H₂O computes |
| `watereos_api/routes/__init__.py` | Router aggregation |
| `watereos_api/routes/metadata.py` | `GET /api/health`, `GET /api/metadata` |
| `watereos_api/routes/point.py` | `POST /api/point` |
| `watereos_api/routes/figures.py` | `POST /api/figures/{curves,surface2d,surface3d,compare,eos-phase-diagram,h2o-phase-diagram}` |
| `tests/test_api_*.py` | pytest + `TestClient`, ground-truth parity |

Source references for the figure port (read these to reproduce logic faithfully):
- `watereos_visualizer/style.py`: `make_layout`, `make_layout_3d`, `get_phase_traces`, `get_palette`, `get_model_colors`, `range_warning_banner`, `PALETTE_OPTIONS`, `_is_dark`.
- `watereos_visualizer/callbacks/property_explorer.py`: `_render_curves` (~L332), `_render_surface_2d` (~L371), `_render_surface_3d` (~L450), `_add_phase_traces_2d` (~L498), `_add_phase_traces_3d` (~L543), `_error_figure` (~L46).
- `watereos_visualizer/callbacks/model_comparison.py`: `_build_overlay` (~L84), `_build_sidebyside` (~L128).
- `watereos_visualizer/callbacks/phase_diagram.py`: `_CURVE_STYLES` (~L21-32), `replot` figure assembly (~L120), `_empty_figure` (~L205).
- `watereos/tv_phase_diagram.py`: `plot_tv_phase_diagram_plotly`, `plot_tp_phase_diagram_plotly`, `plot_ptv_phase_diagram_plotly` (already Dash-independent — call directly).
- `watereos/computation.py`: `compute_property_curves`, `compute_property_surface`, `compute_multi_model_curves`, `compute_point_properties`, `compute_phase_diagram_data`, `compute_property_at_forced_x`.
- `watereos/model_registry.py`: `MODEL_REGISTRY`, `MODEL_ORDER`, `PROPERTY_LABELS`, `PROPERTY_UNITS`, `ModelInfo`, `get_display_label`, `get_common_properties`, `models_with_phase_diagram`.

Testing note: figure-builder ports are verified by (a) **structure** assertions on the returned Plotly JSON and (b) **ground-truth parity** — the figure's trace data must equal the corresponding `watereos.computation` output within float tolerance. This guarantees the port reproduces validated science without inlining the plotting code into this plan.

---

## Task 1: Package scaffold + health endpoint

**Files:**
- Create: `watereos_api/__init__.py`, `watereos_api/app.py`, `watereos_api/routes/__init__.py`, `watereos_api/routes/metadata.py`
- Create: `tests/test_api_health.py`
- Modify: `requirements-web.txt`

- [ ] **Step 1: Write the failing test** — `tests/test_api_health.py`:

```python
from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'watereos_api'`.

- [ ] **Step 3: Create the package files**

`watereos_api/__init__.py`:
```python
"""waterEoS HTTP API (FastAPI) over the watereos package."""
__version__ = "0.1.0"
```

`watereos_api/routes/__init__.py`:
```python
from fastapi import APIRouter
from watereos_api.routes import metadata

api_router = APIRouter(prefix="/api")
api_router.include_router(metadata.router)
```

`watereos_api/routes/metadata.py`:
```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    try:
        import watereos
        version = getattr(watereos, "__version__", "unknown")
    except Exception:
        version = "unknown"
    return {"status": "ok", "version": str(version)}
```

`watereos_api/app.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from watereos_api.routes import api_router


def create_app() -> FastAPI:
    app = FastAPI(title="waterEoS API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app


app = create_app()
```

- [ ] **Step 4: Add API deps to `requirements-web.txt`** — append these lines (keep existing lines intact):

```
fastapi>=0.115
uvicorn[standard]>=0.30
```

- [ ] **Step 5: Run test, expect PASS**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_health.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add watereos_api/__init__.py watereos_api/app.py watereos_api/routes/__init__.py watereos_api/routes/metadata.py tests/test_api_health.py requirements-web.txt
git commit -m "feat(api): scaffold FastAPI app with health endpoint"
```

---

## Task 2: Relocate `units.py` into `watereos` with back-compat shim

**Files:**
- Create: `watereos/units.py` (verbatim move of current `watereos_visualizer/units.py` content)
- Modify: `watereos_visualizer/units.py` → shim
- Create: `tests/test_units_relocation.py`

- [ ] **Step 1: Write the failing test** — `tests/test_units_relocation.py`:

```python
def test_units_importable_from_watereos():
    from watereos.units import (
        get_factor, get_unit_string, convert_array, display_label,
        UNIT_DEFAULTS, UNIT_OPTIONS, CATEGORY_LABELS,
    )
    # density g/cm³ factor is 1e-3 of SI kg/m³
    assert get_factor("rho", {"unit_density": "g/cm³"}) == 1e-3
    assert get_factor("rho", None) == 1.0
    assert "unit_energy" in UNIT_DEFAULTS
    assert display_label("rho", None) == "Density [kg/m³]"


def test_legacy_shim_still_works():
    # Transitional Dash app must keep importing from the old path
    from watereos_visualizer.units import get_factor as legacy_get_factor
    from watereos.units import get_factor as new_get_factor
    assert legacy_get_factor("V", {"unit_volume": "cm³/g"}) == \
        new_get_factor("V", {"unit_volume": "cm³/g"}) == 1e3
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_units_relocation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'watereos.units'`.

- [ ] **Step 3: Move the file**

Copy the **entire current content** of `watereos_visualizer/units.py` into a new file `watereos/units.py` **unchanged** (its only import is `from watereos.model_registry import PROPERTY_LABELS, PROPERTY_UNITS`, which is valid from inside the `watereos` package).

Then replace the whole content of `watereos_visualizer/units.py` with this shim:
```python
"""Back-compat shim. Unit logic moved to watereos.units (single source of truth)."""
from watereos.units import (  # noqa: F401
    MW_WATER, PROP_CATEGORY, UNIT_DEFAULTS, UNIT_OPTIONS, CATEGORY_LABELS,
    get_factor, get_unit_string, convert_array, display_label,
)
```

- [ ] **Step 4: Run tests, expect PASS (incl. no regression)**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_units_relocation.py tests/ -q`
Expected: new test passes; full existing suite stays green (the Dash app and any test importing `watereos_visualizer.units` resolve through the shim).

- [ ] **Step 5: Commit**

```bash
git add watereos/units.py watereos_visualizer/units.py tests/test_units_relocation.py
git commit -m "refactor(units): move units into watereos with back-compat shim"
```

---

## Task 3: JSON-safe Plotly serialization

**Files:**
- Create: `watereos_api/serialization.py`
- Create: `tests/test_api_serialization.py`

- [ ] **Step 1: Write the failing test** — `tests/test_api_serialization.py`:

```python
import json
import math
import numpy as np
import plotly.graph_objects as go
from watereos_api.serialization import figure_to_jsonable


def test_nan_and_inf_become_null_and_valid_json():
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1.0, np.nan, np.inf]))
    obj = figure_to_jsonable(fig)
    s = json.dumps(obj)            # must not raise
    parsed = json.loads(s)         # round-trips as valid JSON
    ys = parsed["data"][0]["y"]
    assert ys[0] == 1.0
    assert ys[1] is None           # NaN -> null
    assert ys[2] is None           # Inf -> null
    assert "NaN" not in s and "Infinity" not in s


def test_numpy_arrays_serialize():
    fig = go.Figure(go.Scatter(x=np.array([1, 2]), y=np.array([3.0, 4.0])))
    obj = figure_to_jsonable(fig)
    json.dumps(obj)
    assert list(obj["data"][0]["y"]) == [3.0, 4.0]
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_serialization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'watereos_api.serialization'`.

- [ ] **Step 3: Implement** — `watereos_api/serialization.py`:

```python
"""Convert a Plotly figure to a strictly-JSON-safe dict (NaN/Inf -> null)."""
import json
import math

import plotly.io as pio


def _sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def figure_to_jsonable(fig) -> dict:
    """Return a dict that json.dumps can serialize with no NaN/Infinity tokens.

    plotly.io.to_json handles numpy; we then round-trip with a strict parser
    and replace any non-finite floats with None.
    """
    raw = pio.to_json(fig, engine="json")
    obj = json.loads(raw)  # plotly may emit NaN/Infinity tokens; parse leniently
    return _sanitize(obj)
```

Note: `json.loads` accepts `NaN`/`Infinity` tokens by default (lenient), so the parse succeeds; `_sanitize` then removes them so the *output* `json.dumps(obj)` is strict-valid.

- [ ] **Step 4: Run test, expect PASS**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_serialization.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/serialization.py tests/test_api_serialization.py
git commit -m "feat(api): add NaN/Inf-safe Plotly JSON serialization"
```

---

## Task 4: Plotly theming from prototype tokens

**Files:**
- Create: `watereos_api/theming.py`
- Create: `tests/test_api_theming.py`

The prototype `tokens.css` palette (verbatim values to use):
- dark: bg `#0A0A0B`, bg_elev `#111114`, border `#1E1E22`, border_strong `#2A2A30`, text `#ECECF0`, text_muted `#9398A2`, text_faint `#5E626A`, grid `#1A1A1E`, grid_strong `#25252B`
- light: bg `#FAFAFA`, bg_elev `#FFFFFF`, border `#ECECEE`, border_strong `#DEDEE1`, text `#0A0A0A`, text_muted `#6B7280`, text_faint `#9CA3AF`, grid `#EEEEF0`, grid_strong `#DCDCDF`
- accent `#5b8def`; curves `#ef4444 #38bdf8 #84cc16 #e2e8f0 #a78bfa #f59e0b #ec4899`
- font sans `"Geist", ui-sans-serif, system-ui, -apple-system, sans-serif`; mono `"Geist Mono", ui-monospace, monospace`

- [ ] **Step 1: Write the failing test** — `tests/test_api_theming.py`:

```python
import plotly.graph_objects as go
from watereos_api.theming import apply_theme, CURVE_PALETTE, THEMES


def test_themes_defined():
    assert set(THEMES) == {"dark", "light"}
    assert THEMES["dark"]["bg_elev"] == "#111114"
    assert THEMES["light"]["bg_elev"] == "#FFFFFF"
    assert CURVE_PALETTE[0] == "#ef4444" and len(CURVE_PALETTE) == 7


def test_apply_theme_sets_layout_colors():
    fig = go.Figure(go.Scatter(x=[1], y=[1]))
    apply_theme(fig, "dark")
    lay = fig.layout
    assert lay.paper_bgcolor == "#111114"
    assert lay.plot_bgcolor == "#111114"
    assert lay.colorway == tuple(CURVE_PALETTE)
    apply_theme(fig, "light")
    assert fig.layout.paper_bgcolor == "#FFFFFF"


def test_apply_theme_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        apply_theme(go.Figure(), "neon")
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_theming.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** — `watereos_api/theming.py`:

```python
"""Plotly theming derived from the React prototype's design tokens."""

FONT_SANS = '"Geist", ui-sans-serif, system-ui, -apple-system, sans-serif'
FONT_MONO = '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace'

CURVE_PALETTE = [
    "#ef4444", "#38bdf8", "#84cc16", "#e2e8f0",
    "#a78bfa", "#f59e0b", "#ec4899",
]

THEMES = {
    "dark": {
        "bg": "#0A0A0B", "bg_elev": "#111114",
        "border": "#1E1E22", "border_strong": "#2A2A30",
        "text": "#ECECF0", "text_muted": "#9398A2", "text_faint": "#5E626A",
        "grid": "#1A1A1E", "grid_strong": "#25252B", "accent": "#5b8def",
    },
    "light": {
        "bg": "#FAFAFA", "bg_elev": "#FFFFFF",
        "border": "#ECECEE", "border_strong": "#DEDEE1",
        "text": "#0A0A0A", "text_muted": "#6B7280", "text_faint": "#9CA3AF",
        "grid": "#EEEEF0", "grid_strong": "#DCDCDF", "accent": "#5b8def",
    },
}


def _axis(c):
    return dict(
        gridcolor=c["grid"], zerolinecolor=c["grid_strong"],
        linecolor=c["grid_strong"],
        tickfont=dict(family=FONT_MONO, color=c["text_faint"], size=11),
        title=dict(font=dict(color=c["text_muted"], size=12)),
    )


def apply_theme(fig, theme: str):
    """Mutate fig.layout in place with the named theme. Raises ValueError if unknown."""
    if theme not in THEMES:
        raise ValueError(f"unknown theme: {theme!r}")
    c = THEMES[theme]
    fig.update_layout(
        paper_bgcolor=c["bg_elev"], plot_bgcolor=c["bg_elev"],
        font=dict(family=FONT_SANS, color=c["text"], size=12),
        colorway=CURVE_PALETTE,
        margin=dict(l=64, r=110, t=24, b=44),
        legend=dict(font=dict(family=FONT_MONO, size=11, color=c["text_muted"]),
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=c["bg_elev"], bordercolor=c["border_strong"],
                        font=dict(family=FONT_MONO, size=12, color=c["text"])),
    )
    fig.update_xaxes(**_axis(c))
    fig.update_yaxes(**_axis(c))
    # 3D scenes: apply same colors if a scene exists
    if fig.layout.scene is not None:
        sc = dict(xaxis=_axis(c), yaxis=_axis(c), zaxis=_axis(c),
                  bgcolor=c["bg_elev"])
        fig.update_layout(scene=sc)
    return fig
```

- [ ] **Step 4: Run test, expect PASS**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_theming.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/theming.py tests/test_api_theming.py
git commit -m "feat(api): add Plotly theming from prototype tokens"
```

---

## Task 5: Pydantic schemas

**Files:**
- Create: `watereos_api/schemas.py`
- Create: `tests/test_api_schemas.py`

- [ ] **Step 1: Write the failing test** — `tests/test_api_schemas.py`:

```python
import pytest
from pydantic import ValidationError
from watereos_api.schemas import (
    CurvesRequest, SurfaceRequest, CompareRequest, PointRequest,
    EosPhaseRequest, H2OPhaseRequest,
)


def test_curves_request_defaults_and_validation():
    req = CurvesRequest(model="duska2020", property="rho",
                         T_range=[200, 300], P_range=[0.1, 200])
    assert req.n_curves == 5 and req.n_points == 200
    assert req.isobar_mode is True and req.theme == "dark"
    with pytest.raises(ValidationError):
        CurvesRequest(model="duska2020", property="rho",
                      T_range=[200], P_range=[0.1, 200])  # bad tuple len
    with pytest.raises(ValidationError):
        CurvesRequest(model="duska2020", property="rho", T_range=[200, 300],
                      P_range=[0.1, 200], theme="neon")   # bad theme


def test_point_request():
    r = PointRequest(model_keys=["duska2020", "holten2014"], T_K=273.15, P_MPa=0.1)
    assert r.units is None


def test_h2o_request_projection_enum():
    H2OPhaseRequest(projection="tv")
    with pytest.raises(ValidationError):
        H2OPhaseRequest(projection="xyz")
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_schemas.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** — `watereos_api/schemas.py`:

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field, conlist

Theme = Literal["dark", "light"]
Pair = conlist(float, min_length=2, max_length=2)


class _UnitSettings(BaseModel):
    unit_density: Optional[str] = None
    unit_volume: Optional[str] = None
    unit_energy: Optional[str] = None
    unit_entropy: Optional[str] = None
    unit_bulk_modulus: Optional[str] = None
    unit_viscosity: Optional[str] = None


class CurvesRequest(BaseModel):
    model: str
    property: str
    T_range: Pair
    P_range: Pair
    n_curves: int = Field(5, ge=1, le=50)
    n_points: int = Field(200, ge=10, le=2000)
    isobar_mode: bool = True
    show_phase_boundaries: bool = False
    theme: Theme = "dark"
    units: Optional[_UnitSettings] = None


class SurfaceRequest(BaseModel):
    model: str
    property: str
    T_range: Pair
    P_range: Pair
    n_points: int = Field(80, ge=10, le=400)
    colormap: str = "rdbu"
    theme: Theme = "dark"
    units: Optional[_UnitSettings] = None


class CompareRequest(BaseModel):
    model_keys: conlist(str, min_length=1)
    property: str
    T_range: Pair
    P_range: Pair
    n_curves: int = Field(5, ge=1, le=50)
    n_points: int = Field(200, ge=10, le=2000)
    isobar_mode: bool = True
    layout: Literal["overlay", "sidebyside"] = "overlay"
    theme: Theme = "dark"
    units: Optional[_UnitSettings] = None


class EosPhaseRequest(BaseModel):
    model: str
    show: conlist(str, min_length=1) = Field(
        default=["binodal", "hdl_spinodal", "ldl_spinodal", "LLCP"])
    auto_limits: bool = True
    T_range: Optional[Pair] = None
    P_range: Optional[Pair] = None
    theme: Theme = "dark"


class H2OPhaseRequest(BaseModel):
    projection: Literal["tv", "tp", "ptv"] = "tv"
    V_range: Optional[Pair] = None
    T_range: Optional[Pair] = None
    P_range: Optional[Pair] = None
    theme: Theme = "dark"


class PointRequest(BaseModel):
    model_keys: conlist(str, min_length=1)
    T_K: float
    P_MPa: float
    units: Optional[_UnitSettings] = None
```

- [ ] **Step 4: Run test, expect PASS**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_schemas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/schemas.py tests/test_api_schemas.py
git commit -m "feat(api): add Pydantic request schemas"
```

---

## Task 6: `GET /api/metadata`

**Files:**
- Modify: `watereos_api/routes/metadata.py`
- Create: `tests/test_api_metadata.py`

- [ ] **Step 1: Write the failing test** — `tests/test_api_metadata.py`:

```python
from fastapi.testclient import TestClient
from watereos_api.app import create_app
from watereos.model_registry import MODEL_ORDER, MODEL_REGISTRY
from watereos.units import UNIT_DEFAULTS

client = TestClient(create_app())


def test_metadata_matches_registry():
    r = client.get("/api/metadata")
    assert r.status_code == 200
    body = r.json()
    keys = [m["key"] for m in body["models"]]
    assert keys == list(MODEL_ORDER)
    first = body["models"][0]
    info = MODEL_REGISTRY[first["key"]]
    assert first["display_name"] == info.display_name
    assert first["is_two_state"] == info.is_two_state
    assert first["T_min"] == info.T_min and first["P_max"] == info.P_max
    assert set(first["properties"]) == set(info.properties)
    assert body["units"]["defaults"] == UNIT_DEFAULTS
    assert "rho" in body["properties"]
    assert body["properties"]["rho"]["unit"]  # non-empty unit string
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_metadata.py -v`
Expected: FAIL — `/api/metadata` returns 404.

- [ ] **Step 3: Implement** — append to `watereos_api/routes/metadata.py`:

```python
@router.get("/metadata")
def metadata():
    from watereos.model_registry import (
        MODEL_ORDER, MODEL_REGISTRY, PROPERTY_LABELS, PROPERTY_UNITS,
    )
    from watereos.units import UNIT_OPTIONS, UNIT_DEFAULTS, CATEGORY_LABELS

    models = []
    for key in MODEL_ORDER:
        info = MODEL_REGISTRY[key]
        models.append({
            "key": key,
            "display_name": info.display_name,
            "is_two_state": info.is_two_state,
            "has_phase_diagram": info.has_phase_diagram,
            "has_transport": info.has_transport,
            "T_min": info.T_min, "T_max": info.T_max,
            "P_min": info.P_min, "P_max": info.P_max,
            "properties": list(info.properties),
        })
    properties = {
        k: {"label": PROPERTY_LABELS.get(k, k), "unit": PROPERTY_UNITS.get(k, "")}
        for k in PROPERTY_LABELS
    }
    return {
        "models": models,
        "properties": properties,
        "units": {
            "options": UNIT_OPTIONS,
            "defaults": UNIT_DEFAULTS,
            "category_labels": CATEGORY_LABELS,
        },
    }
```

- [ ] **Step 4: Run test, expect PASS**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_metadata.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/routes/metadata.py tests/test_api_metadata.py
git commit -m "feat(api): add GET /api/metadata"
```

---

## Task 7: `POST /api/point` (raw values + validity warnings)

**Files:**
- Create: `watereos_api/routes/point.py`
- Modify: `watereos_api/routes/__init__.py` (register router)
- Create: `tests/test_api_point.py`

- [ ] **Step 1: Write the failing test** — `tests/test_api_point.py`:

```python
from fastapi.testclient import TestClient
from watereos_api.app import create_app
from watereos.computation import compute_point_properties

client = TestClient(create_app())


def test_point_matches_ground_truth():
    payload = {"model_keys": ["duska2020"], "T_K": 273.15, "P_MPa": 0.1}
    r = client.post("/api/point", json=payload)
    assert r.status_code == 200
    body = r.json()
    truth = compute_point_properties(["duska2020"], 273.15, 0.1)["duska2020"]
    got = body["results"]["duska2020"]
    # rho present and equal within tolerance
    assert abs(got["rho"] - truth["rho"]) < 1e-6 * abs(truth["rho"])


def test_point_out_of_range_warns():
    # duska2020 valid T is [200,370] K; 500 K is out of range
    r = client.post("/api/point",
                     json={"model_keys": ["duska2020"], "T_K": 500.0, "P_MPa": 0.1})
    assert r.status_code == 200
    assert any("duska" in w["model"].lower() or "Duska" in w["message"]
               for w in r.json()["warnings"])


def test_point_unit_conversion():
    r = client.post("/api/point", json={
        "model_keys": ["duska2020"], "T_K": 273.15, "P_MPa": 0.1,
        "units": {"unit_density": "g/cm³"}})
    si = compute_point_properties(["duska2020"], 273.15, 0.1)["duska2020"]["rho"]
    got = r.json()["results"]["duska2020"]["rho"]
    assert abs(got - si * 1e-3) < 1e-9 * abs(si * 1e-3)
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_point.py -v`
Expected: FAIL — `/api/point` 404.

- [ ] **Step 3: Implement** — `watereos_api/routes/point.py`:

```python
import math
from fastapi import APIRouter, HTTPException

from watereos_api.schemas import PointRequest
from watereos.computation import compute_point_properties
from watereos.model_registry import MODEL_REGISTRY
from watereos.units import get_factor

router = APIRouter()


def _validity_warnings(model_keys, T_K, P_MPa):
    out = []
    for mk in model_keys:
        info = MODEL_REGISTRY.get(mk)
        if info is None:
            continue
        parts = []
        if T_K < info.T_min or T_K > info.T_max:
            parts.append(f"T {T_K:.2f} K outside [{info.T_min:.1f}, "
                         f"{info.T_max:.1f}] K")
        if P_MPa < info.P_min or P_MPa > info.P_max:
            parts.append(f"P {P_MPa:.2f} MPa outside [{info.P_min:.1f}, "
                         f"{info.P_max:.1f}] MPa")
        if parts:
            out.append({"model": info.display_name,
                        "message": f"{info.display_name}: " + "; ".join(parts)})
    return out


@router.post("/point")
def point(req: PointRequest):
    unknown = [m for m in req.model_keys if m not in MODEL_REGISTRY]
    if unknown:
        raise HTTPException(status_code=404,
                            detail=f"unknown model(s): {unknown}")
    raw = compute_point_properties(req.model_keys, req.T_K, req.P_MPa)
    units = req.units.model_dump(exclude_none=True) if req.units else None
    results = {}
    for mk, props in raw.items():
        conv = {}
        for pk, val in props.items():
            if val is None:
                conv[pk] = None
                continue
            v = float(val) * get_factor(pk, units)
            conv[pk] = v if math.isfinite(v) else None
        results[mk] = conv
    return {"results": results,
            "warnings": _validity_warnings(req.model_keys, req.T_K, req.P_MPa)}
```

Register it — in `watereos_api/routes/__init__.py` change to:
```python
from fastapi import APIRouter
from watereos_api.routes import metadata, point

api_router = APIRouter(prefix="/api")
api_router.include_router(metadata.router)
api_router.include_router(point.router)
```

- [ ] **Step 4: Run test, expect PASS**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_point.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/routes/point.py watereos_api/routes/__init__.py tests/test_api_point.py
git commit -m "feat(api): add POST /api/point with unit conversion and warnings"
```

---

## Task 8: `figures.py` — ported layout/curve builders + `/api/figures/curves`

**Files:**
- Create: `watereos_api/figures.py`
- Create: `watereos_api/routes/figures.py`
- Modify: `watereos_api/routes/__init__.py`
- Create: `tests/test_api_figures_curves.py`

Port reference: reproduce the logic of `watereos_visualizer/style.py::make_layout`, `get_palette`, `get_model_colors`, `get_phase_traces`, and `watereos_visualizer/callbacks/property_explorer.py::_render_curves` + `_add_phase_traces_2d`. The new functions take explicit args, not a Dash settings dict; theming comes from `watereos_api.theming.apply_theme`. Curve/phase **data** comes from `watereos.computation.compute_property_curves` and `compute_property_at_forced_x` / `compute_phase_diagram_data` — unchanged. Semantic colors (spinodal red dashed, binodal accent, LLCP marker) preserved exactly as in `get_phase_traces`.

- [ ] **Step 1: Write the failing test** — `tests/test_api_figures_curves.py`:

```python
from fastapi.testclient import TestClient
from watereos_api.app import create_app
from watereos.computation import compute_property_curves

client = TestClient(create_app())

BASE = dict(model="duska2020", property="rho", T_range=[200, 360],
            P_range=[0.1, 200], n_curves=4, n_points=120,
            isobar_mode=True, theme="dark")


def test_curves_figure_structure_and_parity():
    r = client.post("/api/figures/curves", json=BASE)
    assert r.status_code == 200
    fig = r.json()["figure"]
    line_traces = [t for t in fig["data"]
                   if t.get("mode", "lines") in ("lines", "lines+markers")
                   and t.get("type", "scatter") == "scatter"]
    assert len(line_traces) >= 4                       # one per curve
    assert fig["layout"]["paper_bgcolor"] == "#111114"  # dark theme applied
    # ground-truth parity: first curve's y equals compute_property_curves
    truth = compute_property_curves("duska2020", "rho", (200, 360),
                                    (0.1, 200), 4, 120, True)
    ty = list(truth["y_values"][0])
    fy = [v for v in line_traces[0]["y"] if v is not None]
    assert len(fy) == len(ty)
    assert max(abs(a - b) for a, b in zip(fy, ty[:len(fy)])) < 1e-6 * (
        max(abs(v) for v in ty) or 1)


def test_curves_phase_boundaries_add_traces():
    n0 = len(client.post("/api/figures/curves", json=BASE).json()["figure"]["data"])
    n1 = len(client.post("/api/figures/curves",
             json={**BASE, "show_phase_boundaries": True}
             ).json()["figure"]["data"])
    assert n1 > n0


def test_curves_unknown_model_404():
    r = client.post("/api/figures/curves", json={**BASE, "model": "nope"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_curves.py -v`
Expected: FAIL — endpoint 404 / module missing.

- [ ] **Step 3: Implement `watereos_api/figures.py`** (curve builder; reproduces `_render_curves` + `get_phase_traces` logic with explicit args):

```python
"""Dash-independent Plotly figure builders (ported from watereos_visualizer)."""
import plotly.graph_objects as go

from watereos.computation import (
    compute_property_curves, compute_phase_diagram_data,
)
from watereos.model_registry import MODEL_REGISTRY, get_display_label
from watereos.units import convert_array, display_label
from watereos_api.theming import apply_theme, CURVE_PALETTE


def build_curves_figure(*, model, prop, T_range, P_range, n_curves,
                        n_points, isobar_mode, show_phase, theme, units):
    data = compute_property_curves(model, prop, tuple(T_range),
                                   tuple(P_range), n_curves, n_points,
                                   isobar_mode)
    fig = go.Figure()
    for i, (xs, ys, lbl) in enumerate(zip(
            data["x_values"], data["y_values"], data["curve_labels"])):
        y = convert_array(prop, list(ys), units)
        fig.add_trace(go.Scatter(
            x=list(xs), y=y, mode="lines", name=lbl,
            line=dict(color=CURVE_PALETTE[i % len(CURVE_PALETTE)], width=2),
        ))
    if show_phase and MODEL_REGISTRY[model].has_phase_diagram:
        pd = compute_phase_diagram_data(model)
        _add_phase_traces(fig, pd)
    fig.update_layout(
        title=dict(text=data.get("title")),
        xaxis_title=data.get("x_label"),
        yaxis_title=display_label(prop, units),
    )
    apply_theme(fig, theme)
    return fig


def _add_phase_traces(fig, pd):
    """Spinodal (red dashed), binodal (accent), LLCP (violet marker).

    Semantic colors reproduced verbatim from
    watereos_visualizer/style.py::get_phase_traces.
    """
    sp = pd.get("hdl_spinodal") or {}
    if sp.get("T_K") is not None:
        fig.add_trace(go.Scatter(
            x=list(sp["p_MPa"]), y=list(sp["T_K"]), mode="lines",
            name="Spinodal",
            line=dict(color="#ef4444", width=1.5, dash="dash")))
    sp2 = pd.get("ldl_spinodal") or {}
    if sp2.get("T_K") is not None:
        fig.add_trace(go.Scatter(
            x=list(sp2["p_MPa"]), y=list(sp2["T_K"]), mode="lines",
            name="Spinodal", showlegend=False,
            line=dict(color="#ef4444", width=1.5, dash="dash")))
    bn = pd.get("binodal") or {}
    if bn.get("T_K") is not None:
        fig.add_trace(go.Scatter(
            x=list(bn["p_MPa"]), y=list(bn["T_K"]), mode="lines",
            name="Binodal", line=dict(color="#5b8def", width=1.5)))
    llcp = pd.get("LLCP") or {}
    if llcp.get("T_K") is not None:
        fig.add_trace(go.Scatter(
            x=[float(llcp["p_MPa"])], y=[float(llcp["T_K"])],
            mode="markers", name="LLCP",
            marker=dict(color="#a78bfa", size=10,
                        line=dict(width=1, color="white"))))
```

> NOTE TO IMPLEMENTER: confirm the exact key names returned by `compute_phase_diagram_data` for the spinodal/binodal/LLCP sub-dicts by reading `watereos/computation.py` (the spec references `hdl_spinodal`/`ldl_spinodal`/`binodal`/`LLCP` with `T_K`/`p_MPa`). Adjust the key access above to match the actual structure if it differs (e.g. `spinodal.T_upper/p_array`). The parity test only checks the main curve traces; ensure phase traces are added (count increases) — exact phase geometry is parity-checked in Task 11.

`watereos_api/routes/figures.py`:
```python
from fastapi import APIRouter, HTTPException
from watereos_api.schemas import CurvesRequest
from watereos_api.serialization import figure_to_jsonable
from watereos_api import figures
from watereos.model_registry import MODEL_REGISTRY

router = APIRouter(prefix="/figures")


def _check_model(m):
    if m not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown model: {m}")


@router.post("/curves")
def curves(req: CurvesRequest):
    _check_model(req.model)
    units = req.units.model_dump(exclude_none=True) if req.units else None
    fig = figures.build_curves_figure(
        model=req.model, prop=req.property, T_range=req.T_range,
        P_range=req.P_range, n_curves=req.n_curves, n_points=req.n_points,
        isobar_mode=req.isobar_mode, show_phase=req.show_phase_boundaries,
        theme=req.theme, units=units)
    return {"figure": figure_to_jsonable(fig), "warnings": []}
```

Register in `watereos_api/routes/__init__.py`:
```python
from fastapi import APIRouter
from watereos_api.routes import metadata, point, figures

api_router = APIRouter(prefix="/api")
api_router.include_router(metadata.router)
api_router.include_router(point.router)
api_router.include_router(figures.router)
```

- [ ] **Step 4: Run test, expect PASS** (read `watereos/computation.py` first to confirm `compute_phase_diagram_data` key names; fix `_add_phase_traces` accessors if needed)

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_curves.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/figures.py watereos_api/routes/figures.py watereos_api/routes/__init__.py tests/test_api_figures_curves.py
git commit -m "feat(api): add POST /api/figures/curves with phase boundaries"
```

---

## Task 9: `/api/figures/surface2d` and `/api/figures/surface3d`

**Files:**
- Modify: `watereos_api/figures.py`, `watereos_api/routes/figures.py`
- Create: `tests/test_api_figures_surface.py`

Port reference: `watereos_visualizer/callbacks/property_explorer.py::_render_surface_2d` (Heatmap + Contour, `colorscale=colormap`) and `_render_surface_3d` (`go.Surface`). Data from `watereos.computation.compute_property_surface` → `{T_grid, P_grid, Z}` (2-D meshgrids).

- [ ] **Step 1: Write the failing test** — `tests/test_api_figures_surface.py`:

```python
from fastapi.testclient import TestClient
from watereos_api.app import create_app
from watereos.computation import compute_property_surface

client = TestClient(create_app())
BASE = dict(model="duska2020", property="rho", T_range=[210, 360],
            P_range=[0.1, 200], n_points=40, colormap="rdbu", theme="dark")


def test_surface2d_has_heatmap_and_parity():
    r = client.post("/api/figures/surface2d", json=BASE)
    assert r.status_code == 200
    fig = r.json()["figure"]
    types = {t["type"] for t in fig["data"]}
    assert "heatmap" in types or "contour" in types
    truth = compute_property_surface("duska2020", "rho", (210, 360),
                                     (0.1, 200), 40)
    z_truth = [float(v) for row in truth["Z"] for v in row
               if v == v]  # drop NaN
    hm = next(t for t in fig["data"] if t["type"] in ("heatmap", "contour"))
    z_fig = [float(v) for row in hm["z"] for v in row if v is not None]
    assert abs(max(z_fig) - max(z_truth)) < 1e-6 * (abs(max(z_truth)) or 1)


def test_surface3d_has_surface_trace():
    r = client.post("/api/figures/surface3d", json=BASE)
    assert r.status_code == 200
    fig = r.json()["figure"]
    assert any(t["type"] == "surface" for t in fig["data"])
    assert fig["layout"]["scene"]["xaxis"]["gridcolor"] == "#1A1A1E"
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_surface.py -v`
Expected: FAIL — endpoints 404.

- [ ] **Step 3: Implement** — append to `watereos_api/figures.py`:

```python
from watereos.computation import compute_property_surface


def build_surface2d_figure(*, model, prop, T_range, P_range, n_points,
                           colormap, theme, units):
    d = compute_property_surface(model, prop, tuple(T_range),
                                 tuple(P_range), n_points)
    z = d["Z"]
    if units:
        f = __import__("watereos.units", fromlist=["get_factor"]).get_factor(
            prop, units)
        if f != 1.0:
            z = [[(v * f) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        x=list(d["T_grid"][0]), y=[r[0] for r in d["P_grid"]],
        z=z, colorscale=colormap, colorbar=dict(title=display_label(prop, units))))
    fig.update_layout(xaxis_title="Temperature [K]",
                      yaxis_title="Pressure [MPa]")
    apply_theme(fig, theme)
    return fig


def build_surface3d_figure(*, model, prop, T_range, P_range, n_points,
                           colormap, theme, units):
    d = compute_property_surface(model, prop, tuple(T_range),
                                 tuple(P_range), n_points)
    z = d["Z"]
    if units:
        f = __import__("watereos.units", fromlist=["get_factor"]).get_factor(
            prop, units)
        if f != 1.0:
            z = [[(v * f) for v in row] for row in z]
    fig = go.Figure(go.Surface(
        x=d["T_grid"], y=d["P_grid"], z=z, colorscale=colormap, opacity=0.95))
    fig.update_layout(scene=dict(
        xaxis_title="Temperature [K]", yaxis_title="Pressure [MPa]",
        zaxis_title=display_label(prop, units), aspectmode="cube"))
    apply_theme(fig, theme)
    return fig
```

> NOTE TO IMPLEMENTER: verify `compute_property_surface` return keys/orientation (`T_grid`, `P_grid`, `Z` as 2-D meshgrids) by reading `watereos/computation.py`; align x/y extraction (`T_grid[0]` row vs column) with the actual meshgrid axis order so the heatmap is not transposed. The parity test checks max(Z); also eyeball a transpose by asserting shape if needed.

Append to `watereos_api/routes/figures.py`:
```python
from watereos_api.schemas import SurfaceRequest


@router.post("/surface2d")
def surface2d(req: SurfaceRequest):
    _check_model(req.model)
    units = req.units.model_dump(exclude_none=True) if req.units else None
    fig = figures.build_surface2d_figure(
        model=req.model, prop=req.property, T_range=req.T_range,
        P_range=req.P_range, n_points=req.n_points, colormap=req.colormap,
        theme=req.theme, units=units)
    return {"figure": figure_to_jsonable(fig), "warnings": []}


@router.post("/surface3d")
def surface3d(req: SurfaceRequest):
    _check_model(req.model)
    units = req.units.model_dump(exclude_none=True) if req.units else None
    fig = figures.build_surface3d_figure(
        model=req.model, prop=req.property, T_range=req.T_range,
        P_range=req.P_range, n_points=req.n_points, colormap=req.colormap,
        theme=req.theme, units=units)
    return {"figure": figure_to_jsonable(fig), "warnings": []}
```

- [ ] **Step 4: Run test, expect PASS** (read `compute_property_surface` first; fix axis orientation if the heatmap is transposed)

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_surface.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/figures.py watereos_api/routes/figures.py tests/test_api_figures_surface.py
git commit -m "feat(api): add surface2d and surface3d figure endpoints"
```

---

## Task 10: `/api/figures/compare`

**Files:**
- Modify: `watereos_api/figures.py`, `watereos_api/routes/figures.py`
- Create: `tests/test_api_figures_compare.py`

Port reference: `watereos_visualizer/callbacks/model_comparison.py::_build_overlay` (single axes, one trace per model×curve, model colors via `get_model_colors`) and `_build_sidebyside` (`plotly.subplots.make_subplots(rows=1, cols=n_models, shared_yaxes=True)`). Data from `watereos.computation.compute_multi_model_curves` → `{model_key: curves_dict}`.

- [ ] **Step 1: Write the failing test** — `tests/test_api_figures_compare.py`:

```python
from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())
BASE = dict(model_keys=["duska2020", "holten2014"], property="rho",
            T_range=[210, 290], P_range=[0.1, 200], n_curves=3,
            n_points=80, isobar_mode=True, theme="dark")


def test_compare_overlay_single_axes():
    fig = client.post("/api/figures/compare",
                      json={**BASE, "layout": "overlay"}).json()["figure"]
    # overlay: traces share one xaxis ("x" or unset), no x2 axis in layout
    assert "xaxis2" not in fig["layout"]
    assert len(fig["data"]) >= 2


def test_compare_sidebyside_subplots():
    fig = client.post("/api/figures/compare",
                      json={**BASE, "layout": "sidebyside"}).json()["figure"]
    assert "xaxis2" in fig["layout"]      # 2 models -> 2 subplot columns


def test_compare_unknown_model_404():
    r = client.post("/api/figures/compare",
                    json={**BASE, "model_keys": ["duska2020", "bogus"]})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_compare.py -v`
Expected: FAIL — endpoint 404.

- [ ] **Step 3: Implement** — append to `watereos_api/figures.py`:

```python
from plotly.subplots import make_subplots
from watereos.computation import compute_multi_model_curves


def build_compare_figure(*, model_keys, prop, T_range, P_range, n_curves,
                         n_points, isobar_mode, layout, theme, units):
    multi = compute_multi_model_curves(model_keys, prop, tuple(T_range),
                                       tuple(P_range), n_curves, n_points,
                                       isobar_mode)
    if layout == "sidebyside":
        titles = [MODEL_REGISTRY[m].display_name for m in model_keys]
        fig = make_subplots(rows=1, cols=len(model_keys),
                            subplot_titles=titles, shared_yaxes=True)
        for col, mk in enumerate(model_keys, start=1):
            d = multi[mk]
            for i, (xs, ys, lbl) in enumerate(zip(
                    d["x_values"], d["y_values"], d["curve_labels"])):
                fig.add_trace(go.Scatter(
                    x=list(xs), y=convert_array(prop, list(ys), units),
                    mode="lines", name=f"{mk}:{lbl}", showlegend=(col == 1),
                    line=dict(color=CURVE_PALETTE[i % len(CURVE_PALETTE)])),
                    row=1, col=col)
    else:
        fig = go.Figure()
        for j, mk in enumerate(model_keys):
            d = multi[mk]
            for i, (xs, ys, lbl) in enumerate(zip(
                    d["x_values"], d["y_values"], d["curve_labels"])):
                fig.add_trace(go.Scatter(
                    x=list(xs), y=convert_array(prop, list(ys), units),
                    mode="lines", name=f"{MODEL_REGISTRY[mk].display_name} · {lbl}",
                    line=dict(color=CURVE_PALETTE[
                        (j * n_curves + i) % len(CURVE_PALETTE)])))
        fig.update_layout(yaxis_title=display_label(prop, units))
    apply_theme(fig, theme)
    return fig
```

Append to `watereos_api/routes/figures.py`:
```python
from watereos_api.schemas import CompareRequest


@router.post("/compare")
def compare(req: CompareRequest):
    for m in req.model_keys:
        _check_model(m)
    units = req.units.model_dump(exclude_none=True) if req.units else None
    fig = figures.build_compare_figure(
        model_keys=req.model_keys, prop=req.property, T_range=req.T_range,
        P_range=req.P_range, n_curves=req.n_curves, n_points=req.n_points,
        isobar_mode=req.isobar_mode, layout=req.layout, theme=req.theme,
        units=units)
    return {"figure": figure_to_jsonable(fig), "warnings": []}
```

- [ ] **Step 4: Run test, expect PASS**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/figures.py watereos_api/routes/figures.py tests/test_api_figures_compare.py
git commit -m "feat(api): add model comparison figure endpoint"
```

---

## Task 11: `/api/figures/eos-phase-diagram` + caching

**Files:**
- Create: `watereos_api/cache.py`
- Modify: `watereos_api/figures.py`, `watereos_api/routes/figures.py`
- Create: `tests/test_api_figures_eos_phase.py`

Port reference: `watereos_visualizer/callbacks/phase_diagram.py` — `_CURVE_STYLES` (per-curve color/dash/width: binodal `#9333ea` solid; hdl/ldl spinodal `#ec4899` dash; tmd `#ffffff` dash; widom `#f97316` dashdot; ice_ih `#3b82f6`; ice_iii `#ef4444`; nuc_ih/nuc_iii `#9ca3af`; kauzmann `#22c55e`; LLCP marker `#9333ea`; triple point `#166534`) and the `replot` assembly that adds a trace per selected key from `compute_phase_diagram_data`. Reproduce `_CURVE_STYLES` **verbatim** (these are semantically meaningful) by reading the source.

- [ ] **Step 1: Write the failing test** — `tests/test_api_figures_eos_phase.py`:

```python
import time
from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())


def test_eos_phase_returns_requested_curves():
    r = client.post("/api/figures/eos-phase-diagram", json={
        "model": "duska2020",
        "show": ["binodal", "hdl_spinodal", "ldl_spinodal", "LLCP"],
        "theme": "dark"})
    assert r.status_code == 200
    fig = r.json()["figure"]
    names = " ".join(str(t.get("name", "")) for t in fig["data"]).lower()
    assert "binodal" in names and "spinodal" in names
    assert any(t.get("mode") == "markers" for t in fig["data"])  # LLCP


def test_eos_phase_cached_second_call_faster_and_identical():
    body = {"model": "duska2020", "show": ["binodal"], "theme": "dark"}
    t0 = time.perf_counter()
    a = client.post("/api/figures/eos-phase-diagram", json=body).json()
    t1 = time.perf_counter()
    b = client.post("/api/figures/eos-phase-diagram", json=body).json()
    t2 = time.perf_counter()
    assert a == b                                  # byte-identical
    assert (t2 - t1) <= (t1 - t0)                  # cache hit not slower


def test_eos_phase_unknown_model_404():
    r = client.post("/api/figures/eos-phase-diagram",
                    json={"model": "nope", "show": ["binodal"]})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_eos_phase.py -v`
Expected: FAIL — endpoint 404.

- [ ] **Step 3: Implement** — `watereos_api/cache.py`:

```python
"""Process-local memoization for slow computes."""
import json
import threading

_LOCK = threading.Lock()
_STORE: dict[str, object] = {}


def memoize(key_parts, producer):
    key = json.dumps(key_parts, sort_keys=True, default=str)
    with _LOCK:
        if key in _STORE:
            return _STORE[key]
    value = producer()                 # compute outside the lock
    with _LOCK:
        _STORE.setdefault(key, value)
        return _STORE[key]
```

Append to `watereos_api/figures.py` (reproduce `_CURVE_STYLES` from the source file verbatim):
```python
from watereos.computation import compute_phase_diagram_data

# Verbatim from watereos_visualizer/callbacks/phase_diagram.py::_CURVE_STYLES
_PD_STYLES = {
    "binodal":      dict(color="#9333ea", dash="solid", width=2),
    "hdl_spinodal": dict(color="#ec4899", dash="dash", width=2),
    "ldl_spinodal": dict(color="#ec4899", dash="dash", width=2),
    "tmd":          dict(color="#ffffff", dash="dash", width=2),
    "widom":        dict(color="#f97316", dash="dashdot", width=2),
    "ice_ih":       dict(color="#3b82f6", dash="solid", width=2),
    "ice_iii":      dict(color="#ef4444", dash="solid", width=2),
    "nuc_ih":       dict(color="#9ca3af", dash="solid", width=1.5),
    "nuc_iii":      dict(color="#9ca3af", dash="dash", width=1.5),
    "kauzmann":     dict(color="#22c55e", dash="solid", width=2),
}
_PD_KEYMAP = {
    "ice_ih": "ice_ih_liquidus", "ice_iii": "ice_iii_liquidus",
    "nuc_ih": "nucleation_ih", "nuc_iii": "nucleation_iii",
}


def build_eos_phase_figure(*, model, show, theme):
    pd = compute_phase_diagram_data(model)
    fig = go.Figure()
    for key in show:
        if key == "LLCP":
            llcp = pd.get("LLCP") or {}
            if llcp.get("T_K") is not None:
                fig.add_trace(go.Scatter(
                    x=[float(llcp["T_K"])], y=[float(llcp["p_MPa"])],
                    mode="markers", name="LLCP",
                    marker=dict(color="#9333ea", size=12,
                                line=dict(width=1, color="white"))))
            continue
        data_key = _PD_KEYMAP.get(key, key)
        d = pd.get(data_key) or {}
        if d.get("T_K") is None:
            continue
        st = _PD_STYLES.get(key, dict(color="#888", dash="solid", width=2))
        fig.add_trace(go.Scatter(
            x=list(d["T_K"]), y=list(d["p_MPa"]), mode="lines", name=key,
            line=dict(color=st["color"], dash=st["dash"], width=st["width"])))
    tp = pd.get("triple_point") or {}
    if tp.get("T_K") is not None and "ice_iii" in show:
        fig.add_trace(go.Scatter(
            x=[float(tp["T_K"])], y=[float(tp["p_MPa"])], mode="markers",
            name="Triple point",
            marker=dict(color="#166534", size=12, symbol="square",
                        line=dict(width=1, color="white"))))
    fig.update_layout(xaxis_title="Temperature [K]",
                      yaxis_title="Pressure [MPa]")
    apply_theme(fig, theme)
    return fig
```

> NOTE TO IMPLEMENTER: confirm the `compute_phase_diagram_data` return keys (spec lists `LLCP`, `hdl_spinodal`, `ldl_spinodal`, `binodal`, `tmd`, `widom`, `ice_ih_liquidus`, `ice_iii_liquidus`, `nucleation_ih`, `nucleation_iii`, `kauzmann`, `triple_point`, each with `T_K`/`p_MPa`). If a curve uses a nested shape (e.g. `spinodal.T_upper/p_array`), adapt the accessor while keeping `_PD_STYLES` colors verbatim. Cross-check one curve's array equals the raw `compute_phase_diagram_data` output (add an inline assert in the test if shapes differ).

Append to `watereos_api/routes/figures.py`:
```python
from watereos_api.schemas import EosPhaseRequest
from watereos_api.cache import memoize


@router.post("/eos-phase-diagram")
def eos_phase(req: EosPhaseRequest):
    _check_model(req.model)
    fig = figures.build_eos_phase_figure(
        model=req.model, show=list(req.show), theme=req.theme)
    payload = {"figure": figure_to_jsonable(fig), "warnings": []}
    return memoize(["eos", req.model, sorted(req.show), req.theme],
                   lambda: payload)
```

- [ ] **Step 4: Run test, expect PASS** (read `compute_phase_diagram_data` first)

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_eos_phase.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/cache.py watereos_api/figures.py watereos_api/routes/figures.py tests/test_api_figures_eos_phase.py
git commit -m "feat(api): add EoS phase diagram endpoint with caching"
```

---

## Task 12: `/api/figures/h2o-phase-diagram`

**Files:**
- Modify: `watereos_api/figures.py`, `watereos_api/routes/figures.py`
- Create: `tests/test_api_figures_h2o.py`

Port reference: `watereos/tv_phase_diagram.py` already provides Dash-independent figure builders `plot_tv_phase_diagram_plotly(diagram, V_min, V_max, T_min, T_max)`, `plot_tp_phase_diagram_plotly(diagram, T_min, T_max, P_min, P_max)`, `plot_ptv_phase_diagram_plotly(diagram, T_stride, n_pts_per_phase, V_min, V_max, P_max)`. The diagram itself comes from `compute_tv_phase_diagram(...)` (heavy ~seconds). Reuse these directly; only re-theme via `apply_theme`.

- [ ] **Step 1: Write the failing test** — `tests/test_api_figures_h2o.py`:

```python
import time
from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())


def test_h2o_tv_returns_figure():
    r = client.post("/api/figures/h2o-phase-diagram",
                    json={"projection": "tv", "theme": "dark"})
    assert r.status_code == 200
    fig = r.json()["figure"]
    assert len(fig["data"]) >= 1
    assert fig["layout"]["paper_bgcolor"] == "#111114"


def test_h2o_projection_switch_and_cache():
    t0 = time.perf_counter()
    a = client.post("/api/figures/h2o-phase-diagram",
                    json={"projection": "tp", "theme": "dark"}).json()
    t1 = time.perf_counter()
    b = client.post("/api/figures/h2o-phase-diagram",
                    json={"projection": "tp", "theme": "dark"}).json()
    t2 = time.perf_counter()
    assert a == b and (t2 - t1) <= (t1 - t0)
    c = client.post("/api/figures/h2o-phase-diagram",
                    json={"projection": "ptv", "theme": "dark"})
    assert c.status_code == 200
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_h2o.py -v`
Expected: FAIL — endpoint 404.

- [ ] **Step 3: Implement** — append to `watereos_api/figures.py`:

```python
from watereos.tv_phase_diagram import (
    compute_tv_phase_diagram, plot_tv_phase_diagram_plotly,
    plot_tp_phase_diagram_plotly, plot_ptv_phase_diagram_plotly,
)

_DIAGRAM_CACHE = {}


def _get_diagram():
    if "d" not in _DIAGRAM_CACHE:
        _DIAGRAM_CACHE["d"] = compute_tv_phase_diagram(verbose=False)
    return _DIAGRAM_CACHE["d"]


def build_h2o_figure(*, projection, V_range, T_range, P_range, theme):
    diagram = _get_diagram()
    if projection == "tv":
        v0, v1 = (V_range or [7e-4, 1.1e-3])
        t0, t1 = (T_range or [190, 300])
        fig = plot_tv_phase_diagram_plotly(diagram, V_min=v0, V_max=v1,
                                           T_min=t0, T_max=t1)
    elif projection == "tp":
        t0, t1 = (T_range or [190, 300])
        p0, p1 = (P_range or [1e-4, 1000])
        fig = plot_tp_phase_diagram_plotly(diagram, T_min=t0, T_max=t1,
                                           P_min=p0, P_max=p1)
    else:  # ptv
        v0, v1 = (V_range or [7e-4, 1.1e-3])
        p1 = (P_range or [0, 1000])[1]
        fig = plot_ptv_phase_diagram_plotly(diagram, T_stride=4,
                                            n_pts_per_phase=80,
                                            V_min=v0, V_max=v1, P_max=p1)
    apply_theme(fig, theme)
    return fig
```

> NOTE TO IMPLEMENTER: confirm the exact parameter names/signatures of the three `plot_*_phase_diagram_plotly` functions and `compute_tv_phase_diagram` by reading `watereos/tv_phase_diagram.py`; adjust kwargs to match. `apply_theme` must not erase scientific traces — it only updates layout/axes; verify the figure still has its data traces after theming.

Append to `watereos_api/routes/figures.py`:
```python
from watereos_api.schemas import H2OPhaseRequest


@router.post("/h2o-phase-diagram")
def h2o_phase(req: H2OPhaseRequest):
    fig = figures.build_h2o_figure(
        projection=req.projection, V_range=req.V_range,
        T_range=req.T_range, P_range=req.P_range, theme=req.theme)
    payload = {"figure": figure_to_jsonable(fig), "warnings": []}
    return memoize(["h2o", req.projection, req.V_range, req.T_range,
                    req.P_range, req.theme], lambda: payload)
```

- [ ] **Step 4: Run test, expect PASS** (read `watereos/tv_phase_diagram.py` first; this test may take ~10s on first compute)

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_figures_h2o.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watereos_api/figures.py watereos_api/routes/figures.py tests/test_api_figures_h2o.py
git commit -m "feat(api): add H2O phase diagram endpoint"
```

---

## Task 13: Error handling, CORS finalization, app import check

**Files:**
- Modify: `watereos_api/app.py`
- Create: `tests/test_api_errors.py`

- [ ] **Step 1: Write the failing test** — `tests/test_api_errors.py`:

```python
from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())


def test_validation_error_is_422():
    r = client.post("/api/figures/curves", json={"model": "duska2020"})
    assert r.status_code == 422


def test_unknown_route_404():
    assert client.get("/api/does-not-exist").status_code == 404


def test_unknown_model_is_404_not_500():
    r = client.post("/api/point",
                    json={"model_keys": ["bogus"], "T_K": 273.0, "P_MPa": 0.1})
    assert r.status_code == 404


def test_internal_error_is_safe_500(monkeypatch):
    import watereos_api.figures as figmod

    def boom(**kw):
        raise RuntimeError("explode-with-secret")

    monkeypatch.setattr(figmod, "build_curves_figure", boom)
    r = client.post("/api/figures/curves", json={
        "model": "duska2020", "property": "rho",
        "T_range": [200, 300], "P_range": [0.1, 200]})
    assert r.status_code == 500
    assert "explode-with-secret" not in r.text     # no leak
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_errors.py -v`
Expected: FAIL — `test_internal_error_is_safe_500` fails (default FastAPI 500 may differ / leak).

- [ ] **Step 3: Implement** — replace `watereos_api/app.py` `create_app` body to add a global exception handler (keep existing imports/CORS):

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from watereos_api.routes import api_router

_log = logging.getLogger("watereos_api")


def create_app() -> FastAPI:
    app = FastAPI(title="waterEoS API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        _log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error"})

    return app


app = create_app()
```

(`HTTPException` 404/`RequestValidationError` 422 keep FastAPI defaults; the catch-all only converts unexpected errors to a safe 500. Note: FastAPI invokes the `Exception` handler but still re-raises `HTTPException` via its own handler, so 404/422 are unaffected.)

- [ ] **Step 4: Run test, expect PASS**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/test_api_errors.py -v`
Expected: PASS.

- [ ] **Step 5: Verify the app boots under uvicorn import**

Run: `cd "<worktree>" && C:/Python314/python.exe -c "import uvicorn; from watereos_api.app import app; print('app ok', len(app.routes))"`
Expected: `app ok <N>` with no error.

- [ ] **Step 6: Commit**

```bash
git add watereos_api/app.py tests/test_api_errors.py
git commit -m "feat(api): safe error handling and exception handler"
```

---

## Task 14: Full verification

- [ ] **Step 1: Run the entire test suite**

Run: `cd "<worktree>" && C:/Python314/python.exe -m pytest tests/ -q`
Expected: all pass — the new `tests/test_api_*.py` + `tests/test_units_relocation.py` plus the pre-existing backend suite (no regressions; the units shim keeps `watereos_visualizer` working).

- [ ] **Step 2: Confirm the API serves end-to-end (manual smoke, optional)**

Run (background): `cd "<worktree>" && C:/Python314/python.exe -m uvicorn watereos_api.app:app --port 8060`
Then: `curl -s http://127.0.0.1:8060/api/health` → `{"status":"ok",...}`; `curl -s -X POST http://127.0.0.1:8060/api/point -H "Content-Type: application/json" -d "{\"model_keys\":[\"duska2020\"],\"T_K\":273.15,\"P_MPa\":0.1}"` → JSON results. Stop the server afterward.

- [ ] **Step 3: Commit any fixups (only if needed)**

```bash
git add -A -- watereos_api tests
git commit -m "test(api): verification fixups"
```
(Skip if nothing changed. Never `git add -A` at repo root — only the `watereos_api`/`tests` pathspecs.)

---

## Self-Review

**Spec coverage (spec §):**
- §4.1 package layout → Tasks 1,3,4,5,7,8,11; §4.2(1) units move+shim → Task 2; §4.2(2) figure port → Tasks 8–12; §4.2(3) theming → Task 4. ✓
- §5.1 health → T1; §5.2 metadata → T6; §5.3 curves → T8; §5.4/5.5 surface2d/3d → T9; §5.6 compare → T10; §5.7 eos-phase → T11; §5.8 h2o-phase → T12; §5.9 point → T7. ✓
- §6.1 serialization → T3; §6.2 units conversion → T2/T7/T8/T9 (`convert_array`/`get_factor`); §6.3 validation/warnings → T5/T7/T13; §6.4 caching → T11/T12; §6.5 CORS → T1/T13; §6.6 error handling → T13. ✓
- §7 testing (TestClient, ground-truth parity, metadata/point parity, unit conversion, serialization NaN, caching, regression) → tests across T2–T14. ✓
- §2 non-goals (FE/deploy) untouched. ✓

**Placeholder scan:** No "TBD"/"add error handling"-style placeholders. The three `NOTE TO IMPLEMENTER` blocks are not placeholders — they instruct reading a named source file to confirm exact return-dict keys before the parity test is run, with explicit fallback guidance; required because reproducing validated scientific plotting must be verified against the real `watereos.computation` output, which the parity tests enforce. Every code step contains complete code.

**Type/name consistency:** `create_app`/`app` (T1,T13); `figure_to_jsonable` (T3,T8+); `apply_theme`/`CURVE_PALETTE`/`THEMES` (T4,T8–12); `memoize` (T11,T12); schema classes `CurvesRequest`/`SurfaceRequest`/`CompareRequest`/`EosPhaseRequest`/`H2OPhaseRequest`/`PointRequest` (T5 → consumed T7–12); `watereos.units` API `get_factor`/`convert_array`/`display_label` (T2 → used T7–10); router registration kept consistent (T1→T7→T8). Endpoint paths match spec §5. ✓

**Scope:** Sub-project A only; B (Vite/React FE) and C (deploy + retire Dash) explicitly out, per spec §2/§5(A5).
