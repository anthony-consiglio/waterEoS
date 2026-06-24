from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, conlist

Theme = Literal["dark", "light"]
Pair = conlist(float, min_length=2, max_length=2)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _UnitSettings(_StrictModel):
    unit_temperature: Optional[str] = None
    unit_pressure: Optional[str] = None
    unit_density: Optional[str] = None
    unit_volume: Optional[str] = None
    unit_energy: Optional[str] = None
    unit_entropy: Optional[str] = None
    unit_bulk_modulus: Optional[str] = None
    unit_viscosity: Optional[str] = None


class CurvesRequest(_StrictModel):
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


class SurfaceRequest(_StrictModel):
    model: str
    property: str
    T_range: Pair
    P_range: Pair
    n_points: int = Field(80, ge=10, le=400)
    colormap: str = "rdbu"
    # Overlay spinodal / binodal / LLCP on top of the surface. For 2D
    # heatmap mode the boundaries are drawn in T-P space; for 3D mode
    # the property value is evaluated along each boundary curve so the
    # overlay sits on the surface itself.
    show_phase_boundaries: bool = False
    theme: Theme = "dark"
    units: Optional[_UnitSettings] = None


class CompareRequest(_StrictModel):
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


class EosPhaseRequest(_StrictModel):
    model: str
    show: conlist(str, min_length=1) = Field(
        default_factory=lambda: ["binodal", "hdl_spinodal", "ldl_spinodal", "LLCP"])
    auto_limits: bool = True
    T_range: Optional[Pair] = None
    P_range: Optional[Pair] = None
    theme: Theme = "dark"


class H2OPhaseRequest(_StrictModel):
    projection: Literal["tv", "tp", "ptv"] = "tv"
    V_range: Optional[Pair] = None
    T_range: Optional[Pair] = None
    P_range: Optional[Pair] = None
    theme: Theme = "dark"


class PointRequest(_StrictModel):
    model_keys: conlist(str, min_length=1)
    T_K: float
    P_MPa: float
    units: Optional[_UnitSettings] = None
