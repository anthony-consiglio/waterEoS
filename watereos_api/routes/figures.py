from fastapi import APIRouter, HTTPException
from watereos_api.schemas import CompareRequest, CurvesRequest, EosPhaseRequest, H2OPhaseRequest, SurfaceRequest
from watereos_api.serialization import figure_to_jsonable
from watereos_api import figures
from watereos_api.cache import memoize
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


@router.post("/surface2d")
def surface2d(req: SurfaceRequest):
    _check_model(req.model)
    units = req.units.model_dump(exclude_none=True) if req.units else None
    fig = figures.build_surface2d_figure(
        model=req.model, prop=req.property, T_range=req.T_range,
        P_range=req.P_range, n_points=req.n_points, colormap=req.colormap,
        show_phase=req.show_phase_boundaries,
        theme=req.theme, units=units)
    return {"figure": figure_to_jsonable(fig), "warnings": []}


@router.post("/surface3d")
def surface3d(req: SurfaceRequest):
    _check_model(req.model)
    units = req.units.model_dump(exclude_none=True) if req.units else None
    fig = figures.build_surface3d_figure(
        model=req.model, prop=req.property, T_range=req.T_range,
        P_range=req.P_range, n_points=req.n_points, colormap=req.colormap,
        show_phase=req.show_phase_boundaries,
        theme=req.theme, units=units)
    return {"figure": figure_to_jsonable(fig), "warnings": []}


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


@router.post("/eos-phase-diagram")
def eos_phase(req: EosPhaseRequest):
    _check_model(req.model)

    def _build():
        fig = figures.build_eos_phase_figure(
            model=req.model, show=list(req.show), theme=req.theme,
            T_range=req.T_range, P_range=req.P_range, auto_limits=req.auto_limits)
        return {"figure": figure_to_jsonable(fig), "warnings": []}

    return memoize(["eos", req.model, sorted(req.show), req.theme,
                    req.auto_limits, req.T_range, req.P_range],
                   _build)


@router.post("/h2o-phase-diagram")
def h2o_phase(req: H2OPhaseRequest):
    def _build():
        fig = figures.build_h2o_figure(
            projection=req.projection, V_range=req.V_range,
            T_range=req.T_range, P_range=req.P_range, theme=req.theme)
        return {"figure": figure_to_jsonable(fig), "warnings": []}

    return memoize(["h2o", req.projection, req.V_range, req.T_range,
                    req.P_range, req.theme], _build)
