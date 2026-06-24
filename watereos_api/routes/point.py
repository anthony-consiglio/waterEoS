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
            # defensive: the /point route already 404s on unknown keys,
            # but this helper is safe to call standalone.
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
                        "message": "; ".join(parts)})
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
