from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    try:
        from importlib.metadata import version
        ver = version("watereos")
    except Exception:
        ver = "unknown"
    return {"status": "ok", "version": ver}


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
