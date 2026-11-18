from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "version": _package_version()}


def _package_version():
    try:
        from importlib.metadata import version
        return version("watereos")
    except Exception:
        return "unknown"


def _references():
    """All references with DOIs, and their display groups, for the Info page."""
    from watereos.eos_phase_diagram import reference_points
    from watereos.references import REFERENCE_GROUPS, REFERENCES

    refs = {key: dict(entry) for key, entry in REFERENCES.items()}
    groups = [{"title": title, "keys": list(keys)} for title, keys in REFERENCE_GROUPS]
    points = []
    for key, entry in reference_points()["sources"].items():
        if key not in refs:
            refs[key] = {"short": entry.get("short", key), "citation": entry["citation"],
                         "doi": entry["doi"], "notes": []}
        points.append(key)
    groups.append({"title": "EoS phase diagram reference points", "keys": points})
    return refs, groups


@router.get("/metadata")
def metadata():
    from watereos.model_registry import (
        MODEL_ORDER, MODEL_REGISTRY, PROPERTY_LABELS, PROPERTY_UNITS,
    )
    from watereos.references import MODEL_REFERENCES
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
            "water_model": info.water_model,
            "references": MODEL_REFERENCES.get(key, []),
        })
    properties = {
        k: {"label": PROPERTY_LABELS.get(k, k), "unit": PROPERTY_UNITS.get(k, "")}
        for k in PROPERTY_LABELS
    }
    references, reference_groups = _references()
    return {
        "version": _package_version(),
        "models": models,
        "references": references,
        "reference_groups": reference_groups,
        "properties": properties,
        "units": {
            "options": UNIT_OPTIONS,
            "defaults": UNIT_DEFAULTS,
            "category_labels": CATEGORY_LABELS,
        },
    }
