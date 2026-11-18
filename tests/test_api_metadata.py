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


def test_metadata_serves_version_and_references():
    import re
    from watereos.references import REFERENCE_GROUPS

    body = client.get("/api/metadata").json()
    assert re.match(r"^\d+\.\d+\.\d+", body["version"])
    refs, groups = body["references"], body["reference_groups"]
    titles = [g["title"] for g in groups]
    assert titles[:len(REFERENCE_GROUPS)] == [title for title, _ in REFERENCE_GROUPS]
    assert titles[-1] == "EoS phase diagram reference points"
    for group in groups:
        for key in group["keys"]:
            assert refs[key]["short"] and refs[key]["citation"]
            assert refs[key]["doi"] or key == "bridge29"
    for model in body["models"]:
        assert model["references"] and all(key in refs for key in model["references"])
        assert "water_model" in model
