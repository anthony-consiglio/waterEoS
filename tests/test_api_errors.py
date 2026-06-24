from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app(), raise_server_exceptions=False)


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
    assert "explode-with-secret" not in r.text
