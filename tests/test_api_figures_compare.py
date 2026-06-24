from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())
BASE = dict(model_keys=["duska2020", "holten2014"], property="rho",
            T_range=[210, 290], P_range=[0.1, 200], n_curves=3,
            n_points=80, isobar_mode=True, theme="dark")


def test_compare_overlay_single_axes():
    body = client.post("/api/figures/compare",
                       json={**BASE, "layout": "overlay"}).json()
    fig = body["figure"]
    assert "xaxis2" not in fig["layout"]
    traces = fig["data"]
    # 2 models * 3 curves = 6 total traces
    assert len(traces) == 6
    # Legend deduplicated: exactly one visible legend entry per model
    visible = [t for t in traces if t.get("showlegend", True)]
    assert len(visible) == 2


def test_compare_sidebyside_subplots():
    fig = client.post("/api/figures/compare",
                      json={**BASE, "layout": "sidebyside"}).json()["figure"]
    assert "xaxis2" in fig["layout"]


def test_compare_unknown_model_404():
    r = client.post("/api/figures/compare",
                    json={**BASE, "model_keys": ["duska2020", "bogus"]})
    assert r.status_code == 404
