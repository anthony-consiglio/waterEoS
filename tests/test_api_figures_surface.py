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
