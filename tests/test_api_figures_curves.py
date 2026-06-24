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
