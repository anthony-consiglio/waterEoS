import time
from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())


def test_eos_phase_returns_requested_curves():
    r = client.post("/api/figures/eos-phase-diagram", json={
        "model": "duska2020",
        "show": ["binodal", "hdl_spinodal", "ldl_spinodal", "LLCP"],
        "theme": "dark"})
    assert r.status_code == 200
    fig = r.json()["figure"]
    names = " ".join(str(t.get("name", "")) for t in fig["data"]).lower()
    assert "binodal" in names and "spinodal" in names
    assert any(t.get("mode") == "markers" for t in fig["data"])  # LLCP


def test_eos_phase_cached_second_call_faster_and_identical():
    body = {"model": "duska2020", "show": ["binodal"], "theme": "dark"}
    t0 = time.perf_counter()
    a = client.post("/api/figures/eos-phase-diagram", json=body).json()
    t1 = time.perf_counter()
    b = client.post("/api/figures/eos-phase-diagram", json=body).json()
    t2 = time.perf_counter()
    assert a == b
    assert (t2 - t1) <= (t1 - t0)


def test_eos_phase_unknown_model_404():
    r = client.post("/api/figures/eos-phase-diagram",
                    json={"model": "nope", "show": ["binodal"]})
    assert r.status_code == 404


def test_eos_phase_triple_point_for_ice_ih_only():
    r = client.post("/api/figures/eos-phase-diagram", json={
        "model": "duska2020", "show": ["ice_ih"], "theme": "dark"})
    assert r.status_code == 200
    fig = r.json()["figure"]
    triple = [t for t in fig["data"] if t.get("name") == "Triple point"]
    assert len(triple) == 1
    assert triple[0].get("mode") == "markers"
    # square marker (matches Dash _CURVE_STYLES triple-point symbol)
    assert triple[0]["marker"]["symbol"] == "square"


def test_eos_phase_axis_clamp_when_auto_limits_false():
    body = {"model": "duska2020", "show": ["binodal"],
            "auto_limits": False, "T_range": [230, 280], "P_range": [50, 150],
            "theme": "dark"}
    fig = client.post("/api/figures/eos-phase-diagram", json=body).json()["figure"]
    assert list(fig["layout"]["xaxis"]["range"]) == [230, 280]
    assert list(fig["layout"]["yaxis"]["range"]) == [50, 150]
    # auto_limits=True (default) should NOT set explicit ranges
    body2 = {"model": "duska2020", "show": ["binodal"], "theme": "dark"}
    fig2 = client.post("/api/figures/eos-phase-diagram", json=body2).json()["figure"]
    # the xaxis dict may not have a 'range' key at all
    assert "range" not in fig2["layout"].get("xaxis", {})
