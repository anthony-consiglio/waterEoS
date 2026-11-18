import time
from fastapi.testclient import TestClient
from watereos_api.app import create_app
from watereos.eos_phase_diagram import DEFAULT_LAYERS, LAYERS, PLOT_P_RANGE, T_RANGE

client = TestClient(create_app())


def _names(fig):
    return {str(t.get("name", "")) for t in fig["data"]}


def test_eos_phase_returns_requested_curves():
    r = client.post("/api/figures/eos-phase-diagram", json={
        "model": "duska2020",
        "show": ["binodal", "hdl_spinodal", "ldl_spinodal", "LLCP"],
        "theme": "dark"})
    assert r.status_code == 200
    fig = r.json()["figure"]
    assert _names(fig) == {LAYERS[k] for k in ("binodal", "hdl_spinodal", "ldl_spinodal", "LLCP")}
    assert any(t.get("mode") == "markers" for t in fig["data"])  # LLCP
    assert any("critical point" in a["text"] for a in fig["layout"]["annotations"])


def test_eos_phase_default_shows_every_layer_but_kauzmann():
    r = client.post("/api/figures/eos-phase-diagram", json={"model": "caupin2019", "theme": "light"})
    assert r.status_code == 200
    fig = r.json()["figure"]
    names = _names(fig)
    assert {LAYERS[k] for k in DEFAULT_LAYERS} <= names
    assert LAYERS["kauzmann"] not in names
    assert "Caupin and Anisimov (2019)" in fig["layout"]["title"]["text"]


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


def test_eos_phase_model_without_phase_diagram_404():
    r = client.post("/api/figures/eos-phase-diagram",
                    json={"model": "water1", "show": ["binodal"]})
    assert r.status_code == 404


def test_eos_phase_accepts_older_layer_names():
    # The Info page hero still asks for these.
    r = client.post("/api/figures/eos-phase-diagram", json={
        "model": "duska2020", "show": ["tmd", "nuc_ih", "nuc_iii", "ice_ih"], "theme": "dark"})
    assert r.status_code == 200
    names = _names(r.json()["figure"])
    assert {LAYERS["density"], LAYERS["nucleation"], LAYERS["ice_ih"]} <= names


def test_eos_phase_axis_clamp_when_auto_limits_false():
    body = {"model": "duska2020", "show": ["binodal"],
            "auto_limits": False, "T_range": [230, 280], "P_range": [50, 150],
            "theme": "dark"}
    fig = client.post("/api/figures/eos-phase-diagram", json=body).json()["figure"]
    assert list(fig["layout"]["xaxis"]["range"]) == [230, 280]
    assert list(fig["layout"]["yaxis"]["range"]) == [50, 150]
    # auto_limits=True (default) keeps the diagram's standard window.
    body2 = {"model": "duska2020", "show": ["binodal"], "theme": "dark"}
    fig2 = client.post("/api/figures/eos-phase-diagram", json=body2).json()["figure"]
    assert list(fig2["layout"]["xaxis"]["range"]) == list(T_RANGE)
    assert list(fig2["layout"]["yaxis"]["range"]) == list(PLOT_P_RANGE)
