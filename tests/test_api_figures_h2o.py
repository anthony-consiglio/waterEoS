import time
from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())


def test_h2o_tv_returns_figure():
    r = client.post("/api/figures/h2o-phase-diagram",
                    json={"projection": "tv", "theme": "dark"})
    assert r.status_code == 200
    fig = r.json()["figure"]
    assert len(fig["data"]) >= 1
    assert fig["layout"]["paper_bgcolor"] == "#111114"


def test_h2o_projection_switch_and_cache():
    t0 = time.perf_counter()
    a = client.post("/api/figures/h2o-phase-diagram",
                    json={"projection": "tp", "theme": "dark"}).json()
    t1 = time.perf_counter()
    b = client.post("/api/figures/h2o-phase-diagram",
                    json={"projection": "tp", "theme": "dark"}).json()
    t2 = time.perf_counter()
    assert a == b and (t2 - t1) <= (t1 - t0)
    c = client.post("/api/figures/h2o-phase-diagram",
                    json={"projection": "ptv", "theme": "dark"})
    assert c.status_code == 200
