from fastapi.testclient import TestClient
from watereos_api.app import create_app
from watereos.computation import compute_point_properties

client = TestClient(create_app())


def test_point_matches_ground_truth():
    payload = {"model_keys": ["duska2020"], "T_K": 273.15, "P_MPa": 0.1}
    r = client.post("/api/point", json=payload)
    assert r.status_code == 200
    body = r.json()
    truth = compute_point_properties(["duska2020"], 273.15, 0.1)["duska2020"]
    got = body["results"]["duska2020"]
    assert abs(got["rho"] - truth["rho"]) < 1e-6 * abs(truth["rho"])


def test_point_out_of_range_warns():
    # duska2020 valid T is [200,370] K; 500 K is out of range
    r = client.post("/api/point",
                     json={"model_keys": ["duska2020"], "T_K": 500.0, "P_MPa": 0.1})
    assert r.status_code == 200
    warnings = r.json()["warnings"]
    assert len(warnings) == 1
    w = warnings[0]
    assert w["model"] == "Duska (2020)"
    assert "T 500.00 K outside" in w["message"]
    # message must NOT redundantly repeat the display name
    assert not w["message"].startswith("Duska (2020):")


def test_point_unit_conversion():
    r = client.post("/api/point", json={
        "model_keys": ["duska2020"], "T_K": 273.15, "P_MPa": 0.1,
        "units": {"unit_density": "g/cm³"}})
    si = compute_point_properties(["duska2020"], 273.15, 0.1)["duska2020"]["rho"]
    got = r.json()["results"]["duska2020"]["rho"]
    assert abs(got - si * 1e-3) < 1e-9 * abs(si * 1e-3)
