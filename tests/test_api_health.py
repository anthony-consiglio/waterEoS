from fastapi.testclient import TestClient
from watereos_api.app import create_app

client = TestClient(create_app())


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]
    assert body["version"] != "unknown"
