"""Tests for the SPA static-serving behavior of watereos_api.app.

Verifies three things:
  - With no dist directory present, the app continues to expose /api/*
    routes and returns 404 (not 500) for unrelated paths — the dev-mode
    contract.
  - With a fake dist directory pointed at via WATEREOS_WEB_DIST, the
    /assets mount serves hashed files and an unknown path returns the
    React shell (index.html) via the SPA fallback.
  - The fallback never serves files outside the dist tree (path-
    traversal guard).
"""

import importlib
import os
import textwrap
from pathlib import Path

from fastapi.testclient import TestClient


def _reload_app():
    """Re-import watereos_api.app so create_app() picks up env changes."""
    import watereos_api.app as app_module
    importlib.reload(app_module)
    return app_module


def test_dev_mode_no_dist(monkeypatch, tmp_path):
    # Point at an empty directory so the dist mount is definitely off.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("WATEREOS_WEB_DIST", str(empty))
    app_module = _reload_app()
    client = TestClient(app_module.create_app())

    # API still works.
    r = client.get("/api/health")
    assert r.status_code == 200

    # Unrelated paths 404 (no SPA fallback was registered).
    r = client.get("/some-spa-route")
    assert r.status_code == 404


def test_prod_mode_serves_index_and_assets(monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body>SHELL</body></html>",
        encoding="utf-8",
    )
    (dist / "assets" / "index-abc.js").write_text(
        "console.log('hi');", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")

    monkeypatch.setenv("WATEREOS_WEB_DIST", str(dist))
    app_module = _reload_app()
    client = TestClient(app_module.create_app())

    # API still works.
    r = client.get("/api/health")
    assert r.status_code == 200

    # Hashed asset is served from the assets mount.
    r = client.get("/assets/index-abc.js")
    assert r.status_code == 200
    assert "console.log" in r.text

    # Root returns the shell.
    r = client.get("/")
    assert r.status_code == 200
    assert "SHELL" in r.text

    # Unknown path also returns the shell (SPA history fallback).
    r = client.get("/property-explorer")
    assert r.status_code == 200
    assert "SHELL" in r.text

    # A real top-level file (favicon) is served directly.
    r = client.get("/favicon.svg")
    assert r.status_code == 200
    assert r.text.startswith("<svg")


def test_spa_fallback_blocks_path_traversal(monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("SHELL", encoding="utf-8")
    # A sibling secret file that must NOT be reachable.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")

    monkeypatch.setenv("WATEREOS_WEB_DIST", str(dist))
    app_module = _reload_app()
    client = TestClient(app_module.create_app())

    # The handler resolves the candidate path and rejects anything
    # outside dist; the traversal request falls back to the shell.
    r = client.get("/../secret.txt")
    assert r.status_code == 200
    assert "TOPSECRET" not in r.text
