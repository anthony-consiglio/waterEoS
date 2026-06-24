import os
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from watereos_api.routes import api_router

_log = logging.getLogger("watereos_api")


def _resolve_web_dist() -> Path | None:
    """Return the watereos-web/dist directory if it exists, else None.

    Resolution rules:
      - If $WATEREOS_WEB_DIST is set it is the authoritative answer:
        return that path if it contains index.html, otherwise None. No
        fallback. This lets tests force dev-mode behaviour even when a
        real dist exists in the repo.
      - Otherwise try <repo-root>/watereos-web/dist (next to the package)
        and then <cwd>/watereos-web/dist.

    The mount is optional: in dev we serve the frontend via Vite on
    :5173 and proxy /api -> :8000, so a missing dist directory is not an
    error. In production (single-service deploy on Render) the build
    step runs `npm run build` and dist will be present.
    """
    env_path = os.getenv("WATEREOS_WEB_DIST")
    if env_path:
        p = Path(env_path)
        if p.is_dir() and (p / "index.html").is_file():
            return p
        return None
    pkg_root = Path(__file__).resolve().parent.parent
    for c in (pkg_root / "watereos-web" / "dist",
              Path.cwd() / "watereos-web" / "dist"):
        if c.is_dir() and (c / "index.html").is_file():
            return c
    return None


def _assert_backends_present() -> None:
    """Fail fast at startup if a backend the API depends on is missing.

    The web runtime intentionally ships without the upstream ``seafreeze``
    package — its functionality is covered by the in-process Rust
    evaluator (``watereos_rs``) built into the image. If that extension
    didn't compile (e.g. because the Dockerfile lost the Rust toolchain
    step), every water1/IAPWS95/ice request would crash with a confusing
    ImportError on first hit. Raising here surfaces the misconfiguration
    in the deploy logs instead.
    """
    try:
        import watereos_rs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "watereos_rs is required at runtime (the native SeaFreeze "
            "backend) but is not importable. Rebuild the image with the "
            "Rust toolchain present so setuptools-rust can compile the "
            "extension."
        ) from exc


def create_app() -> FastAPI:
    _assert_backends_present()
    app = FastAPI(title="waterEoS API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.getenv(
            "WATEREOS_API_ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",") if o.strip()],
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        _log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error"})

    # Mount the built React bundle when present. In dev the frontend is
    # served by Vite on a separate port, so this block is skipped.
    dist = _resolve_web_dist()
    if dist is not None:
        _log.info("serving frontend bundle from %s", dist)

        # Hashed asset files (immutable; safe for aggressive caching).
        assets_dir = dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets",
                      StaticFiles(directory=str(assets_dir)),
                      name="assets")

        index_path = dist / "index.html"

        # SPA fallback: any non-/api, non-/assets GET returns the React
        # shell so client-side routing works on a hard refresh. Registered
        # last so it doesn't shadow the API router or the /assets mount.
        @app.get("/{full_path:path}", include_in_schema=False)
        async def _spa_fallback(full_path: str):
            # Unknown /api/* paths must surface as 404, not the SPA shell
            # (otherwise typos in the JS client would silently render the
            # frontend HTML, which then fails JSON-decoding).
            if full_path.startswith("api/") or full_path == "api":
                return JSONResponse(status_code=404,
                                    content={"detail": "not found"})
            # Specific top-level static files (favicon, etc.) live next
            # to index.html and should be served directly.
            candidate = dist / full_path if full_path else None
            if (candidate
                    and candidate.is_file()
                    and candidate.resolve().is_relative_to(dist.resolve())):
                return FileResponse(candidate)
            return FileResponse(index_path)

    return app


app = create_app()
