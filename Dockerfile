# Multi-stage build for the waterEoS web app.
#
# Stage 1 (web) installs npm deps and runs `vite build` to produce the
# static React bundle at /web/dist.
#
# Stage 2 (runtime) is a slim Python image with the FastAPI backend +
# the dist bundle copied in. gunicorn serves both /api/* and the SPA
# shell from one process. See watereos_api/app.py.
#
# Railway auto-builds this Dockerfile on each push to master.

# -----------------------------------------------------------------------------
# Stage 1: build the React bundle
# -----------------------------------------------------------------------------
FROM node:20-slim AS web
WORKDIR /web

# Install npm deps first against the lockfile so subsequent rebuilds
# that only touch source files reuse the cached layer. The .npmrc must
# be COPYed here too: it carries `legacy-peer-deps=true`, which we need
# because eslint-plugin-react@7.37.x has a stale peer-dep range that
# doesn't list eslint 10. Without it, npm ci dies with ERESOLVE.
COPY watereos-web/package.json watereos-web/package-lock.json watereos-web/.npmrc ./
RUN npm ci --no-audit --no-fund

# Pull in the rest of the frontend source and build.
COPY watereos-web/ ./
RUN npm run build
# Produces /web/dist/{index.html, assets/*, favicon.svg}

# -----------------------------------------------------------------------------
# Stage 2: Python runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app

# OS packages:
#   * gcc           : a couple of scientific-python wheels still build
#                     from source on slim (no prebuilt manylinux wheel).
#   * libc6-dev     : ships Scrt1.o/crti.o/crtn.o and libc/libm/libpthread
#                     etc. that rustc needs at link time. python:3.11-slim
#                     installs gcc-only when you ask for gcc — without this
#                     the cdylib link step fails with `cannot open Scrt1.o`
#                     and `unable to find library -lc`.
#   * curl + ca-cert: needed by the rustup installer below.
# The whole apt cache is pruned after install to keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install a minimal Rust toolchain so setuptools-rust can build
# watereos_rs at pip-install time. The native module is no longer
# optional in practice: with seafreeze dropped from the runtime deps,
# watereos._seafreeze_native depends on watereos_rs to evaluate the
# eight bundled GLBF splines. The toolchain is wiped right after the
# build to keep the final image lean (~600 MB transient, 0 retained).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"

# Bring in the backend source first. requirements-web.txt contains a
# bare `.` line (install the local watereos package), so pyproject.toml
# has to exist on disk before pip runs. setuptools-rust will compile
# the watereos_rs cdylib in release mode via the toolchain installed
# above, then setuptools wires it into the package's import path.
# .dockerignore keeps node_modules, the pre-existing local dist, the
# .git directory, etc. out of the context.
COPY . .

# Drop in the built bundle from stage 1. This overrides any stale dist
# that slipped through .dockerignore.
COPY --from=web /web/dist ./watereos-web/dist

# Install Python deps (this also triggers the Rust build for
# watereos_rs because requirements-web.txt's bare `.` line installs
# the local package). Verify the extension actually compiled before
# we wipe the toolchain — setuptools-rust's `optional = true` setting
# would otherwise silently install a Rust-less wheel if cargo blew up
# inside the container, and the runtime startup assertion would only
# fire at first request.
#
# `-v` on pip surfaces cargo's stdout/stderr in the Railway build log,
# which is what we need to debug the silent skip we hit on 2026-05-24.
# The `cargo --version` line is a sanity check that the rustup install
# above actually put cargo on PATH for this RUN's shell.
RUN cargo --version \
    && pip install -v --no-cache-dir -r requirements-web.txt \
    && python -c "import watereos_rs; assert hasattr(watereos_rs, 'sf_phases'), 'watereos_rs imported without seafreeze entry points'"

# Wipe the toolchain in a separate RUN so the trailing `|| true` from
# the __pycache__ cleanup (which exists to swallow find's noise on
# /proc) can't accidentally mask a real failure in the verification
# above.
RUN rm -rf /root/.cargo /root/.rustup ./target ./build ./*.egg-info \
    && find / -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

ENV PYTHONPATH=. \
    PYTHONUNBUFFERED=1 \
    WATEREOS_API_ALLOWED_ORIGINS=https://watereos-visualizer.up.railway.app

# Railway injects $PORT. gunicorn supervises one uvicorn worker on the
# free tier; bump -w on a paid plan if you want more concurrency.
CMD ["sh", "-c", "gunicorn watereos_api.app:app -k uvicorn.workers.UvicornWorker -w 1 --bind 0.0.0.0:$PORT --timeout 120 --access-logfile - --error-logfile -"]
