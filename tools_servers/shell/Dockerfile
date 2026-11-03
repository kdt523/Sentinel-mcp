# Hardened sandbox image for the SentinelMCP shell tool server.
# Build from the repo root:  docker build -f tools_servers/shell/Dockerfile -t sentinel-shell .
# Runtime hardening (network none, read-only rootfs, cap_drop, limits) is set in compose /
# `docker run`; see docker-compose.yml service `shell`.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY tools_servers/__init__.py tools_servers/__init__.py
COPY tools_servers/shell tools_servers/shell

FROM python:3.12-slim
RUN groupadd --system --gid 10001 sandbox \
 && useradd --system --uid 10001 --gid sandbox --home-dir /sandbox --shell /usr/sbin/nologin sandbox \
 && mkdir -p /sandbox && chown sandbox:sandbox /sandbox \
 && mkdir -p /run/sentinel-shell && chown sandbox:sandbox /run/sentinel-shell \
 && chmod 0700 /run/sentinel-shell
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SANDBOX_IN_CONTAINER=1 \
    SANDBOX_FS_ROOT=/sandbox
WORKDIR /sandbox
USER 10001:10001
ENTRYPOINT ["python", "-m", "tools_servers.shell"]
