# SentinelMCP gateway image (also runs the sandbox web services mockweb / exfil-sink).
# The gateway spawns the stdio tool servers (filesystem, sql_db, email_send, http_fetch and
# the deliberately poisoned test servers) as subprocesses inside this container.
#
#   docker build -t sentinel-mcp/gateway .

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY gateway gateway
COPY tools_servers tools_servers
COPY policies policies
COPY config config
COPY alembic.ini alembic.ini

FROM python:3.12-slim
# uid/gid 10001 matches the shell sandbox: the gateway must own the shell's 0600 socket.
RUN groupadd --system --gid 10001 sentinel \
 && useradd --system --uid 10001 --gid sentinel --home-dir /app --shell /usr/sbin/nologin sentinel \
 && mkdir -p /app/sandbox_runtime/fs /app/sandbox_runtime/db /app/sandbox_runtime/outbox \
             /app/sandbox_runtime/poison /run/sentinel-shell \
 && chown -R sentinel:sentinel /app /run/sentinel-shell \
 && chmod 0700 /run/sentinel-shell
COPY --from=build --chown=sentinel:sentinel /app /app
ENV PATH="/app/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTINEL_HOST=0.0.0.0 \
    SENTINEL_PORT=8000 \
    SENTINEL_SERVERS_PATH=config/servers.compose.yaml
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=12 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status == 200 else 1)"
CMD ["python", "-m", "gateway"]
