FROM python:3.12.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /bin/uv

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/tmp/uv-cache \
    TIKTOKEN_CACHE_DIR=/opt/tokenizer \
    HOME=/tmp

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY scripts ./scripts
COPY configs ./configs
COPY deploy ./deploy
COPY .test-cache/tiktoken/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 /opt/tokenizer/
RUN .venv/bin/python -c "import hashlib,pathlib; p=pathlib.Path('/opt/tokenizer/9b5ad71b2ce5302211f9c61530b329a4922fc6a4'); assert hashlib.sha256(p.read_bytes()).hexdigest() == '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'" \
    && chmod 644 /opt/tokenizer/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 \
    && mkdir -p /app/runs && chown 10001:10001 /app/runs
USER 10001:10001

EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers", "--no-access-log"]
