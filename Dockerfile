# Serving image only (ARCHITECTURE.md §3/§8). Built and run exclusively in
# CI -- this machine has no Docker (see GIT_WORKFLOW.md / ROADMAP Sprint 6).
#
# Two stages so the final image never carries pip's build cache or a
# compiler toolchain: stage 1 resolves requirements-serve.txt into a venv,
# stage 2 copies only that venv plus the serving code and the committed
# model bundle.

FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

FROM python:3.12-slim

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
# api/main.py resolves PROJECT_ROOT as two parents up from itself
# (src/api/main.py -> src -> /app), so model_bundle/ must sit here too.
COPY src/ src/
COPY model_bundle/ model_bundle/

USER appuser

# Cloud Run injects $PORT (Sprint 7); 8080 is its own default and what
# the container listens on when run standalone (docker-compose, CI).
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
