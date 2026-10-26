"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.auth import require_library
from app.auth import router as auth_router
from app.config import get_settings
from app.database import init_db
from app.routers import budget, documents, feedback, query, traces
from app.services.application_limits import ApplicationBoundary, require_capacity
from app.services.application_schema import ready
from app.services.resilience import ProviderUnavailable
from app.services.tracing import TraceBoundary

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.database import engine

    settings = get_settings()
    try:
        if settings.run_db_init:
            init_db()
        if settings.application_startup_check and not ready(engine, settings.embedding_dim):
            raise RuntimeError("Database is not ready")
    except Exception:
        # Never expose credentials/SQL parameters or serve after a failed bootstrap.
        raise RuntimeError(
            "Application database startup failed; inspect readiness/migrations safely"
        ) from None
    yield


app = FastAPI(
    title="RAG Starter — FastAPI + pgvector",
    description=(
        "A small, production-shaped Retrieval-Augmented Generation API: "
        "ingest documents, embed & store them in Postgres/pgvector, then "
        "answer questions with cited, grounded responses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(budget.router)
app.include_router(traces.router)
app.include_router(feedback.router)
app.include_router(
    documents.router, dependencies=[Depends(require_library), Depends(require_capacity)]
)
app.include_router(query.router, dependencies=[Depends(require_library), Depends(require_capacity)])
app.add_middleware(ApplicationBoundary)
app.add_middleware(TraceBoundary)


@app.exception_handler(ProviderUnavailable)
async def provider_unavailable(request: Request, exc: ProviderUnavailable):
    return JSONResponse(status_code=503, content={"detail": exc.code})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/auth/"):
        return JSONResponse(status_code=422, content={"detail": "Invalid authentication request"})
    if request.url.path.startswith("/traces/"):
        return JSONResponse(status_code=422, content={"detail": "Invalid feedback/export request"})
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def private_responses(request: Request, call_next):
    """Never cache private documents, answers, identity or budget amounts."""
    response = await call_next(request)
    if request.url.path.startswith(
        ("/documents", "/query", "/auth", "/knowledge-bases", "/budget", "/traces")
    ):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Cookie"
    return response


@app.get("/", include_in_schema=False)
def home():
    """Serve the interactive chat demo UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
def readiness():
    from app.database import engine

    available = ready(engine, get_settings().embedding_dim)
    return JSONResponse(
        {"status": "ready" if available else "not_ready"},
        status_code=200 if available else 503,
        headers={"Cache-Control": "no-store"},
    )
