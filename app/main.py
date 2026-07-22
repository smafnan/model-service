"""The FastAPI service that serves the sentiment model.

Operational features a real service needs, all here:
  * **/health** — a liveness/readiness probe (does the model load? what classes?)
    so an orchestrator (Docker, k8s) can tell if the container is ready.
  * **request logging middleware** — every request logs a unique id, method,
    path, status, and latency in milliseconds.
  * **input validation** — Pydantic rejects bad payloads with a 422.
  * **lifespan model loading** — the model is loaded once at startup, not per
    request.
  * **CORS allowlist** — origins come from `ALLOWED_ORIGINS` (comma-separated),
    defaulting to the local dev origins so `docker compose up` / `uvicorn
    --reload` keep working with no env set.
  * **optional API-key auth** — set `API_KEY` to require a matching
    `X-API-Key` header on `/predict*`; unset (the default) leaves the local
    demo open.
  * **in-memory per-client rate limiting** — a fixed-window limiter guards
    `/predict*`, configurable via `RATE_LIMIT_PER_MINUTE`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .model import SentimentModel
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("model-service")

# Holds the loaded model for the app's lifetime.
state: dict = {"model": None}

# --- CORS -------------------------------------------------------------- #
# Comma-separated allowlist from the environment; falls back to the app's
# real dev origins so local dev works unconfigured.
_DEFAULT_ORIGINS = ["http://localhost:5173", "http://localhost:8000"]
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS")
ALLOWED_ORIGINS = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else _DEFAULT_ORIGINS
)

# --- optional API-key auth ---------------------------------------------- #
# Auth is only enforced when API_KEY is set, so the local demo stays open.
API_KEY = os.environ.get("API_KEY")


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- simple in-memory rate limiting -------------------------------------- #
class RateLimiter:
    """Fixed-window rate limiter, keyed per client, all in memory.

    Good enough for a single-process demo/small deployment; a multi-worker or
    multi-instance deployment would need a shared store (e.g. Redis) instead.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window_seconds
            while hits and hits[0] < cutoff:
                hits.pop(0)
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
rate_limiter = RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window_seconds=60.0)


def enforce_rate_limit(request: Request) -> None:
    client_key = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded, slow down")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load (or train) the model once when the service starts.
    logger.info("Loading model ...")
    state["model"] = SentimentModel.load_or_train()
    logger.info("Model ready with classes: %s", state["model"].classes)
    yield
    state["model"] = None  # cleanup on shutdown


app = FastAPI(
    title="Sentiment Model Service",
    version="1.0.0",
    description="A containerised sentiment classifier served over HTTP.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with a correlation id and latency."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "rid=%s %s %s -> %d (%.1f ms)",
        request_id, request.method, request.url.path,
        response.status_code, elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/info", summary="Service info")
def root():
    return {"service": "sentiment-model", "version": app.version,
            "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    model = state["model"]
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model_loaded=model is not None,
        classes=model.classes if model is not None else [],
    )


@app.post("/predict", response_model=PredictResponse, summary="Classify one text",
          dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def predict(req: PredictRequest):
    pred = state["model"].predict(req.text)
    return PredictResponse(label=pred.label, score=pred.score, scores=pred.scores)


@app.post("/predict/batch", response_model=BatchPredictResponse,
          summary="Classify many texts",
          dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def predict_batch(req: BatchPredictRequest):
    model = state["model"]
    preds = model.predict_batch(req.texts)
    return BatchPredictResponse(predictions=[
        PredictResponse(label=p.label, score=p.score, scores=p.scores) for p in preds
    ])


# Serve the built React playground at / (after the API routes above).
from pathlib import Path  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")
