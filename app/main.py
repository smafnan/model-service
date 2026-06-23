"""The FastAPI service that serves the sentiment model.

Operational features a real service needs, all here:
  * **/health** — a liveness/readiness probe (does the model load? what classes?)
    so an orchestrator (Docker, k8s) can tell if the container is ready.
  * **request logging middleware** — every request logs a unique id, method,
    path, status, and latency in milliseconds.
  * **input validation** — Pydantic rejects bad payloads with a 422.
  * **lifespan model loading** — the model is loaded once at startup, not per
    request.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

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


@app.post("/predict", response_model=PredictResponse, summary="Classify one text")
def predict(req: PredictRequest):
    pred = state["model"].predict(req.text)
    return PredictResponse(label=pred.label, score=pred.score, scores=pred.scores)


@app.post("/predict/batch", response_model=BatchPredictResponse,
          summary="Classify many texts")
def predict_batch(req: BatchPredictRequest):
    model = state["model"]
    preds = [model.predict(t) for t in req.texts]
    return BatchPredictResponse(predictions=[
        PredictResponse(label=p.label, score=p.score, scores=p.scores) for p in preds
    ])


# Serve the built React playground at / (after the API routes above).
from pathlib import Path  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")
