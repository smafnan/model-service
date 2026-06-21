"""Request/response schemas — Pydantic validates input at the boundary.

Bad input (empty text, wrong types, oversized payloads) is rejected with a clear
422 before it ever reaches the model, which is exactly what you want at a service
edge.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000,
                      description="The text to classify.")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100,
                             description="Up to 100 texts to classify.")


class PredictResponse(BaseModel):
    label: str
    score: float
    scores: dict[str, float]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]
