"""Request/response schemas — Pydantic validates input at the boundary.

Bad input (empty text, wrong types, oversized payloads) is rejected with a clear
422 before it ever reaches the model, which is exactly what you want at a service
edge.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# Same per-text constraints as PredictRequest.text, reused so batch items can't
# smuggle in an empty string that the single-predict endpoint would reject.
NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=5000)]


class PredictRequest(BaseModel):
    text: NonEmptyText = Field(description="The text to classify.")


class BatchPredictRequest(BaseModel):
    texts: list[NonEmptyText] = Field(min_length=1, max_length=100,
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
