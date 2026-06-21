"""Model training, persistence, and inference — the ML behind the service.

Kept separate from the web layer so the API code stays about HTTP, and the model
can be trained, saved, loaded, and tested on its own. The service loads a saved
artifact at startup; if none exists it trains one from the bundled CSV, so the
container is always runnable with no manual step.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "sentiment.csv"
DEFAULT_ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"


@dataclass
class Prediction:
    label: str
    score: float                     # confidence of the predicted label
    scores: dict[str, float]         # full probability distribution


def _load_csv(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def train(data_path: Path = DEFAULT_DATA) -> Pipeline:
    """Train a TF-IDF + logistic-regression sentiment classifier."""
    texts, labels = _load_csv(data_path)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(texts, labels)
    return pipe


def save(model: Pipeline, path: Path = DEFAULT_ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


class SentimentModel:
    """Thin inference wrapper around the fitted pipeline."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.classes = list(pipeline.named_steps["clf"].classes_)

    @classmethod
    def load_or_train(
        cls, artifact: Path = DEFAULT_ARTIFACT, data: Path = DEFAULT_DATA
    ) -> "SentimentModel":
        """Load a saved model, or train+save one if the artifact is missing."""
        if artifact.exists():
            return cls(joblib.load(artifact))
        model = train(data)
        save(model, artifact)
        return cls(model)

    def predict(self, text: str) -> Prediction:
        proba = self.pipeline.predict_proba([text])[0]
        scores = {c: float(p) for c, p in zip(self.classes, proba)}
        label = max(scores, key=scores.get)
        return Prediction(label=label, score=scores[label], scores=scores)


if __name__ == "__main__":  # `python -m app.model` trains and saves the artifact
    path = save(train())
    print(f"Trained and saved model to {path}")
