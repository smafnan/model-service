"""Tests for the model service, using FastAPI's TestClient.

TestClient runs the app in-process (triggering the lifespan model load), so these
exercise the real endpoints, validation, and headers without a running server.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.main import app
from app.model import SentimentModel


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # 'with' triggers startup (model load) + shutdown
        yield c


# --- model unit ------------------------------------------------------------ #

def test_model_predicts_sentiment():
    model = SentimentModel.load_or_train()
    assert model.predict("I love this, it is wonderful").label == "positive"
    assert model.predict("terrible, broke and useless").label == "negative"


def test_prediction_scores_sum_to_one():
    model = SentimentModel.load_or_train()
    pred = model.predict("it is fine, nothing special")
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-6
    assert pred.score == max(pred.scores.values())


# --- endpoints ------------------------------------------------------------- #

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert set(body["classes"]) == {"positive", "negative", "neutral"}


def test_info(client):
    r = client.get("/info")
    assert r.status_code == 200
    assert r.json()["service"] == "sentiment-model"


def test_predict(client):
    r = client.post("/predict", json={"text": "absolutely fantastic, love it"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["score"] <= 1.0
    assert "X-Request-ID" in r.headers     # logging middleware sets a correlation id


def test_predict_batch(client):
    r = client.post("/predict/batch",
                    json={"texts": ["love it", "hate it", "it is okay"]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3


# --- input validation ------------------------------------------------------ #

def test_empty_text_rejected(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422        # Pydantic min_length=1


def test_missing_field_rejected(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_batch_too_large_rejected(client):
    r = client.post("/predict/batch", json={"texts": ["x"] * 101})
    assert r.status_code == 422        # max_length=100


def test_batch_empty_item_rejected(client):
    r = client.post("/predict/batch", json={"texts": ["love it", ""]})
    assert r.status_code == 422        # each item must match PredictRequest.text's min_length=1


# --- optional API-key auth -------------------------------------------------- #

def test_predict_open_when_api_key_unset(client):
    assert app_main.API_KEY is None    # default: no env set, demo stays open
    r = client.post("/predict", json={"text": "love it"})
    assert r.status_code == 200


def test_predict_rejected_without_matching_api_key(client, monkeypatch):
    monkeypatch.setattr(app_main, "API_KEY", "s3cret")

    r = client.post("/predict", json={"text": "love it"})
    assert r.status_code == 401        # no header

    r = client.post("/predict", json={"text": "love it"},
                    headers={"X-API-Key": "wrong"})
    assert r.status_code == 401        # wrong header


def test_predict_accepted_with_matching_api_key(client, monkeypatch):
    monkeypatch.setattr(app_main, "API_KEY", "s3cret")

    r = client.post("/predict", json={"text": "love it"},
                    headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200


# --- rate limiting ----------------------------------------------------------- #

def test_rate_limit_triggers_429(client, monkeypatch):
    monkeypatch.setattr(app_main.rate_limiter, "limit", 2)
    app_main.rate_limiter.reset()      # drop hits recorded by earlier tests
    try:
        for _ in range(2):
            r = client.post("/predict", json={"text": "love it"})
            assert r.status_code == 200
        r = client.post("/predict", json={"text": "love it"})
        assert r.status_code == 429
    finally:
        app_main.rate_limiter.reset()  # don't leak into later tests
