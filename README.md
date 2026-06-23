# Sentiment Model Service — FastAPI + Docker

> **AI Engineer Roadmap — Project 5.1**
> *Teaches: serving, Docker, API design, the operational reality of models.*
> *Done when: someone else can run your service from your README without asking you anything.*

A sentiment classifier (TF-IDF + logistic regression) wrapped in a **FastAPI**
service, **containerised with Docker**, with **request logging**, **input
validation**, and a **health check**. The model is trained from a bundled dataset
and baked into the image, so the service runs with **zero manual steps**.

## Run it

### Option A — Docker (recommended)

```bash
docker compose up --build
# ...service is live at http://localhost:8000
```

That's the whole setup. The image installs dependencies, trains the model, and
starts the server. A container `HEALTHCHECK` reports readiness.

### Option B — locally with Python

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload                        # http://localhost:8000
```

The model is trained automatically on first start if no artifact exists.

## 🖥️ Web playground (React + Tailwind)

A live **Sentiment Playground** ships with the service and is served at the root
URL: type anything and it classifies in real time (debounced) into positive /
neutral / negative with animated probability bars and a model-health badge.

- Run the service (Docker or `uvicorn app.main:app`) and open **http://localhost:8000**.
- The prebuilt `web/dist` is committed (and baked into the Docker image), so the
  UI works straight from a clone — no Node build needed just to run it.
- Rebuild/develop the frontend: `cd web && npm install && npm run build`.

## Use it

Interactive API docs (Swagger UI) are at **http://localhost:8000/docs**, and the
JSON service info moved to **/info** (the root now serves the playground).

```bash
# Health check
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"classes":["negative","neutral","positive"]}

# Classify one text
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"absolutely love it, fantastic quality"}'
# {"label":"positive","score":0.43,"scores":{...}}

# Classify a batch
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["love it","hate it","it is okay"]}'
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | service info |
| GET | `/health` | liveness/readiness probe (used by Docker `HEALTHCHECK`) |
| POST | `/predict` | classify one text |
| POST | `/predict/batch` | classify up to 100 texts |
| GET | `/docs` | interactive Swagger UI |

## Operational features (the "reality of models")

- **Health check** — `/health` reports whether the model is loaded and its
  classes; the Dockerfile and compose file wire it into a container `HEALTHCHECK`
  so an orchestrator knows when the service is ready.
- **Request logging** — middleware logs every request with a correlation id,
  method, path, status, and **latency in ms**, and echoes the id back as an
  `X-Request-ID` header.
- **Input validation** — Pydantic rejects empty text, missing fields, and
  oversized batches with a clear **422** before the model is ever called.
- **Load once at startup** — the model is loaded in the FastAPI lifespan, not per
  request.
- **Container hygiene** — runs as a non-root user; dependency layer is cached;
  `PYTHONUNBUFFERED` so `docker logs` is live.

## Test it

```bash
pip install -r requirements.txt
pytest -q   # 9 tests via FastAPI TestClient: endpoints, validation, the model
```

## Layout

```
app/
├── main.py      # FastAPI app: routes, logging middleware, lifespan model load
├── model.py     # train / save / load + inference wrapper
└── schemas.py   # Pydantic request/response models (validation)
data/sentiment.csv  # bundled training data
tests/              # 9 TestClient tests
Dockerfile          # slim image, model baked in, HEALTHCHECK, non-root
docker-compose.yml  # one-command run
requirements.txt
```

> The dataset is intentionally tiny (for a self-contained demo), so confidence
> scores are modest — the point of this project is the **serving and ops**, not
> the model's accuracy. Swap in any scikit-learn pipeline and the service is
> unchanged.

## License

MIT.
