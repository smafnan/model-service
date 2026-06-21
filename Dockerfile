# Small, reproducible image for the FastAPI model service.
FROM python:3.11-slim

# Don't write .pyc files; flush logs immediately (so `docker logs` is live).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /service

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and data.
COPY app ./app
COPY data ./data

# Bake the trained model into the image so startup is instant and offline.
RUN python -m app.model

# Run as a non-root user (good container hygiene).
RUN useradd --create-home appuser && chown -R appuser /service
USER appuser

EXPOSE 8000

# Container-level health check hits the /health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
