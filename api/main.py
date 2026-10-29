"""API de detecção de anomalias (FastAPI).

Sobe com:
    MODELS_DIR=models uvicorn api.main:app --workers 1

Use 1 worker: as janelas de drift e de anomaly_ratio vivem na memória do
processo. Com N workers, cada um veria ~1/N do tráfego e o Prometheus
coletaria a visão de um worker diferente a cada scrape.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel

from anomalias.data import FEATURES
from anomalias.model_store import load_current
from anomalias.train import predict_labels
from api.metrics import ApiMetrics, MetricsMiddleware
from api.schemas import HealthResponse, ModelInfo, Prediction, PredictRequest, PredictResponse

logger = logging.getLogger("anomaly_api")


def _models_dir() -> Path:
    # Lido na subida (e não no import) para respeitar MODELS_DIR definido depois
    return Path(os.environ.get("MODELS_DIR", "models"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.started_at = time.monotonic()
    app.state.model = None
    app.state.metadata = None
    app.state.load_error = None
    # Registry novo a cada subida (ver api/metrics.py)
    app.state.metrics = ApiMetrics()

    models_dir = _models_dir()
    try:
        model, metadata = load_current(models_dir)
        if metadata.get("features") != FEATURES:
            raise ValueError(f"features do modelo {metadata.get('features')} diferem de {FEATURES}")
        app.state.model, app.state.metadata = model, metadata
        app.state.metrics.set_model(int(metadata["version"]), metadata["reference_stats"])
        logger.info("Modelo v%s carregado de %s", metadata["version"], models_dir)
    except Exception as exc:
        # A API sobe mesmo sem modelo: /health mostra o problema e /predict responde 503
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Falha ao carregar modelo de %s: %s", models_dir, app.state.load_error)
    yield


app = FastAPI(title="Anomaly detection API", version="0.1.0", lifespan=lifespan)
app.add_middleware(MetricsMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # O handler padrão ecoa o "input" recebido; com NaN/Infinity isso quebra a
    # serialização da resposta e vira 500. Devolvemos os erros sem o input.
    errors = [{k: v for k, v in err.items() if k != "input"} for err in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": jsonable_encoder(errors)},
    )


def _require_model(request: Request) -> tuple[Any, dict[str, Any]]:
    state = request.app.state
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Nenhum modelo carregado ({state.load_error}). Treine um modelo e reinicie a API."
            ),
        )
    return state.model, state.metadata


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    loaded = state.model is not None
    return HealthResponse(
        status="ok" if loaded else "no_model",
        model_version=state.metadata["version"] if loaded else None,
        uptime_seconds=round(time.monotonic() - state.started_at, 3),
        detail=None if loaded else state.load_error,
    )


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    return Response(request.app.state.metrics.render(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    _, metadata = _require_model(request)
    return metadata


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:  # type: ignore[valid-type]
    model, metadata = _require_model(request)
    records: list[BaseModel] = payload if isinstance(payload, list) else [payload]
    X = np.array([[getattr(r, name) for name in FEATURES] for r in records], dtype=float)

    labels = predict_labels(model, X)
    # score_samples é maior para pontos normais; negamos para que maior = mais anômalo,
    # a mesma convenção da roc_auc do treino. O modelo marca anomalia quando
    # anomaly_score > -model.offset_ (limiar definido pela contamination).
    scores = -model.score_samples(X)

    request.app.state.metrics.record_predictions(X, labels)

    version = int(metadata["version"])
    return PredictResponse(
        predictions=[
            Prediction(is_anomaly=bool(label), anomaly_score=float(score), model_version=version)
            for label, score in zip(labels, scores, strict=True)
        ]
    )
