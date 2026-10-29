"""Schemas de entrada e saída da API.

Os campos do registro são gerados a partir de FEATURES, para que o schema
acompanhe o gerador de dados sem repetir os nomes das features.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from anomalias.data import FEATURES

MAX_BATCH = 1000

# strict=True rejeita texto (inclusive "70"); allow_inf_nan=False rejeita NaN e infinito
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]

SensorRecord: type[BaseModel] = create_model(
    "SensorRecord",
    __config__=ConfigDict(extra="forbid"),
    **{name: (FiniteFloat, ...) for name in FEATURES},
)

SensorBatch = Annotated[list[SensorRecord], Field(min_length=1, max_length=MAX_BATCH)]  # type: ignore[valid-type]
PredictRequest = SensorRecord | SensorBatch  # type: ignore[valid-type]


class Prediction(BaseModel):
    is_anomaly: bool
    anomaly_score: float = Field(description="Maior = mais anômalo (-score_samples).")
    model_version: int


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_version: int | None
    uptime_seconds: float
    detail: str | None = None


ModelInfo = dict[str, Any]
