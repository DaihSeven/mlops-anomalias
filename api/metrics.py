"""Métricas Prometheus da API (prefixo anomaly_api_).

Cada instância de ApiMetrics tem o seu próprio CollectorRegistry, criado no
lifespan da API. Assim os testes (um TestClient por teste) não acumulam estado
no registry global do prometheus-client.

O estado (janelas de drift e de anomaly_ratio) é por processo: a API deve rodar
com 1 worker, senão cada worker exportaria uma visão parcial do tráfego.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from anomalias.drift import DEFAULT_WINDOW_SIZE, DriftMonitor

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
UNMATCHED = "unmatched"
METRICS_PATH = "/metrics"


class _DriftCollector(Collector):
    """Exporta drift_score e drift_score_max calculados na hora do scrape.

    Um Gauge comum sem labels (caso do drift_score_max) sempre exporta uma
    amostra, mesmo que 0. Com um collector, quando a janela ainda não tem
    registros suficientes simplesmente não emitimos nada, e o alerta não
    dispara com um valor falso.
    """

    def __init__(self, metrics: ApiMetrics) -> None:
        self._metrics = metrics

    def collect(self) -> Iterator[GaugeMetricFamily]:
        monitor = self._metrics.drift_monitor
        scores = monitor.scores() if monitor is not None else None
        if scores is None:
            return
        per_feature = GaugeMetricFamily(
            "anomaly_api_drift_score",
            "|média da janela - média do treino| / desvio do treino, por feature.",
            labels=["feature"],
        )
        for feature, value in scores.items():
            per_feature.add_metric([feature], value)
        yield per_feature
        yield GaugeMetricFamily(
            "anomaly_api_drift_score_max",
            "Maior drift_score entre as features.",
            value=max(scores.values()),
        )


class ApiMetrics:
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "anomaly_api_requests_total",
            "Requisições HTTP por template de rota e status.",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "anomaly_api_request_latency_seconds",
            "Latência das requisições HTTP.",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predictions_total = Counter(
            "anomaly_api_predictions_total",
            "Registros avaliados (não requisições).",
            registry=self.registry,
        )
        self.anomalies_total = Counter(
            "anomaly_api_anomalies_total",
            "Registros marcados como anomalia.",
            registry=self.registry,
        )
        self.anomaly_ratio = Gauge(
            "anomaly_api_anomaly_ratio",
            f"Proporção de anomalias nas últimas {window_size} predições.",
            registry=self.registry,
        )
        self.model_version = Gauge(
            "anomaly_api_model_version",
            "Versão do modelo ativo (valor sempre 1).",
            ["version"],
            registry=self.registry,
        )
        self.registry.register(_DriftCollector(self))

        self.window_size = window_size
        self.drift_monitor: DriftMonitor | None = None
        self._current_version: int | None = None
        self._recent_labels: deque[int] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def set_model(self, version: int, reference_stats: dict[str, dict[str, float]]) -> None:
        """Registra a versão ativa; ao trocar de versão, zera as janelas."""
        with self._lock:
            if version == self._current_version:
                return
            self._current_version = version
            # Remove a série da versão antiga para não ficarem duas versões "ativas"
            self.model_version.clear()
            self.model_version.labels(version=str(version)).set(1)
            # Janela da versão anterior não vale para a nova referência
            self.drift_monitor = DriftMonitor(reference_stats, window_size=self.window_size)
            self._recent_labels.clear()
            self.anomaly_ratio.set(0)

    def record_predictions(self, X: np.ndarray, labels: Iterable[int]) -> None:
        labels = [int(v) for v in labels]
        n_anomalies = sum(labels)
        self.predictions_total.inc(len(labels))
        if n_anomalies:
            self.anomalies_total.inc(n_anomalies)
        with self._lock:
            self._recent_labels.extend(labels)
            ratio = sum(self._recent_labels) / len(self._recent_labels)
            monitor = self.drift_monitor
        self.anomaly_ratio.set(ratio)
        if monitor is not None:
            monitor.update(X)

    def observe_request(self, endpoint: str, status: int, seconds: float) -> None:
        self.requests_total.labels(endpoint=endpoint, status=str(status)).inc()
        self.request_latency.labels(endpoint=endpoint).observe(seconds)

    def render(self) -> bytes:
        return generate_latest(self.registry)


class MetricsMiddleware:
    """Middleware ASGI que conta requisições e mede latência.

    Usa o template da rota (ex.: "/predict") e não a URL crua, para a
    cardinalidade dos labels ser limitada. Rotas inexistentes viram
    endpoint="unmatched". O próprio /metrics fica de fora.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500  # se a app estourar antes de responder

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # O roteador grava a rota casada em scope["route"] durante a chamada
            route: Any = scope.get("route")
            endpoint = getattr(route, "path", None) or UNMATCHED
            metrics: ApiMetrics | None = getattr(scope["app"].state, "metrics", None)
            if endpoint != METRICS_PATH and metrics is not None:
                metrics.observe_request(endpoint, status_code, time.perf_counter() - start)
