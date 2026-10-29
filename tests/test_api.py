import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from anomalias import model_store
from anomalias.data import FEATURES, MEANS, STDS
from anomalias.train import train
from api.main import app
from api.schemas import MAX_BATCH

NORMAL = dict(zip(FEATURES, MEANS.tolist(), strict=True))
# Vários desvios padrão fora da operação normal em todas as features
ANOMALOUS = dict(zip(FEATURES, (MEANS + 8 * STDS * [1, 1, 1, -1]).tolist(), strict=True))


@pytest.fixture(scope="module")
def trained():
    return train(n_samples=1000, seed=0, n_estimators=50)


@pytest.fixture
def client(tmp_path, monkeypatch, trained):
    model, meta = trained
    model_store.save_model(model, meta, tmp_path)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    with TestClient(app) as c:
        yield c


def test_health_with_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == 1
    assert body["uptime_seconds"] >= 0


def test_model_metadata(client):
    body = client.get("/model").json()
    assert body["version"] == 1
    assert body["features"] == FEATURES
    assert "roc_auc" in body["metrics"]


def test_predict_single_record(client):
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    (pred,) = r.json()["predictions"]
    assert pred["is_anomaly"] is False
    assert pred["model_version"] == 1
    assert isinstance(pred["anomaly_score"], float)


def test_predict_batch_and_score_order(client):
    r = client.post("/predict", json=[NORMAL, ANOMALOUS, NORMAL])
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["is_anomaly"] for p in preds] == [False, True, False]
    # Convenção: maior = mais anômalo
    assert preds[1]["anomaly_score"] > preds[0]["anomaly_score"]


def test_clear_anomaly_is_flagged(client):
    (pred,) = client.post("/predict", json=ANOMALOUS).json()["predictions"]
    assert pred["is_anomaly"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in NORMAL.items() if k != "pressure"},  # feature faltando
        {**NORMAL, "temperature": "quente"},  # texto
        {**NORMAL, "temperature": "70"},  # número como texto
        {**NORMAL, "unknown": 1.0},  # campo extra
        [],  # lote vazio
        [NORMAL] * (MAX_BATCH + 1),  # lote grande demais
    ],
    ids=["missing", "text", "numeric_text", "extra", "empty_batch", "batch_too_large"],
)
def test_invalid_inputs_return_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_return_422(client, literal):
    # JSON padrão não tem NaN; enviamos o literal cru como o json do Python faria
    body = f'{{"temperature": {literal}, "pressure": 5, "vibration": 2, "current": 15}}'
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert any(e["type"] == "finite_number" for e in r.json()["detail"])


def test_api_without_model(client_without_model):
    health = client_without_model.get("/health").json()
    assert health["status"] == "no_model"
    assert health["model_version"] is None
    assert "FileNotFoundError" in health["detail"]

    r = client_without_model.post("/predict", json=NORMAL)
    assert r.status_code == 503
    assert "Nenhum modelo carregado" in r.json()["detail"]
    assert client_without_model.get("/model").status_code == 503


# --- /metrics ---------------------------------------------------------------


def _samples(client) -> list:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    return [s for fam in text_string_to_metric_families(r.text) for s in fam.samples]


def _value(samples, name, **labels):
    for s in samples:
        if s.name == name and s.labels == labels:
            return s.value
    return None


def test_metrics_after_some_calls(client):
    client.get("/health")
    client.post("/predict", json=NORMAL)
    samples = _samples(client)

    names = {s.name for s in samples}
    expected = {
        "anomaly_api_requests_total",
        "anomaly_api_request_latency_seconds_bucket",
        "anomaly_api_predictions_total",
        "anomaly_api_anomalies_total",
        "anomaly_api_anomaly_ratio",
        "anomaly_api_model_version",
    }
    assert expected <= names
    assert _value(samples, "anomaly_api_requests_total", endpoint="/health", status="200") == 1
    assert _value(samples, "anomaly_api_requests_total", endpoint="/predict", status="200") == 1
    assert _value(samples, "anomaly_api_model_version", version="1") == 1
    # Poucos registros na janela: drift não é exportado (nem como 0)
    assert not any(n.startswith("anomaly_api_drift_score") for n in names)


def test_metrics_endpoint_not_counted(client):
    client.get("/metrics")
    client.get("/metrics")
    samples = _samples(client)
    assert not any(s.labels.get("endpoint") == "/metrics" for s in samples)


def test_unknown_route_is_unmatched(client):
    assert client.get("/nao-existe/123").status_code == 404
    samples = _samples(client)
    assert _value(samples, "anomaly_api_requests_total", endpoint="unmatched", status="404") == 1
    assert not any("nao-existe" in str(s.labels) for s in samples)


def test_predictions_total_counts_records(client):
    client.post("/predict", json=[NORMAL, ANOMALOUS, NORMAL])
    client.post("/predict", json=ANOMALOUS)
    samples = _samples(client)
    assert _value(samples, "anomaly_api_predictions_total") == 4
    assert _value(samples, "anomaly_api_anomalies_total") == 2
    assert _value(samples, "anomaly_api_anomaly_ratio") == pytest.approx(0.5)
    assert _value(samples, "anomaly_api_requests_total", endpoint="/predict", status="200") == 2


def test_validation_error_counted_as_422(client):
    client.post("/predict", json={"temperature": "quente"})
    samples = _samples(client)
    assert _value(samples, "anomaly_api_requests_total", endpoint="/predict", status="422") == 1
    assert _value(samples, "anomaly_api_predictions_total") == 0


def test_drift_exported_after_min_samples(client):
    client.post("/predict", json=[NORMAL] * 99)
    assert _value(_samples(client), "anomaly_api_drift_score_max") is None
    client.post("/predict", json=[NORMAL])
    samples = _samples(client)
    for feature in FEATURES:
        assert _value(samples, "anomaly_api_drift_score", feature=feature) is not None
    assert _value(samples, "anomaly_api_drift_score_max") is not None


def test_metrics_without_model(client_without_model):
    client_without_model.post("/predict", json=NORMAL)
    samples = _samples(client_without_model)
    assert _value(samples, "anomaly_api_requests_total", endpoint="/predict", status="503") == 1
    assert not any(s.name == "anomaly_api_model_version" for s in samples)


def test_metrics_not_in_openapi(client):
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]
