import pytest
from fastapi.testclient import TestClient

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
