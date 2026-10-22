import json

import pytest
from sklearn.ensemble import IsolationForest

from anomalias import model_store
from anomalias.data import generate_dataset


@pytest.fixture
def fitted_model():
    X, _ = generate_dataset(300, 0.05, seed=0)
    return IsolationForest(n_estimators=10, random_state=0).fit(X)


def test_versions_increment_and_current_points_to_last(tmp_path, fitted_model):
    v1 = model_store.save_model(fitted_model, {"metrics": {"f1": 0.9}}, tmp_path)
    v2 = model_store.save_model(fitted_model, {"metrics": {"f1": 0.95}}, tmp_path)

    assert (v1, v2) == (1, 2)
    assert (tmp_path / "model_v1.joblib").exists()
    assert (tmp_path / "model_v2.json").exists()
    assert json.loads((tmp_path / "current.json").read_text())["version"] == 2


def test_metadata_contents(tmp_path, fitted_model):
    model_store.save_model(fitted_model, {"metrics": {"f1": 0.9}}, tmp_path)
    meta = json.loads((tmp_path / "model_v1.json").read_text())
    assert meta["version"] == 1
    assert "created_at" in meta
    assert meta["metrics"] == {"f1": 0.9}


def test_load_current_returns_working_model(tmp_path, fitted_model):
    model_store.save_model(fitted_model, {"tag": "a"}, tmp_path)
    model, meta = model_store.load_current(tmp_path)
    X, _ = generate_dataset(20, 0.1, seed=1)
    assert model.predict(X).shape == (20,)
    assert meta["version"] == 1
    assert meta["tag"] == "a"


def test_save_without_activate_keeps_current(tmp_path, fitted_model):
    model_store.save_model(fitted_model, {}, tmp_path)
    v2 = model_store.save_model(fitted_model, {}, tmp_path, activate=False)
    assert v2 == 2
    _, meta = model_store.load_current(tmp_path)
    assert meta["version"] == 1


def test_list_versions(tmp_path, fitted_model):
    assert model_store.list_versions(tmp_path) == []
    for _ in range(3):
        model_store.save_model(fitted_model, {}, tmp_path)
    assert [m["version"] for m in model_store.list_versions(tmp_path)] == [1, 2, 3]


def test_load_current_without_models_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_store.load_current(tmp_path)
