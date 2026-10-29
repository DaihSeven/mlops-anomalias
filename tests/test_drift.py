import threading

import numpy as np
import pytest

from anomalias.data import FEATURES, STDS, generate_dataset
from anomalias.drift import DriftMonitor
from anomalias.train import train

TRAIN_SEED = 42
TRAFFIC_SEED = 2024  # diferente do treino


@pytest.fixture(scope="module")
def reference_stats():
    _, meta = train(seed=TRAIN_SEED)
    return meta["reference_stats"]


@pytest.fixture(scope="module")
def traffic():
    return generate_dataset(n_samples=5000, contamination=0.05, seed=TRAFFIC_SEED)


def test_normal_traffic_has_low_drift(reference_stats, traffic):
    X, y = traffic
    monitor = DriftMonitor(reference_stats)
    monitor.update(X[y == 0][:500])
    assert monitor.max_score() < 0.3, monitor.scores()


def test_traffic_with_real_anomalies_has_low_drift(reference_stats, traffic):
    X, y = traffic
    monitor = DriftMonitor(reference_stats)
    monitor.update(X[:500])
    assert y[:500].sum() > 0  # a janela tem anomalias reais misturadas
    assert monitor.max_score() < 0.3, monitor.scores()


def test_temperature_shift_of_one_std_is_detected(reference_stats, traffic):
    X, y = traffic
    shifted = X[y == 0][:500].copy()
    shifted[:, FEATURES.index("temperature")] += STDS[FEATURES.index("temperature")]
    monitor = DriftMonitor(reference_stats)
    monitor.update(shifted)
    scores = monitor.scores()
    assert 0.8 <= scores["temperature"] <= 1.3, scores
    assert all(scores[f] < 0.3 for f in FEATURES if f != "temperature"), scores


def test_returns_none_below_min_samples(reference_stats, traffic):
    X, _ = traffic
    monitor = DriftMonitor(reference_stats)
    monitor.update(X[:99])
    assert monitor.scores() is None
    assert monitor.max_score() is None
    monitor.update(X[99:100])
    assert monitor.scores() is not None


def test_window_keeps_only_last_records(reference_stats, traffic):
    X, y = traffic
    normal = X[y == 0]
    shifted = normal[:500].copy()
    shifted[:, 0] += 10 * STDS[0]
    monitor = DriftMonitor(reference_stats)
    monitor.update(shifted)
    monitor.update(normal[500:1000])  # empurra todo o tráfego deslocado para fora
    assert len(monitor) == 500
    assert monitor.max_score() < 0.3


def test_reset_empties_window(reference_stats, traffic):
    X, _ = traffic
    monitor = DriftMonitor(reference_stats)
    monitor.update(X[:200])
    monitor.reset()
    assert len(monitor) == 0
    assert monitor.scores() is None


def test_concurrent_updates(reference_stats, traffic):
    X, _ = traffic
    monitor = DriftMonitor(reference_stats, window_size=5000)
    threads = [threading.Thread(target=monitor.update, args=(X[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(monitor) == len(X)


def test_invalid_reference_stats():
    with pytest.raises(ValueError):
        DriftMonitor({"temperature": {"mean": 0.0, "std": 1.0}})
    stats = {f: {"mean": 0.0, "std": 1.0} for f in FEATURES}
    stats["pressure"]["std"] = 0.0
    with pytest.raises(ValueError):
        DriftMonitor(stats)


def test_update_rejects_wrong_shape(reference_stats):
    monitor = DriftMonitor(reference_stats)
    with pytest.raises(ValueError):
        monitor.update(np.zeros((3, 2)))
