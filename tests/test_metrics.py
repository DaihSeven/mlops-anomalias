from anomalias.data import FEATURES
from api.metrics import ApiMetrics

STATS = {f: {"mean": 0.0, "std": 1.0} for f in FEATURES}


def _version_series(m: ApiMetrics) -> list[str]:
    return [
        s.labels["version"]
        for fam in m.registry.collect()
        for s in fam.samples
        if s.name == "anomaly_api_model_version"
    ]


def test_model_version_change_removes_old_series_and_resets_windows():
    m = ApiMetrics()
    m.set_model(1, STATS)
    m.drift_monitor.update([[0.0] * len(FEATURES)] * 200)
    m.set_model(2, STATS)
    assert _version_series(m) == ["2"]
    assert len(m.drift_monitor) == 0


def test_registries_are_independent():
    a, b = ApiMetrics(), ApiMetrics()
    a.predictions_total.inc(5)
    assert b.registry.get_sample_value("anomaly_api_predictions_total") == 0
