"""Geração de dados SINTÉTICOS de sensores com anomalias conhecidas.

Os dados normais vêm de uma normal multivariada com correlação leve entre as
features. As anomalias são injetadas de forma controlada, então o rótulo
(1 = anomalia) é conhecido e permite medir precision/recall/F1.
"""

from __future__ import annotations

import numpy as np

FEATURES: list[str] = ["temperature", "pressure", "vibration", "current"]

# Operação normal: temperatura (°C), pressão (bar), vibração (mm/s), corrente (A)
MEANS = np.array([70.0, 5.0, 2.0, 15.0])
STDS = np.array([5.0, 0.5, 0.4, 2.0])

# Correlação leve: temperatura acompanha a corrente, vibração acompanha a pressão
CORRELATION = np.array(
    [
        [1.00, 0.10, 0.15, 0.40],
        [0.10, 1.00, 0.30, 0.05],
        [0.15, 0.30, 1.00, 0.10],
        [0.40, 0.05, 0.10, 1.00],
    ]
)

ANOMALY_TYPES = ("spike", "out_of_range", "improbable_combo")


def _covariance() -> np.ndarray:
    return CORRELATION * np.outer(STDS, STDS)


def _inject_anomalies(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Transforma amostras normais em anomalias de três tipos."""
    n = normal.shape[0]
    out = normal.copy()
    kinds = rng.choice(len(ANOMALY_TYPES), size=n)
    signs = rng.choice([-1.0, 1.0], size=(n, len(FEATURES)))

    for i in range(n):
        kind = ANOMALY_TYPES[kinds[i]]
        if kind == "spike":
            # Pico isolado: uma feature salta 6 a 10 desvios padrão
            j = rng.integers(len(FEATURES))
            out[i, j] = MEANS[j] + signs[i, j] * rng.uniform(6, 10) * STDS[j]
        elif kind == "out_of_range":
            # Várias features fora da faixa operacional ao mesmo tempo
            cols = rng.choice(len(FEATURES), size=rng.integers(2, 5), replace=False)
            out[i, cols] = MEANS[cols] + signs[i, cols] * rng.uniform(4, 7, len(cols)) * STDS[cols]
        else:
            # Combinação improvável: temperatura alta com corrente baixa e
            # vibração alta, contrariando a correlação dos dados normais
            out[i, 0] = MEANS[0] + rng.uniform(3, 5) * STDS[0]
            out[i, 3] = MEANS[3] - rng.uniform(3, 5) * STDS[3]
            out[i, 2] = MEANS[2] + rng.uniform(3, 5) * STDS[2]
    return out


def generate_dataset(
    n_samples: int = 5000, contamination: float = 0.05, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Gera X (n_samples x 4) e y (1 = anomalia, 0 = normal), embaralhados.

    O número de anomalias é round(n_samples * contamination).
    """
    if n_samples <= 0:
        raise ValueError("n_samples deve ser positivo")
    if not 0.0 <= contamination < 0.5:
        raise ValueError("contamination deve estar em [0, 0.5)")

    rng = np.random.default_rng(seed)
    n_anomalies = int(round(n_samples * contamination))

    X = rng.multivariate_normal(MEANS, _covariance(), size=n_samples)
    y = np.zeros(n_samples, dtype=int)
    if n_anomalies:
        X[:n_anomalies] = _inject_anomalies(X[:n_anomalies], rng)
        y[:n_anomalies] = 1

    order = rng.permutation(n_samples)
    return X[order], y[order]
