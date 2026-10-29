"""Detecção simples de drift por janela deslizante.

Medida usada, por feature:

    drift = |média_janela - média_treino| / desvio_treino

ou seja, quantos desvios padrão do treino a média recente se afastou da média
de referência. É simples de propósito: barata (cabe numa máquina de 2 núcleos
a cada scrape), fácil de explicar e com unidade intuitiva, o que facilita
escolher o limiar do alerta.

Limitações (conhecidas e aceitas nesta etapa):
- Só detecta mudança de MÉDIA. Não vê mudança de variância (ex.: sensor
  ficando ruidoso com a mesma média), de forma da distribuição (ex.: bimodal
  com a mesma média) nem de correlação entre features (ex.: temperatura deixa
  de acompanhar a corrente). Testes como KS ou PSI cobririam parte disso.
- A janela inclui as anomalias reais do tráfego (~5% nos dados sintéticos),
  que puxam a média; por isso o drift basal com tráfego "normal" não é zero e
  o limiar do alerta deve ficar acima dele.
- O estado é por processo: com vários workers cada um teria a sua janela.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from anomalias.data import FEATURES

DEFAULT_WINDOW_SIZE = 500
# Com n registros, o erro padrão da média da janela é ~ 1/sqrt(n) desvios:
# n=10 -> ~0,32; n=100 -> ~0,10; n=500 -> ~0,045. Abaixo de 100 o ruído sozinho
# já se aproxima de limiares de alerta razoáveis, então não reportamos nada
# (melhor ausência de dado do que um valor falso logo após a API subir).
DEFAULT_MIN_SAMPLES = 100


class DriftMonitor:
    """Janela deslizante thread-safe dos últimos registros recebidos."""

    def __init__(
        self,
        reference_stats: dict[str, dict[str, float]],
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        features: list[str] = FEATURES,
    ) -> None:
        if set(reference_stats) != set(features):
            raise ValueError(f"reference_stats deve ter exatamente as features {features}")
        if not 0 < min_samples <= window_size:
            raise ValueError("min_samples deve estar em (0, window_size]")

        self.features = list(features)
        self.window_size = window_size
        self.min_samples = min_samples
        self._means = np.array([reference_stats[f]["mean"] for f in self.features], dtype=float)
        self._stds = np.array([reference_stats[f]["std"] for f in self.features], dtype=float)
        if np.any(self._stds <= 0):
            raise ValueError("desvios de referência devem ser positivos")

        self._window: deque[np.ndarray] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def update(self, X: np.ndarray) -> None:
        """Acrescenta registros (n x n_features, colunas na ordem de features)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError(f"X deve ter shape (n, {len(self.features)})")
        with self._lock:
            self._window.extend(X)

    def scores(self) -> dict[str, float] | None:
        """Drift por feature, ou None se a janela tiver menos de min_samples."""
        with self._lock:
            if len(self._window) < self.min_samples:
                return None
            window_mean = np.mean(np.stack(self._window), axis=0)
        drift = np.abs(window_mean - self._means) / self._stds
        return dict(zip(self.features, drift.tolist(), strict=True))

    def max_score(self) -> float | None:
        scores = self.scores()
        return None if scores is None else max(scores.values())

    def reset(self) -> None:
        """Esvazia a janela (chamar quando a versão do modelo mudar)."""
        with self._lock:
            self._window.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._window)
