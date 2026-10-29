"""Treino do IsolationForest com dados sintéticos e rótulos conhecidos.

Uso:
    python -m anomalias.train --n-samples 5000 --contamination 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from anomalias import model_store
from anomalias.data import FEATURES, generate_dataset


def reference_stats(X: np.ndarray) -> dict[str, dict[str, float]]:
    """Média e desvio padrão por feature, usados depois na detecção de drift."""
    return {
        name: {"mean": float(X[:, j].mean()), "std": float(X[:, j].std())}
        for j, name in enumerate(FEATURES)
    }


def predict_labels(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Converte a saída do IsolationForest (-1 anomalia, 1 normal) para 1/0."""
    return (model.predict(X) == -1).astype(int)


def train(
    n_samples: int = 5000,
    contamination: float = 0.05,
    seed: int = 42,
    test_size: float = 0.3,
    n_estimators: int = 200,
    max_samples: int | str = "auto",
) -> tuple[IsolationForest, dict[str, Any]]:
    """Gera dados, treina e avalia. Retorna o modelo e os metadados para salvar."""
    X, y = generate_dataset(n_samples, contamination, seed)
    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    hyperparams = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "contamination": contamination,
        "random_state": seed,
    }
    model = IsolationForest(**hyperparams, n_jobs=1)
    model.fit(X_train)

    # Referência para o drift: só os pontos de treino que o próprio modelo
    # considera normais (sem usar y), para as anomalias não inflarem o desvio
    X_reference = X_train[predict_labels(model, X_train) == 0]

    y_pred = predict_labels(model, X_test)
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        # score_samples é maior para pontos normais; negando, maior = mais anômalo,
        # que é a orientação que a AUC espera para o rótulo 1 = anomalia
        "roc_auc": float(roc_auc_score(y_test, -model.score_samples(X_test))),
    }

    metadata = {
        "model_type": "IsolationForest",
        "features": FEATURES,
        "hyperparams": hyperparams,
        "data": {
            "synthetic": True,
            "n_samples": n_samples,
            "seed": seed,
            "test_size": test_size,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "metrics": metrics,
        "reference_stats": reference_stats(X_reference),
        "reference_n_samples": len(X_reference),
        "reference_criterion": "predicted_normal",
    }
    return model, metadata


def _max_samples(value: str) -> int | str:
    return value if value == "auto" else int(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o detector de anomalias.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-samples", type=_max_samples, default="auto")
    parser.add_argument("--models-dir", type=Path, default=model_store.DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--no-activate", action="store_true", help="Salva sem tornar a versão ativa."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, metadata = train(
        n_samples=args.n_samples,
        contamination=args.contamination,
        seed=args.seed,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
    )
    version = model_store.save_model(
        model, metadata, models_dir=args.models_dir, activate=not args.no_activate
    )
    print(json.dumps({"version": version, "metrics": metadata["metrics"]}, indent=2))
    return version


if __name__ == "__main__":
    main()
