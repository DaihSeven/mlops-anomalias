"""Versionamento simples de modelos em disco.

Layout em models/:
    model_vN.joblib   modelo serializado
    model_vN.json     metadados (versão, data, hiperparâmetros, métricas, estatísticas)
    current.json      aponta para a versão ativa
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

DEFAULT_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
CURRENT_FILE = "current.json"
_VERSION_RE = re.compile(r"^model_v(\d+)\.joblib$")


def _model_path(models_dir: Path, version: int) -> Path:
    return models_dir / f"model_v{version}.joblib"


def _meta_path(models_dir: Path, version: int) -> Path:
    return models_dir / f"model_v{version}.json"


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    # Escreve em arquivo temporário e troca, para a API nunca ler JSON pela metade
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _existing_versions(models_dir: Path) -> list[int]:
    if not models_dir.exists():
        return []
    versions = []
    for p in models_dir.iterdir():
        m = _VERSION_RE.match(p.name)
        if m:
            versions.append(int(m.group(1)))
    return sorted(versions)


def save_model(
    model: Any,
    metadata: dict[str, Any],
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    activate: bool = True,
) -> int:
    """Salva o modelo como a próxima versão e retorna o número dela.

    Com activate=False o modelo fica salvo mas current.json não muda
    (útil quando o portão de qualidade reprova o novo modelo).
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    versions = _existing_versions(models_dir)
    version = (versions[-1] + 1) if versions else 1

    joblib.dump(model, _model_path(models_dir, version))
    meta = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        **metadata,
    }
    _write_json_atomic(_meta_path(models_dir, version), meta)

    if activate:
        _write_json_atomic(
            models_dir / CURRENT_FILE,
            {"version": version, "activated_at": datetime.now(UTC).isoformat()},
        )
    return version


def load_current(models_dir: Path | str = DEFAULT_MODELS_DIR) -> tuple[Any, dict[str, Any]]:
    """Carrega o modelo ativo e seus metadados."""
    models_dir = Path(models_dir)
    current_file = models_dir / CURRENT_FILE
    if not current_file.exists():
        raise FileNotFoundError(f"Nenhum modelo ativo em {models_dir}")

    version = int(json.loads(current_file.read_text())["version"])
    model = joblib.load(_model_path(models_dir, version))
    metadata = json.loads(_meta_path(models_dir, version).read_text())
    return model, metadata


def list_versions(models_dir: Path | str = DEFAULT_MODELS_DIR) -> list[dict[str, Any]]:
    """Lista os metadados de todas as versões, em ordem crescente."""
    models_dir = Path(models_dir)
    return [
        json.loads(_meta_path(models_dir, v).read_text()) for v in _existing_versions(models_dir)
    ]
