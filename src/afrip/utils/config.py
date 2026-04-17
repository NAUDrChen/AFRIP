from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    data = _load_yaml(path)

    base_entries = data.get("_base_", [])
    if isinstance(base_entries, str):
        base_entries = [base_entries]
    if not isinstance(base_entries, list):
        raise ConfigError("_base_ must be a string or a list of strings")

    merged: dict[str, Any] = {}
    for entry in base_entries:
        base_path = (path.parent / entry).resolve()
        merged = deep_merge(merged, load_config(base_path))

    return deep_merge(merged, data)
