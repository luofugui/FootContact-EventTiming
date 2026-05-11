from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def load_config(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return _to_namespace(yaml.safe_load(f))
