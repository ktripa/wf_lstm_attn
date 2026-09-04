"""YAML-backed config with attribute access, and round-trip save for run logging."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Recursive dict -> attribute-access wrapper. Also behaves like a dict."""

    def __init__(self, d: dict):
        self._d = dict(d)
        for k, v in self._d.items():
            if isinstance(v, dict):
                self._d[k] = Config(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self._d.items():
            out[k] = v.to_dict() if isinstance(v, Config) else copy.deepcopy(v)
        return out

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return Config(yaml.safe_load(f))


def save_config(config: Config, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
