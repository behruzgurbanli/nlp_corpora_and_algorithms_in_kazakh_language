#!/usr/bin/env python3
"""
Config utilities.

Purpose:
- Load YAML configs as plain dicts
- Optionally coerce YAML dicts into dataclass config objects
- Convert string paths -> pathlib.Path for dataclass fields typed as Path

This keeps modules reusable:
- UI can generate the same keys
- CLI can load YAML and call library functions
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML mapping file into a Python dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping (dict). Got: {type(data)}")

    return data


def _is_path_type(t: Any) -> bool:
    """
    Handle Path typing even when `from __future__ import annotations`
    stores field types as strings.
    """
    if t is Path:
        return True
    if isinstance(t, str):
        return t in {"Path", "pathlib.Path", "pathlib.PathLike", "PathLike"}
    return False


def load_config_as(path: str | Path, cls: Type[T]) -> T:
    """
    Load YAML and create a dataclass instance of type `cls`.

    Behavior:
    - unknown keys are ignored (so UI can add harmless extras)
    - Path-typed fields are converted from strings to Path
    """
    if not is_dataclass(cls):
        raise TypeError("load_config_as expects a dataclass type")

    data = load_yaml(path)
    allowed = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}

    for k, v in data.items():
        if k not in allowed:
            continue

        f = allowed[k]

        # minimal coercion: Path fields (robust to postponed annotations)
        if _is_path_type(f.type) and isinstance(v, (str, Path)):
            kwargs[k] = Path(v)
        else:
            kwargs[k] = v

    return cls(**kwargs)  # type: ignore[arg-type]
