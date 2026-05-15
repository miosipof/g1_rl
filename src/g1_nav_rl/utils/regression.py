# src/g1_nav_rl/utils/regression.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_jsonable(value: Any) -> Any:
    """Convert tensors, arrays, and scalars into JSON-serializable objects."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    return value


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write a regression/smoke-test JSON report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, indent=2, sort_keys=True)


def assert_metric_range(
    name: str,
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """Small helper for readable smoke-test metric assertions."""
    if min_value is not None and value < min_value:
        raise AssertionError(f"{name}={value} is below minimum {min_value}")

    if max_value is not None and value > max_value:
        raise AssertionError(f"{name}={value} is above maximum {max_value}")