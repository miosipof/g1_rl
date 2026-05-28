"""Run/debug output directory helpers.

These helpers keep real-robot debug outputs from different runs separated.  They
are intentionally dependency-light and use only standard-library modules.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Mapping


def make_run_id(*, prefix: str = "run", timestamp_s: float | None = None) -> str:
    """Return a filesystem-safe run id such as ``run_20260522_153012``."""

    t = time.localtime(time.time() if timestamp_s is None else float(timestamp_s))
    stamp = time.strftime("%Y%m%d_%H%M%S", t)
    prefix_clean = str(prefix).strip().replace(" ", "_") or "run"
    return f"{prefix_clean}_{stamp}"


def timestamped_output_dir(base_dir: str | Path | None, *, run_id: str, enabled: bool = True) -> Path | None:
    """Return ``base_dir/run_id`` when enabled, otherwise ``base_dir``.

    ``None`` remains ``None`` so callers can pass optional debug dirs through
    without extra branching.
    """

    if base_dir is None:
        return None
    base = Path(base_dir)
    return base / str(run_id) if enabled else base


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of common runtime objects.

    Run metadata often contains dataclass configs, ``Path`` objects, numpy/torch
    scalar-like values, and small objects exposing ``to_dict``.  Keep this helper
    dependency-light so output-directory utilities remain safe to import on the
    robot.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (Path, PathLike)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    # Handle numpy / torch scalar-like and array-like objects without importing
    # optional dependencies here.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass

    return str(value)


def write_run_info(output_dir: str | Path | None, payload: Mapping[str, Any]) -> None:
    """Write ``run_info.json`` if ``output_dir`` is not None."""

    if output_dir is None:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_info.json").write_text(json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["make_run_id", "timestamped_output_dir", "write_run_info"]
