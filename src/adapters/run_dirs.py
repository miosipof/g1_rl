"""Run/debug output directory helpers.

These helpers keep real-robot debug outputs from different runs separated.  They
are intentionally dependency-light and use only standard-library modules.
"""

from __future__ import annotations

import json
import time
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


def write_run_info(output_dir: str | Path | None, payload: Mapping[str, Any]) -> None:
    """Write ``run_info.json`` if ``output_dir`` is not None."""

    if output_dir is None:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_info.json").write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["make_run_id", "timestamped_output_dir", "write_run_info"]
