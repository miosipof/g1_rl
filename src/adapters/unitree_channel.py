"""Process-wide initialization for the Unitree DDS channel factory.

The Unitree SDK owns one channel factory per process.  Hardware adapters share
this helper so importing or connecting a second adapter does not initialize the
factory again with conflicting network settings.
"""

from __future__ import annotations

import threading
from typing import Callable


ChannelInitializer = Callable[[int, str], object]

_lock = threading.Lock()
_initialized = False
_initialized_iface: str | None = None


def ensure_unitree_channel_initialized(
    iface: str,
    *,
    initializer: ChannelInitializer | None = None,
) -> None:
    """Initialize the Unitree DDS runtime exactly once for ``iface``.

    ``initializer`` is an injection seam for tests.  Production callers leave
    it unset, which keeps the optional Unitree SDK import lazy.
    """

    global _initialized, _initialized_iface

    iface = str(iface).strip()
    if not iface:
        raise ValueError("iface must be non-empty")

    with _lock:
        if _initialized:
            if _initialized_iface != iface:
                raise RuntimeError(
                    "Unitree DDS already initialized on "
                    f"{_initialized_iface!r}; cannot reinitialize on {iface!r}"
                )
            return

        if initializer is None:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore

            initializer = ChannelFactoryInitialize

        initializer(0, iface)
        _initialized = True
        _initialized_iface = iface


def _reset_unitree_channel_for_tests() -> None:
    """Reset only this module's bookkeeping; never call on a live robot."""

    global _initialized, _initialized_iface
    with _lock:
        _initialized = False
        _initialized_iface = None


__all__ = ["ensure_unitree_channel_initialized"]
