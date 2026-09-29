from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from adapters.unitree_channel import (
    _reset_unitree_channel_for_tests,
    ensure_unitree_channel_initialized,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    _reset_unitree_channel_for_tests()
    yield
    _reset_unitree_channel_for_tests()


def test_channel_initializes_once_and_rejects_another_interface() -> None:
    calls: list[tuple[int, str]] = []
    init = lambda domain, iface: calls.append((domain, iface))
    ensure_unitree_channel_initialized("eth0", initializer=init)
    ensure_unitree_channel_initialized("eth0", initializer=init)
    assert calls == [(0, "eth0")]
    with pytest.raises(RuntimeError, match="cannot reinitialize"):
        ensure_unitree_channel_initialized("en0", initializer=init)


def test_channel_concurrent_initialization_calls_sdk_once() -> None:
    calls: list[tuple[int, str]] = []
    init = lambda domain, iface: calls.append((domain, iface))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: ensure_unitree_channel_initialized("eth0", initializer=init), range(20)))
    assert calls == [(0, "eth0")]
