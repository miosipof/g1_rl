from __future__ import annotations

import pytest

from adapters import Command2D, CommandLimitsConfig
from adapters.real_unitree_loco import (
    DryRunCommandSink,
    UnitreeLocoAdapter,
    UnitreeLocoAdapterConfig,
    call_move_with_sdk_compat,
    call_stop_move,
    gate_loco_state,
    robot_state_from_loco_mapping,
)


class _FakeClient:
    def __init__(self) -> None:
        self.moves: list[tuple[tuple[float, float, float], dict[str, object]]] = []
        self.stop_count = 0

    def Move(self, vx: float, vy: float, omega: float, **kwargs: object) -> int:
        self.moves.append(((vx, vy, omega), dict(kwargs)))
        return 123

    def StopMove(self) -> int:
        self.stop_count += 1
        return 456


class _ContinuousOnlyClient(_FakeClient):
    def Move(self, vx: float, vy: float, omega: float, **kwargs: object) -> int:
        if "continous_move" in kwargs:
            raise TypeError("unexpected keyword: continous_move")
        if "continuous_move" not in kwargs:
            raise TypeError("expected continuous_move")
        self.moves.append(((vx, vy, omega), dict(kwargs)))
        return 321


class _NoKeywordClient(_FakeClient):
    def Move(self, vx: float, vy: float, omega: float, *args: object, **kwargs: object) -> int:
        if kwargs:
            raise TypeError("no keywords supported")
        self.moves.append(((vx, vy, omega), {}))
        return 999


def _limits() -> CommandLimitsConfig:
    return CommandLimitsConfig.from_dict(
        {
            "vx_min": -0.05,
            "vx_max": 0.10,
            "vy_min": -0.03,
            "vy_max": 0.03,
            "wz_min": -0.20,
            "wz_max": 0.20,
            "command_rate_hz": 10.0,
        }
    )


def test_call_move_with_sdk_compat_handles_spelling_variants() -> None:
    cmd = Command2D(0.1, 0.0, 0.2)

    client = _FakeClient()
    assert call_move_with_sdk_compat(client, cmd) == 123
    assert client.moves[-1][1] == {"continous_move": True}

    continuous_client = _ContinuousOnlyClient()
    assert call_move_with_sdk_compat(continuous_client, cmd) == 321
    assert continuous_client.moves[-1][1] == {"continuous_move": True}

    no_kw_client = _NoKeywordClient()
    assert call_move_with_sdk_compat(no_kw_client, cmd) == 999
    assert no_kw_client.moves[-1][1] == {}


def test_call_stop_move_prefers_stopmove() -> None:
    client = _FakeClient()
    assert call_stop_move(client) == 456
    assert client.stop_count == 1
    assert client.moves == []


def test_loco_state_gate_accepts_and_rejects_expected_states() -> None:
    ok, reasons = gate_loco_state([{"fsm_id": 200, "fsm_mode": 0}, {"fsm_id": 200, "fsm_mode": 1}])
    assert ok
    assert reasons == []

    ok, reasons = gate_loco_state([{"fsm_id": 100, "fsm_mode": 0}])
    assert not ok
    assert "fsm_id" in " ".join(reasons)

    ok, reasons = gate_loco_state([{"fsm_id": 200, "fsm_mode": 2}])
    assert not ok
    assert "blocked" in " ".join(reasons)


def test_dry_run_command_sink_logs_without_hardware() -> None:
    sink = DryRunCommandSink()
    result = sink.send(Command2D(0.01, 0.0, 0.0, timestamp_s=1.0))
    stop = sink.stop(timestamp_s=2.0)

    assert result.ok
    assert stop.ok
    assert len(sink.sent_commands) == 2
    assert sink.sent_commands[-1].is_stop


def test_unitree_adapter_requires_arm_and_clips_before_send() -> None:
    client = _FakeClient()
    adapter = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(require_arm=True, require_fsm_id=200, blocked_fsm_modes=(2,)),
        command_limits=_limits(),
        client=client,
        state_reader=lambda _: {"fsm_id": 200, "fsm_mode": 0},
        clock=lambda: 10.0,
    )

    refused = adapter.send_command(Command2D(1.0, -1.0, 1.0, timestamp_s=1.0))
    assert not refused.accepted
    assert "not armed" in refused.status.message
    assert client.moves == []

    adapter.arm()
    accepted = adapter.send_command(Command2D(1.0, -1.0, 1.0, timestamp_s=2.0))
    assert accepted.ok
    assert accepted.command.tuple == pytest.approx((0.10, -0.03, 0.20))
    assert client.moves[-1][0] == pytest.approx((0.10, -0.03, 0.20))




def test_unitree_adapter_applies_final_sdk_command_mapping() -> None:
    client = _FakeClient()
    adapter = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(
            require_arm=False,
            require_fsm_id=200,
            blocked_fsm_modes=(2,),
            command_vx_scale=1.0,
            command_vy_scale=-1.0,
            command_wz_scale=-1.0,
        ),
        command_limits=_limits(),
        client=client,
        state_reader=lambda _: {"fsm_id": 200, "fsm_mode": 0},
        clock=lambda: 11.0,
    )

    accepted = adapter.send_command(Command2D(0.08, 0.02, 0.10, timestamp_s=2.0))
    assert accepted.ok
    assert accepted.command.tuple == pytest.approx((0.08, -0.02, -0.10))
    assert client.moves[-1][0] == pytest.approx((0.08, -0.02, -0.10))
    assert accepted.status.details["clipped_command"] == {"vx": 0.08, "vy": 0.02, "wz": 0.10}
    assert accepted.status.details["sdk_command"] == {"vx": 0.08, "vy": -0.02, "wz": -0.10}


def test_unitree_adapter_state_gate_rejects_bad_state_but_stop_bypasses_gate() -> None:
    client = _FakeClient()
    adapter = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(require_arm=False, require_fsm_id=200, blocked_fsm_modes=(2,)),
        client=client,
        state_reader=lambda _: {"fsm_id": 200, "fsm_mode": 2},
        clock=lambda: 20.0,
    )

    result = adapter.send_command(Command2D(0.01, 0.0, 0.0, timestamp_s=3.0))
    assert not result.accepted
    assert "state gate" in result.status.message
    assert client.moves == []

    stop = adapter.stop(timestamp_s=4.0)
    assert stop.ok
    assert client.stop_count == 1


def test_robot_state_from_loco_mapping_preserves_summary() -> None:
    state = robot_state_from_loco_mapping(
        {"fsm_id": 200, "fsm_mode": 0, "balance_mode": 1, "extra": "kept"},
        timestamp_s=5.0,
    )

    assert state.mode == "unitree_loco"
    assert state.fsm_state == 200
    assert state.details["fsm_mode"] == 0
    assert state.details["raw_loco_state"]["extra"] == "kept"
