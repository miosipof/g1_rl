"""Unitree G1 locomotion command adapter.

This module provides the real-robot command sink boundary for the shared
``Command2D`` contract. It is intentionally safe to import on machines without
Unitree SDK installed: SDK imports happen only inside ``connect()``.

The implementation is based on the conservative motion path used by
``unitree_g1_vibes/g1_loco_nudge_test.py``:

- initialize ``ChannelFactoryInitialize`` + ``LocoClient`` lazily;
- tolerate the SDK spelling difference ``continous_move`` vs
  ``continuous_move``;
- expose FSM/mode readback helpers when the SDK constants are available;
- gate real motion on explicit arming and optional FSM checks;
- always provide a direct ``stop()`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

from .config import CommandLimitsConfig
from .types import AdapterCommandResult, AdapterHealth, AdapterStatus, Command2D, RobotState


StateReader = Callable[[Any], Mapping[str, Any]]
Clock = Callable[[], float]


@dataclass(frozen=True)
class UnitreeLocoAdapterConfig:
    """Configuration for a Unitree G1 locomotion command sink.

    ``dry_run`` logs commands instead of sending them to the SDK client.
    ``require_arm`` prevents accidental motion until ``arm()`` is called.
    """

    iface: str = "eth0"
    timeout_s: float = 10.0
    command_period_s: float = 0.10
    dry_run: bool = False
    require_arm: bool = True
    require_fsm_id: int | None = 200
    blocked_fsm_modes: tuple[int, ...] = (2,)
    state_samples: int = 1
    state_interval_s: float = 0.05
    stop_on_close: bool = True

    def __post_init__(self) -> None:
        if not self.iface:
            raise ValueError("UnitreeLocoAdapterConfig.iface must be non-empty")
        if self.timeout_s <= 0.0 or not math.isfinite(self.timeout_s):
            raise ValueError("timeout_s must be positive and finite")
        if self.command_period_s <= 0.0 or self.command_period_s > 1.0:
            raise ValueError("command_period_s must be > 0 and <= 1.0")
        if self.state_samples < 1:
            raise ValueError("state_samples must be >= 1")
        if self.state_interval_s < 0.0 or not math.isfinite(self.state_interval_s):
            raise ValueError("state_interval_s must be non-negative and finite")
        object.__setattr__(self, "blocked_fsm_modes", tuple(int(v) for v in self.blocked_fsm_modes))


@dataclass
class DryRunCommandSink:
    """In-memory command sink used for dry-run and tests."""

    name: str = "dry_run_unitree_loco"
    sent_commands: list[Command2D] = field(default_factory=list)
    stop_count: int = 0

    def send(self, command: Command2D) -> AdapterCommandResult:
        self.sent_commands.append(command)
        return AdapterCommandResult(
            accepted=True,
            command=command,
            status=AdapterStatus(
                name=self.name,
                health=AdapterHealth.OK,
                timestamp_s=command.timestamp_s,
                message="dry-run command logged; no robot command was sent",
                details={"dry_run": True},
            ),
        )

    def stop(self, *, timestamp_s: float | None = None) -> AdapterCommandResult:
        self.stop_count += 1
        cmd = Command2D.stop(timestamp_s=timestamp_s, source="dry_run_stop")
        self.sent_commands.append(cmd)
        return AdapterCommandResult(
            accepted=True,
            command=cmd,
            status=AdapterStatus(
                name=self.name,
                health=AdapterHealth.OK,
                timestamp_s=timestamp_s,
                message="dry-run stop logged; no robot command was sent",
                details={"dry_run": True, "stop_count": self.stop_count},
            ),
        )


def call_move_with_sdk_compat(client: Any, command: Command2D) -> Any:
    """Call Unitree ``Move`` while tolerating SDK keyword spelling variants."""

    move = getattr(client, "Move")
    try:
        return move(command.vx, command.vy, command.wz, continous_move=True)
    except TypeError:
        try:
            return move(command.vx, command.vy, command.wz, continuous_move=True)
        except TypeError:
            return move(command.vx, command.vy, command.wz)


def call_stop_move(client: Any) -> Any:
    """Call the safest available stop method on a Unitree loco client."""

    if hasattr(client, "StopMove"):
        return client.StopMove()
    if hasattr(client, "Move"):
        return call_move_with_sdk_compat(client, Command2D.stop(source="stop_fallback_move_zero"))
    raise AttributeError("Unitree loco client has neither StopMove nor Move")


def load_loco_api_constants() -> dict[str, int]:
    """Load known G1 loco RPC API IDs when Unitree SDK is installed."""

    out: dict[str, int] = {}
    try:
        from unitree_sdk2py.g1.loco import g1_loco_api as api  # type: ignore
    except Exception:
        return out

    names = {
        "fsm_id": "ROBOT_API_ID_LOCO_GET_FSM_ID",
        "fsm_mode": "ROBOT_API_ID_LOCO_GET_FSM_MODE",
        "balance_mode": "ROBOT_API_ID_LOCO_GET_BALANCE_MODE",
        "stand_height": "ROBOT_API_ID_LOCO_GET_STAND_HEIGHT",
        "swing_height": "ROBOT_API_ID_LOCO_GET_SWING_HEIGHT",
        "phase": "ROBOT_API_ID_LOCO_GET_PHASE",
    }
    for key, const_name in names.items():
        value = getattr(api, const_name, None)
        if isinstance(value, int):
            out[key] = value
    return out


def rpc_call_json(client: Any, api_id: int, payload: str = "{}") -> tuple[int | None, Any, str]:
    """Call private ``LocoClient._Call`` for readbacks; tolerate SDK variants."""

    try:
        code, data = client._Call(api_id, payload)  # type: ignore[attr-defined]
    except Exception as exc:
        return None, None, f"_Call exception: {exc!r}"

    parsed: Any = None
    err = ""
    try:
        if data:
            parsed = json.loads(data)
    except Exception as exc:
        err = f"json parse exception: {exc!r}; raw={data!r}"
    return code, parsed, err


def _extract_rpc_data(parsed: Any) -> Any:
    if isinstance(parsed, Mapping) and "data" in parsed:
        return parsed["data"]
    return parsed


def read_loco_state(client: Any) -> dict[str, Any]:
    """Read available Unitree G1 loco state via SDK RPC constants.

    If the SDK constants are not available, the returned mapping still contains
    ``api_ids_found`` and ``reads`` so callers can distinguish unavailable state
    from a true motion-ready state.
    """

    constants = load_loco_api_constants()
    state: dict[str, Any] = {"api_ids_found": constants, "reads": {}}
    for name, api_id in constants.items():
        code, parsed, err = rpc_call_json(client, api_id)
        state["reads"][name] = {
            "api_id": api_id,
            "code": code,
            "raw": parsed,
            "data": _extract_rpc_data(parsed),
            "error": err,
        }
        state[name] = _extract_rpc_data(parsed)
    return state


def summarize_loco_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small state subset relevant for locomotion gating."""

    return {
        "fsm_id": state.get("fsm_id"),
        "fsm_mode": state.get("fsm_mode"),
        "balance_mode": state.get("balance_mode"),
        "stand_height": state.get("stand_height"),
        "swing_height": state.get("swing_height"),
        "phase": state.get("phase"),
    }


def gate_loco_state(
    states: Sequence[Mapping[str, Any]],
    *,
    require_fsm_id: int | None = 200,
    blocked_fsm_modes: Sequence[int] = (2,),
) -> tuple[bool, list[str]]:
    """Validate Unitree loco state samples before real motion."""

    reasons: list[str] = []
    if not states:
        return False, ["no state samples"]

    fsm_ids = [s.get("fsm_id") for s in states]
    fsm_modes = [s.get("fsm_mode") for s in states]

    if require_fsm_id is not None:
        if any(v is None for v in fsm_ids):
            reasons.append("could not read fsm_id in all samples")
        if any(v != require_fsm_id for v in fsm_ids):
            reasons.append(f"fsm_id is not consistently {require_fsm_id}: {fsm_ids}")

    blocked = set(int(v) for v in blocked_fsm_modes)
    if blocked:
        if any(v is None for v in fsm_modes):
            reasons.append("could not read fsm_mode in all samples")
        bad_modes = [v for v in fsm_modes if v in blocked]
        if bad_modes:
            reasons.append(f"fsm_mode includes blocked value(s) {sorted(blocked)}: {fsm_modes}")

    return len(reasons) == 0, reasons


def robot_state_from_loco_mapping(
    state: Mapping[str, Any],
    *,
    timestamp_s: float,
    pose: Any = None,
) -> RobotState:
    """Convert a loco-state mapping into the shared ``RobotState`` type."""

    return RobotState(
        timestamp_s=timestamp_s,
        pose=pose,
        mode="unitree_loco",
        fsm_state=state.get("fsm_id"),
        is_faulted=False,
        details=dict(summarize_loco_state(state)) | {"raw_loco_state": dict(state)},
    )


class UnitreeLocoAdapter:
    """Safe-ish command sink for Unitree G1 planar locomotion.

    The adapter may be constructed with an already-created ``client`` for tests
    or external ownership. If no client is supplied, ``connect()`` lazily imports
    Unitree SDK, initializes the channel factory, and creates ``LocoClient``.
    """

    def __init__(
        self,
        *,
        config: UnitreeLocoAdapterConfig | None = None,
        command_limits: CommandLimitsConfig | None = None,
        client: Any | None = None,
        state_reader: StateReader | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config or UnitreeLocoAdapterConfig()
        self.command_limits = command_limits
        self._client = client
        self._state_reader = state_reader or read_loco_state
        self._clock = clock or time.monotonic
        self._armed = not self.config.require_arm
        self._connected = client is not None
        self.command_log: list[Command2D] = []
        self.status = AdapterStatus(
            name="unitree_loco",
            health=AdapterHealth.OK if self.config.dry_run else AdapterHealth.UNAVAILABLE,
            timestamp_s=self._clock(),
            message="initialized dry-run adapter" if self.config.dry_run else "not connected",
            details={"dry_run": self.config.dry_run},
        )
        self._dry_sink = DryRunCommandSink() if self.config.dry_run else None

    @property
    def client(self) -> Any | None:
        return self._client

    @property
    def armed(self) -> bool:
        return bool(self._armed)

    @property
    def connected(self) -> bool:
        return bool(self._connected or self.config.dry_run)

    def arm(self) -> AdapterStatus:
        self._armed = True
        self.status = AdapterStatus(
            name="unitree_loco",
            health=AdapterHealth.OK,
            timestamp_s=self._clock(),
            message="adapter armed",
            details={"dry_run": self.config.dry_run},
        )
        return self.status

    def disarm(self) -> AdapterStatus:
        self._armed = False
        self.status = AdapterStatus(
            name="unitree_loco",
            health=AdapterHealth.OK,
            timestamp_s=self._clock(),
            message="adapter disarmed",
            details={"dry_run": self.config.dry_run},
        )
        return self.status

    def connect(self) -> AdapterStatus:
        """Connect to Unitree SDK unless already connected or in dry-run mode."""

        if self.config.dry_run:
            self._connected = True
            self.status = AdapterStatus(
                name="unitree_loco",
                health=AdapterHealth.OK,
                timestamp_s=self._clock(),
                message="dry-run adapter connected",
                details={"dry_run": True},
            )
            return self.status

        if self._client is not None:
            self._connected = True
            self.status = AdapterStatus(
                name="unitree_loco",
                health=AdapterHealth.OK,
                timestamp_s=self._clock(),
                message="using injected Unitree loco client",
            )
            return self.status

        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient  # type: ignore

            ChannelFactoryInitialize(0, self.config.iface)
            client = LocoClient()
            client.SetTimeout(self.config.timeout_s)
            client.Init()
            self._client = client
            self._connected = True
            self.status = AdapterStatus(
                name="unitree_loco",
                health=AdapterHealth.OK,
                timestamp_s=self._clock(),
                message="connected to Unitree LocoClient",
                details={"iface": self.config.iface},
            )
        except Exception as exc:  # pragma: no cover - environment/hardware dependent
            self._connected = False
            self.status = AdapterStatus(
                name="unitree_loco",
                health=AdapterHealth.FAULT,
                timestamp_s=self._clock(),
                message=f"failed to connect Unitree LocoClient: {exc!r}",
                details={"exception_type": type(exc).__name__, "exception": repr(exc)},
            )
        return self.status

    def sample_state(self) -> list[Mapping[str, Any]]:
        """Read one or more loco-state samples through the configured reader."""

        if self.config.dry_run:
            return [
                {
                    "fsm_id": self.config.require_fsm_id,
                    "fsm_mode": 0,
                    "dry_run": True,
                    "sample_index": i,
                }
                for i in range(self.config.state_samples)
            ]

        if self._client is None:
            raise RuntimeError("UnitreeLocoAdapter is not connected")

        samples: list[Mapping[str, Any]] = []
        for i in range(self.config.state_samples):
            sample = dict(self._state_reader(self._client))
            sample["sample_index"] = i
            samples.append(sample)
            if i + 1 < self.config.state_samples:
                time.sleep(self.config.state_interval_s)
        return samples

    def check_state_gate(self) -> tuple[bool, list[str], list[Mapping[str, Any]]]:
        samples = self.sample_state()
        ok, reasons = gate_loco_state(
            samples,
            require_fsm_id=self.config.require_fsm_id,
            blocked_fsm_modes=self.config.blocked_fsm_modes,
        )
        return ok, reasons, samples

    def send_command(self, command: Command2D, *, require_state_gate: bool = True) -> AdapterCommandResult:
        """Send a bounded command to the real or dry-run command sink."""

        clipped = self.command_limits.clip(command) if self.command_limits is not None else command
        self.command_log.append(clipped)

        if self.config.require_arm and not self._armed:
            return self._reject(clipped, "adapter is not armed", health=AdapterHealth.DEGRADED)

        if not self.connected:
            status = self.connect()
            if not status.ok:
                return AdapterCommandResult(False, clipped, status)

        if require_state_gate:
            try:
                ok, reasons, samples = self.check_state_gate()
            except Exception as exc:
                return self._reject(
                    clipped,
                    f"state gate read failed: {exc!r}",
                    health=AdapterHealth.FAULT,
                    details={"exception_type": type(exc).__name__, "exception": repr(exc)},
                )
            if not ok:
                return self._reject(
                    clipped,
                    "state gate rejected motion",
                    health=AdapterHealth.DEGRADED,
                    details={"reasons": reasons, "samples": [dict(s) for s in samples]},
                )

        if self.config.dry_run:
            assert self._dry_sink is not None
            return self._dry_sink.send(clipped)

        if self._client is None:
            return self._reject(clipped, "Unitree client unavailable", health=AdapterHealth.FAULT)

        try:
            code = call_move_with_sdk_compat(self._client, clipped)
            return AdapterCommandResult(
                accepted=True,
                command=clipped,
                status=AdapterStatus(
                    name="unitree_loco",
                    health=AdapterHealth.OK,
                    timestamp_s=clipped.timestamp_s if clipped.timestamp_s is not None else self._clock(),
                    message="Move command sent",
                    details={"code": code, "dry_run": False},
                ),
            )
        except Exception as exc:
            return AdapterCommandResult(
                accepted=False,
                command=clipped,
                status=AdapterStatus(
                    name="unitree_loco",
                    health=AdapterHealth.FAULT,
                    timestamp_s=self._clock(),
                    message=f"Move command failed: {exc!r}",
                    details={"exception_type": type(exc).__name__, "exception": repr(exc)},
                ),
            )

    def stop(self, *, timestamp_s: float | None = None) -> AdapterCommandResult:
        """Send or log a stop command, bypassing arming and state gates."""

        ts = timestamp_s if timestamp_s is not None else self._clock()
        if self.config.dry_run:
            assert self._dry_sink is not None
            return self._dry_sink.stop(timestamp_s=ts)

        if not self.connected:
            status = self.connect()
            if not status.ok:
                return AdapterCommandResult(False, Command2D.stop(timestamp_s=ts), status)

        if self._client is None:
            return AdapterCommandResult(
                accepted=False,
                command=Command2D.stop(timestamp_s=ts),
                status=AdapterStatus(
                    name="unitree_loco",
                    health=AdapterHealth.FAULT,
                    timestamp_s=ts,
                    message="Unitree client unavailable for stop",
                ),
            )

        try:
            code = call_stop_move(self._client)
            return AdapterCommandResult(
                accepted=True,
                command=Command2D.stop(timestamp_s=ts, source="unitree_stop"),
                status=AdapterStatus(
                    name="unitree_loco",
                    health=AdapterHealth.OK,
                    timestamp_s=ts,
                    message="stop command sent",
                    details={"code": code},
                ),
            )
        except Exception as exc:
            return AdapterCommandResult(
                accepted=False,
                command=Command2D.stop(timestamp_s=ts),
                status=AdapterStatus(
                    name="unitree_loco",
                    health=AdapterHealth.FAULT,
                    timestamp_s=ts,
                    message=f"stop command failed: {exc!r}",
                    details={"exception_type": type(exc).__name__, "exception": repr(exc)},
                ),
            )

    def close(self) -> AdapterCommandResult | None:
        """Optional cleanup hook. Sends stop when configured and connected."""

        if self.config.stop_on_close and self.connected:
            return self.stop()
        return None

    def _reject(
        self,
        command: Command2D,
        message: str,
        *,
        health: AdapterHealth,
        details: Mapping[str, Any] | None = None,
    ) -> AdapterCommandResult:
        return AdapterCommandResult(
            accepted=False,
            command=command,
            status=AdapterStatus(
                name="unitree_loco",
                health=health,
                timestamp_s=command.timestamp_s if command.timestamp_s is not None else self._clock(),
                message=message,
                details=dict(details or {}),
            ),
        )


__all__ = [
    "DryRunCommandSink",
    "UnitreeLocoAdapter",
    "UnitreeLocoAdapterConfig",
    "call_move_with_sdk_compat",
    "call_stop_move",
    "gate_loco_state",
    "load_loco_api_constants",
    "read_loco_state",
    "robot_state_from_loco_mapping",
    "rpc_call_json",
    "summarize_loco_state",
]
