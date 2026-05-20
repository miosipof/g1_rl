"""High-level bridge for running g1_rl policies on real or simulated adapters.

``RealBridge`` is intentionally hardware-neutral.  It orchestrates four injected
interfaces:

- an observation provider returning :class:`RobotObservation`;
- a policy/runtime that maps observations to :class:`Command2D`;
- a :class:`SafetySupervisor` that gates commands;
- a command sink such as ``UnitreeLocoAdapter`` or ``DryRunCommandSink``.

No Isaac, Unitree, RealSense, Livox, ROS, or OpenCV modules are imported here.
Concrete providers/sinks are adapted at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .safety_supervisor import SafetyDecision, SafetySupervisor
from .types import (
    AdapterCommandResult,
    AdapterHealth,
    AdapterStatus,
    BridgeMode,
    Command2D,
    MotionCommand,
    RobotObservation,
)


Clock = Callable[[], float]


@runtime_checkable
class ObservationProvider(Protocol):
    """Minimal observation-provider interface used by ``RealBridge``."""

    def read_observation(self) -> RobotObservation:
        ...


@runtime_checkable
class PolicyRuntime(Protocol):
    """Minimal policy/runtime interface used by ``RealBridge``."""

    def step(self, observation: RobotObservation) -> Command2D | MotionCommand:
        ...


@runtime_checkable
class CommandSink(Protocol):
    """Minimal command-sink interface used by ``RealBridge``."""

    def send_command(self, command: Command2D) -> AdapterCommandResult:
        ...

    def stop(self, *, timestamp_s: float | None = None) -> AdapterCommandResult:
        ...


@dataclass(frozen=True)
class RealBridgeConfig:
    """Runtime behavior for the high-level bridge.

    ``send_commands`` can be set false to run a full live/dry policy loop without
    forwarding even safety-filtered commands to a sink.  This is stricter than a
    dry-run sink because no sink method is called except explicit stop commands.
    """

    name: str = "real_bridge"
    send_commands: bool = True
    send_stop_when_inactive: bool = True
    stop_on_observation_exception: bool = True
    stop_on_policy_exception: bool = True
    stop_on_sink_exception: bool = True
    auto_arm_sink_on_follow: bool = True
    auto_disarm_sink_on_stop: bool = False
    policy_source: str = "policy"
    inactive_source: str = "bridge_inactive_stop"
    exception_stop_source: str = "bridge_exception_stop"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RealBridgeConfig.name must be non-empty")


@dataclass(frozen=True)
class BridgeStepResult:
    """Result of one bridge loop iteration."""

    mode: BridgeMode
    status: AdapterStatus
    observation: RobotObservation | None = None
    proposed_command: Command2D | None = None
    safety_decision: SafetyDecision | None = None
    command_result: AdapterCommandResult | None = None
    action: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status.ok

    @property
    def command_sent(self) -> bool:
        return self.command_result is not None and self.command_result.accepted

    @property
    def safe_command(self) -> Command2D | None:
        if self.safety_decision is not None:
            return self.safety_decision.command
        if self.command_result is not None:
            return self.command_result.command
        return None


class RealBridge:
    """High-level follow/stop/pause bridge for sim-to-real execution.

    The bridge owns only orchestration state.  It does not know how RealSense,
    Livox, Isaac, or Unitree are implemented.  Those details are hidden behind
    injected providers and sinks.
    """

    def __init__(
        self,
        *,
        observation_provider: Any,
        policy_runtime: Any,
        safety_supervisor: SafetySupervisor,
        command_sink: Any,
        config: RealBridgeConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.observation_provider = observation_provider
        self.policy_runtime = policy_runtime
        self.safety_supervisor = safety_supervisor
        self.command_sink = command_sink
        self.config = config or RealBridgeConfig()
        self._clock = clock or time.monotonic
        self.mode = BridgeMode.IDLE
        self.operator_enabled = False
        self.last_observation: RobotObservation | None = None
        self.last_proposed_command: Command2D | None = None
        self.last_safety_decision: SafetyDecision | None = None
        self.last_command_result: AdapterCommandResult | None = None
        self.last_status = AdapterStatus(
            name=self.config.name,
            health=AdapterHealth.OK,
            timestamp_s=self._clock(),
            message="bridge initialized",
            details={"mode": self.mode.value},
        )

    def follow(self) -> AdapterStatus:
        """Enter FOLLOWING mode and enable motion through the safety supervisor."""

        if self.mode == BridgeMode.E_STOP:
            return self._set_status(
                AdapterHealth.FAULT,
                "cannot follow while bridge is in E_STOP; call reset_estop() first",
            )
        self.mode = BridgeMode.FOLLOWING
        self.operator_enabled = True
        if self.config.auto_arm_sink_on_follow:
            _call_optional_noargs(self.command_sink, "arm")
        return self._set_status(AdapterHealth.OK, "follow mode enabled")

    def pause(self) -> AdapterStatus:
        """Pause autonomy and send/log a stop command."""

        self.operator_enabled = False
        self.mode = BridgeMode.PAUSED
        self._safe_stop(source="bridge_pause")
        return self._set_status(AdapterHealth.OK, "bridge paused")

    def resume(self) -> AdapterStatus:
        """Resume after pause by entering FOLLOWING mode."""

        return self.follow()

    def stop(self) -> AdapterStatus:
        """Stop motion immediately and return to IDLE mode."""

        self.operator_enabled = False
        self.mode = BridgeMode.STOPPING
        self._safe_stop(source="bridge_stop")
        if self.config.auto_disarm_sink_on_stop:
            _call_optional_noargs(self.command_sink, "disarm")
        self.mode = BridgeMode.IDLE
        return self._set_status(AdapterHealth.OK, "bridge stopped")

    def estop(self, *, reason: str = "operator_estop") -> AdapterStatus:
        """Enter E_STOP mode and send/log a stop command."""

        self.operator_enabled = False
        self.mode = BridgeMode.E_STOP
        self._safe_stop(source="bridge_estop")
        return self._set_status(AdapterHealth.FAULT, f"bridge emergency stop: {reason}", details={"reason": reason})

    def reset_estop(self) -> AdapterStatus:
        """Clear bridge-side E_STOP state without enabling motion."""

        if self.mode != BridgeMode.E_STOP:
            return self._set_status(AdapterHealth.OK, "bridge is not in E_STOP")
        self.mode = BridgeMode.IDLE
        self.operator_enabled = False
        return self._set_status(AdapterHealth.OK, "bridge E_STOP reset; bridge is idle")

    def handle_command(self, command: str) -> AdapterStatus:
        """Handle a high-level textual command.

        Supported commands: ``follow``, ``pause``, ``resume``, ``stop``,
        ``estop``, ``reset_estop``, and ``status``.
        """

        normalized = command.strip().lower().replace("-", "_")
        if normalized in {"follow", "start"}:
            return self.follow()
        if normalized == "pause":
            return self.pause()
        if normalized == "resume":
            return self.resume()
        if normalized == "stop":
            return self.stop()
        if normalized in {"estop", "e_stop", "emergency_stop"}:
            return self.estop(reason=normalized)
        if normalized in {"reset_estop", "reset_e_stop"}:
            return self.reset_estop()
        if normalized == "status":
            return self.status()
        return self._set_status(AdapterHealth.DEGRADED, f"unknown bridge command: {command!r}")

    def status(self) -> AdapterStatus:
        return self._set_status(
            AdapterHealth.OK if self.mode != BridgeMode.E_STOP else AdapterHealth.FAULT,
            "bridge status",
            details=self.status_dict(),
        )

    def status_dict(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "mode": self.mode.value,
            "operator_enabled": self.operator_enabled,
            "send_commands": self.config.send_commands,
            "last_observation_timestamp_s": None if self.last_observation is None else self.last_observation.timestamp_s,
            "last_proposed_command": None if self.last_proposed_command is None else self.last_proposed_command.tuple,
            "last_safe_command": None if self.last_safety_decision is None else self.last_safety_decision.command.tuple,
            "last_command_accepted": None if self.last_command_result is None else self.last_command_result.accepted,
        }

    def step(self, *, now_s: float | None = None) -> BridgeStepResult:
        """Run one bridge iteration.

        In ``FOLLOWING`` mode this reads an observation, queries the policy,
        filters the command through the safety supervisor, and forwards the safe
        command to the sink if configured.  In all other non-fault modes it sends
        a stop command when ``send_stop_when_inactive`` is true.
        """

        now = float(self._clock() if now_s is None else now_s)

        if self.mode != BridgeMode.FOLLOWING:
            return self._inactive_step(now_s=now)

        try:
            observation = self._read_observation()
        except Exception as exc:
            if self.config.stop_on_observation_exception:
                result = self._safe_stop(source=self.config.exception_stop_source, timestamp_s=now)
                self.mode = BridgeMode.FAULT
                return self._result(
                    mode=self.mode,
                    status=self._exception_status("observation read failed", exc, health=AdapterHealth.FAULT),
                    command_result=result,
                    action="observation_exception_stop",
                )
            raise

        self.last_observation = observation

        try:
            proposed = self._policy_step(observation, now_s=now)
        except Exception as exc:
            if self.config.stop_on_policy_exception:
                result = self._safe_stop(source=self.config.exception_stop_source, timestamp_s=now)
                self.mode = BridgeMode.FAULT
                return self._result(
                    mode=self.mode,
                    status=self._exception_status("policy step failed", exc, health=AdapterHealth.FAULT),
                    observation=observation,
                    command_result=result,
                    action="policy_exception_stop",
                )
            raise

        self.last_proposed_command = proposed
        decision = self.safety_supervisor.filter_command(
            proposed,
            observation,
            operator_enabled=self.operator_enabled,
            now_s=now,
            policy_timestamp_s=proposed.timestamp_s,
            metadata={"bridge_mode": self.mode.value},
        )
        self.last_safety_decision = decision

        command_result: AdapterCommandResult | None = None
        if self.config.send_commands:
            try:
                command_result = self._send_command(decision.command)
            except Exception as exc:
                if self.config.stop_on_sink_exception:
                    result = self._safe_stop(source=self.config.exception_stop_source, timestamp_s=now)
                    self.mode = BridgeMode.FAULT
                    return self._result(
                        mode=self.mode,
                        status=self._exception_status("command sink failed", exc, health=AdapterHealth.FAULT),
                        observation=observation,
                        proposed_command=proposed,
                        safety_decision=decision,
                        command_result=result,
                        action="sink_exception_stop",
                    )
                raise
            self.last_command_result = command_result

        health = decision.status.health
        if command_result is not None and not command_result.ok:
            health = AdapterHealth.DEGRADED if command_result.status.health != AdapterHealth.FAULT else AdapterHealth.FAULT

        status = AdapterStatus(
            name=self.config.name,
            health=health,
            timestamp_s=now,
            message="bridge step complete",
            details={
                "mode": self.mode.value,
                "safety_reasons": decision.reasons,
                "safety_warnings": decision.warnings,
                "command_sent": command_result is not None,
                "command_accepted": None if command_result is None else command_result.accepted,
            },
        )
        self.last_status = status
        return self._result(
            mode=self.mode,
            status=status,
            observation=observation,
            proposed_command=proposed,
            safety_decision=decision,
            command_result=command_result,
            action="follow_step",
        )

    def _inactive_step(self, *, now_s: float) -> BridgeStepResult:
        if self.mode == BridgeMode.E_STOP:
            result = self._safe_stop(source="bridge_estop_hold", timestamp_s=now_s)
            return self._result(
                mode=self.mode,
                status=AdapterStatus(
                    name=self.config.name,
                    health=AdapterHealth.FAULT,
                    timestamp_s=now_s,
                    message="bridge is in E_STOP; holding stop",
                    details={"mode": self.mode.value},
                ),
                command_result=result,
                action="estop_hold",
            )

        command_result = None
        action = "inactive_noop"
        if self.config.send_stop_when_inactive:
            command_result = self._safe_stop(source=self.config.inactive_source, timestamp_s=now_s)
            action = "inactive_stop"
        status = AdapterStatus(
            name=self.config.name,
            health=AdapterHealth.OK,
            timestamp_s=now_s,
            message="bridge inactive",
            details={"mode": self.mode.value},
        )
        self.last_status = status
        return self._result(mode=self.mode, status=status, command_result=command_result, action=action)

    def _read_observation(self) -> RobotObservation:
        if hasattr(self.observation_provider, "read_observation"):
            obs = self.observation_provider.read_observation()
        elif callable(self.observation_provider):
            obs = self.observation_provider()
        else:
            raise TypeError("observation_provider must be callable or implement read_observation()")
        if not isinstance(obs, RobotObservation):
            raise TypeError(f"observation provider returned {type(obs).__name__}, expected RobotObservation")
        return obs

    def _policy_step(self, observation: RobotObservation, *, now_s: float) -> Command2D:
        if hasattr(self.policy_runtime, "step"):
            raw = self.policy_runtime.step(observation)
        elif callable(self.policy_runtime):
            raw = self.policy_runtime(observation)
        else:
            raise TypeError("policy_runtime must be callable or implement step(observation)")
        return normalize_policy_command(raw, timestamp_s=now_s, source=self.config.policy_source)

    def _send_command(self, command: Command2D) -> AdapterCommandResult:
        if hasattr(self.command_sink, "send_command"):
            return self.command_sink.send_command(command)
        if hasattr(self.command_sink, "send"):
            return self.command_sink.send(command)
        raise TypeError("command_sink must implement send_command(command) or send(command)")

    def _safe_stop(self, *, source: str, timestamp_s: float | None = None) -> AdapterCommandResult | None:
        ts = self._clock() if timestamp_s is None else timestamp_s
        try:
            if hasattr(self.command_sink, "stop"):
                result = self.command_sink.stop(timestamp_s=ts)
            else:
                result = self._send_command(Command2D.stop(timestamp_s=ts, source=source))
            self.last_command_result = result
            return result
        except Exception as exc:
            self.last_status = self._exception_status("stop command failed", exc, health=AdapterHealth.FAULT)
            if self.config.stop_on_sink_exception:
                return AdapterCommandResult(
                    accepted=False,
                    command=Command2D.stop(timestamp_s=ts, source=source),
                    status=self.last_status,
                )
            raise

    def _set_status(
        self,
        health: AdapterHealth,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> AdapterStatus:
        status = AdapterStatus(
            name=self.config.name,
            health=health,
            timestamp_s=self._clock(),
            message=message,
            details={"mode": self.mode.value, **dict(details or {})},
        )
        self.last_status = status
        return status

    def _exception_status(self, message: str, exc: Exception, *, health: AdapterHealth) -> AdapterStatus:
        status = AdapterStatus(
            name=self.config.name,
            health=health,
            timestamp_s=self._clock(),
            message=f"{message}: {exc!r}",
            details={"exception_type": type(exc).__name__, "exception": repr(exc), "mode": self.mode.value},
        )
        self.last_status = status
        return status

    @staticmethod
    def _result(**kwargs: Any) -> BridgeStepResult:
        return BridgeStepResult(**kwargs)


def normalize_policy_command(
    command: Command2D | MotionCommand,
    *,
    timestamp_s: float | None = None,
    source: str = "policy",
) -> Command2D:
    """Normalize a policy output to ``Command2D``."""

    if isinstance(command, Command2D):
        if command.timestamp_s is None and timestamp_s is not None:
            return Command2D(
                command.vx,
                command.vy,
                command.wz,
                timestamp_s=timestamp_s,
                duration_s=command.duration_s,
                source=command.source,
                metadata=command.metadata,
            )
        return command
    if isinstance(command, MotionCommand):
        return Command2D.from_motion_command(command, timestamp_s=timestamp_s, source=source)
    if all(hasattr(command, name) for name in ("vx", "vy", "wz")):
        return Command2D(
            float(getattr(command, "vx")),
            float(getattr(command, "vy")),
            float(getattr(command, "wz")),
            timestamp_s=timestamp_s,
            source=source,
        )
    raise TypeError(f"policy command must be Command2D/MotionCommand-like, got {type(command).__name__}")


def _call_optional_noargs(obj: Any, method_name: str) -> Any | None:
    method = getattr(obj, method_name, None)
    if callable(method):
        return method()
    return None


__all__ = [
    "BridgeStepResult",
    "CommandSink",
    "ObservationProvider",
    "PolicyRuntime",
    "RealBridge",
    "RealBridgeConfig",
    "normalize_policy_command",
]
