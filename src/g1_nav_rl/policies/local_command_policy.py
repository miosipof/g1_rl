"""Common local command-policy interface for A*-conditioned navigation.

The policy seam introduced here is intentionally narrow.  Perception, target
projection, accumulated occupancy, and A* stay outside the learned policy.  A
local command policy receives a fixed-size representation of the current local
navigation problem and returns a body-frame command ``[vx, vy, wz]``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Mapping, Protocol, runtime_checkable

import torch


class TrackerState(IntEnum):
    """Compact target-tracker state encoding used by command policies."""

    TRACKING = 0
    OCCLUDED_RECENT = 1
    LOST = 2


_TRACKER_STATE_ALIASES = {
    "tracking": TrackerState.TRACKING,
    "track": TrackerState.TRACKING,
    "visible": TrackerState.TRACKING,
    "occluded_recent": TrackerState.OCCLUDED_RECENT,
    "occluded": TrackerState.OCCLUDED_RECENT,
    "recent": TrackerState.OCCLUDED_RECENT,
    "lost": TrackerState.LOST,
    "missing": TrackerState.LOST,
}


def tracker_state_from_value(value: object) -> TrackerState:
    """Normalize enum/int/string tracker-state values.

    This helper deliberately accepts loose metadata values because real adapter
    observations may come from scripts, log replays, or tracker classes.
    """

    if isinstance(value, TrackerState):
        return value
    if isinstance(value, int):
        try:
            return TrackerState(int(value))
        except ValueError as exc:
            raise ValueError(f"invalid tracker state integer: {value!r}") from exc
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
        if key in _TRACKER_STATE_ALIASES:
            return _TRACKER_STATE_ALIASES[key]
    raise ValueError(f"unsupported tracker state value: {value!r}")


@dataclass(frozen=True)
class LocalPolicyInput:
    """Batched input for an A*-conditioned local command policy.

    Tensor conventions:
    - ``occupancy_grid``: ``[N, H, W]`` local body-frame occupancy grid.
    - ``target_xy_b``: ``[N, 2]`` target in current body frame.
    - ``path_xy_b``: ``[N, K, 2]`` fixed local path samples in current body frame.
    - ``nominal_command``: ``[N, 3]`` deterministic command before learned smoothing.
    - ``last_command``: ``[N, 3]`` previous command after the local policy seam.
    - ``target_visible``: ``[N]`` boolean/float flag.
    - ``target_age_s``: ``[N]`` age of the target estimate in seconds.
    - ``tracker_state``: ``[N]`` integer-encoded ``TrackerState``.
    """

    occupancy_grid: torch.Tensor | None
    target_xy_b: torch.Tensor
    path_xy_b: torch.Tensor
    nominal_command: torch.Tensor
    last_command: torch.Tensor
    target_visible: torch.Tensor | None = None
    target_age_s: torch.Tensor | None = None
    tracker_state: torch.Tensor | None = None
    extra_features: torch.Tensor | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_envs(self) -> int:
        return int(self.nominal_command.shape[0])

    @property
    def device(self) -> torch.device:
        return self.nominal_command.device

    def validate(self) -> None:
        if self.target_xy_b.ndim != 2 or self.target_xy_b.shape[-1] != 2:
            raise ValueError(f"target_xy_b must have shape [N, 2], got {tuple(self.target_xy_b.shape)}")
        if self.path_xy_b.ndim != 3 or self.path_xy_b.shape[-1] != 2:
            raise ValueError(f"path_xy_b must have shape [N, K, 2], got {tuple(self.path_xy_b.shape)}")
        if self.nominal_command.ndim != 2 or self.nominal_command.shape[-1] != 3:
            raise ValueError(
                f"nominal_command must have shape [N, 3], got {tuple(self.nominal_command.shape)}"
            )
        if self.last_command.ndim != 2 or self.last_command.shape[-1] != 3:
            raise ValueError(f"last_command must have shape [N, 3], got {tuple(self.last_command.shape)}")

        n = self.nominal_command.shape[0]
        for name, value in (
            ("target_xy_b", self.target_xy_b),
            ("path_xy_b", self.path_xy_b),
            ("last_command", self.last_command),
        ):
            if value.shape[0] != n:
                raise ValueError(f"{name} batch size {value.shape[0]} does not match N={n}")
        if self.occupancy_grid is not None:
            if self.occupancy_grid.ndim != 3:
                raise ValueError(
                    f"occupancy_grid must have shape [N, H, W], got {tuple(self.occupancy_grid.shape)}"
                )
            if self.occupancy_grid.shape[0] != n:
                raise ValueError("occupancy_grid batch size must match nominal_command")
        for name, value in (
            ("target_visible", self.target_visible),
            ("target_age_s", self.target_age_s),
            ("tracker_state", self.tracker_state),
        ):
            if value is not None and (value.ndim != 1 or value.shape[0] != n):
                raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)}")
        if self.extra_features is not None and (
            self.extra_features.ndim != 2 or self.extra_features.shape[0] != n
        ):
            raise ValueError(f"extra_features must have shape [N, F], got {tuple(self.extra_features.shape)}")

    def to(self, device: torch.device | str, *, dtype: torch.dtype = torch.float32) -> "LocalPolicyInput":
        """Move tensor fields to ``device`` and normalize float tensors."""

        target_visible = None
        if self.target_visible is not None:
            target_visible = self.target_visible.to(device=device, dtype=torch.float32)
        target_age_s = None
        if self.target_age_s is not None:
            target_age_s = self.target_age_s.to(device=device, dtype=torch.float32)
        tracker_state = None
        if self.tracker_state is not None:
            tracker_state = self.tracker_state.to(device=device, dtype=torch.long)
        return replace(
            self,
            occupancy_grid=None
            if self.occupancy_grid is None
            else self.occupancy_grid.to(device=device, dtype=dtype),
            target_xy_b=self.target_xy_b.to(device=device, dtype=dtype),
            path_xy_b=self.path_xy_b.to(device=device, dtype=dtype),
            nominal_command=self.nominal_command.to(device=device, dtype=dtype),
            last_command=self.last_command.to(device=device, dtype=dtype),
            target_visible=target_visible,
            target_age_s=target_age_s,
            tracker_state=tracker_state,
            extra_features=None if self.extra_features is None else self.extra_features.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class LocalPolicyOutput:
    """Batched output of a local command policy."""

    command: torch.Tensor
    residual: torch.Tensor | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.command.ndim != 2 or self.command.shape[-1] != 3:
            raise ValueError(f"command must have shape [N, 3], got {tuple(self.command.shape)}")
        if self.residual is not None and self.residual.shape != self.command.shape:
            raise ValueError("residual must have same shape as command")


@runtime_checkable
class LocalCommandPolicy(Protocol):
    """Protocol implemented by deterministic and learned local policies."""

    def reset(self, *, num_envs: int | None = None, device: torch.device | str | None = None) -> None: ...

    def step(self, policy_input: LocalPolicyInput) -> LocalPolicyOutput: ...


__all__ = [
    "LocalCommandPolicy",
    "LocalPolicyInput",
    "LocalPolicyOutput",
    "TrackerState",
    "tracker_state_from_value",
]
