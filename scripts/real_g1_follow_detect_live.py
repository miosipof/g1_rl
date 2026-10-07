#!/usr/bin/env python3
"""Live RGB-D + LiDAR target following bridge for real Unitree G1 tests.

This is the first real-motion-capable detect-and-follow entry point.  It keeps
real motion disabled by default.  When real motion is enabled, commands still pass
through the shared SafetySupervisor and the UnitreeLocoAdapter FSM/state gate.

Pipeline:

    RealSense RGB-D + Livox/MID-360 LiDAR
      -> YOLO person detection + bbox/depth target projection
      -> target smoothing / occlusion grace
      -> LiDAR target-nav policy
      -> SafetySupervisor
      -> DryRunCommandSink or UnitreeLocoAdapter

The script currently uses a local body-frame target.  Robot odometry/SLAM pose is
not yet wired into RobotObservation, so the target is refreshed from RGB-D while
visible and kept only for a bounded occlusion grace interval.  Use low speeds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from adapters import (
    Command2D,
    CommandOdometryConfig,
    CommandOdometryEstimator,
    CommandOdometryObservationProvider,
    CommandOdometryTargetPropagationProvider,
    CompositeDryRunObservationProvider,
    DryRunCommandSink,
    LidarDebugWriter,
    LidarTargetNavPolicyRuntime,
    LidarTargetNavPolicyRuntimeConfig,
    Pose2D,
    RGBDDebugWriter,
    RealBridge,
    RealBridgeConfig,
    RealSenseRGBDAdapter,
    RealSenseRGBDAdapterConfig,
    RobotObservation,
    Target2D,
    make_run_id,
    timestamped_output_dir,
    write_run_info,
    UnitreeLocoAdapter,
    UnitreeLocoAdapterConfig,
    load_adapter_config,
)
from adapters.real_livox import LivoxAdapterConfig, UnitreeVibesLivox2Adapter, UnitreeVibesLivox2AdapterConfig
from adapters.safety_supervisor import SafetySupervisor, SafetySupervisorConfig
from g1_nav_rl.perception.target_from_bbox import (
    Detection2D,
    TargetFromBBoxConfig,
    bbox_depth_to_target2d,
    make_forward_camera_T_body_camera,
    select_detection_for_target,
)
from g1_nav_rl.perception.yolo_detector import YoloDetector, YoloDetectorConfig, filter_detections

# Reuse small helpers from the dry-run script.
from real_g1_follow_dryrun import _TimeoutRealSenseAdapter, _result_to_dict


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


@dataclass(frozen=True)
class TargetTrackingStatus:
    state: str
    reason: str
    active_target: tuple[float, float] | None
    raw_target: tuple[float, float] | None = None
    detection_count: int = 0
    selected_detection: dict[str, Any] | None = None
    target_depth_m: float | None = None
    object_xy_b: tuple[float, float] | None = None
    missed_steps: int = 0
    rejected_jump_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "active_target": None if self.active_target is None else [float(v) for v in self.active_target],
            "raw_target": None if self.raw_target is None else [float(v) for v in self.raw_target],
            "detection_count": int(self.detection_count),
            "selected_detection": self.selected_detection,
            "target_depth_m": self.target_depth_m,
            "object_xy_b": None if self.object_xy_b is None else [float(v) for v in self.object_xy_b],
            "missed_steps": int(self.missed_steps),
            "rejected_jump_m": self.rejected_jump_m,
        }


class DetectTargetObservationProvider:
    """Inject YOLO/RGB-D target hints into RobotObservation.

    The wrapped provider supplies RGB-D, LiDAR, and robot-state placeholders.  This
    class periodically runs YOLO, converts the selected bbox to a local Target2D,
    smooths target updates, and keeps the last target during temporary occlusion.
    """

    def __init__(
        self,
        *,
        base_provider: Any,
        yolo_model: str,
        yolo_confidence: float,
        yolo_iou: float,
        yolo_device: str | None,
        target_class: str,
        redetect_every_steps: int,
        occluded_redetect_every_steps: int,
        lost_redetect_every_steps: int,
        occlusion_grace_steps: int,
        smoothing_alpha: float,
        max_jump_m: float,
        stand_off_m: float,
        depth_min_m: float,
        depth_max_m: float,
        depth_crop_frac: float,
        depth_percentile: float,
        use_forward_camera_convention: bool = True,
        rgbd_body_exclusion_box: Any | None = None,
    ) -> None:
        self.base_provider = base_provider
        self.target_class = str(target_class)
        self.redetect_every_steps = int(redetect_every_steps)
        self.occluded_redetect_every_steps = int(occluded_redetect_every_steps)
        self.lost_redetect_every_steps = int(lost_redetect_every_steps)
        self.occlusion_grace_steps = int(occlusion_grace_steps)
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_jump_m = float(max_jump_m)
        self.use_forward_camera_convention = bool(use_forward_camera_convention)
        self.rgbd_body_exclusion_box = rgbd_body_exclusion_box
        self.target_cfg = TargetFromBBoxConfig(
            depth_min_m=float(depth_min_m),
            depth_max_m=float(depth_max_m),
            bbox_center_crop_frac=float(depth_crop_frac),
            depth_percentile=float(depth_percentile),
            stand_off_distance_m=float(stand_off_m),
            min_target_distance_m=0.05,
        )
        self.detector = YoloDetector(
            YoloDetectorConfig(
                model=yolo_model,
                confidence=float(yolo_confidence),
                iou=float(yolo_iou),
                device=yolo_device,
                verbose=False,
            )
        )
        self.step_index = 0
        self.last_seen_step: int | None = None
        self.active_target: Target2D | None = None
        self.last_status = TargetTrackingStatus(state="lost", reason="not_initialized", active_target=None)
        self.last_detections: list[Detection2D] = []

    def invalidate_body_target(self, *, reason: str = "robot_motion") -> None:
        """Drop coordinates measured in the pre-motion body frame."""

        self.active_target = None
        self.last_seen_step = None
        self.last_detections = []
        self.last_status = TargetTrackingStatus(state="lost", reason=str(reason), active_target=None)

    def read_observation(self) -> RobotObservation:
        obs: RobotObservation = self.base_provider.read_observation()
        redetect_interval = self._current_redetect_interval()
        should_detect = self.active_target is None or redetect_interval <= 1 or (
            self.step_index % redetect_interval == 0
        )
        status: TargetTrackingStatus
        if should_detect:
            measurement, detections, status = self._detect_target(obs)
            self.last_detections = detections
            status = self._update_tracker(measurement, status)
        else:
            status = self._miss_status("redetect_not_due")
        self.last_status = status

        active = None if status.active_target is None else Target2D(status.active_target[0], status.active_target[1])
        meta = dict(obs.metadata)
        meta["target_tracking"] = status.to_dict()
        meta["target_tracking_step"] = self.step_index
        meta["target_redetect_due"] = should_detect
        meta["target_redetect_interval"] = redetect_interval
        out = replace(obs, target_hint=active, metadata=meta)
        self.step_index += 1
        return out

    def _current_redetect_interval(self) -> int:
        # When target is lost or recently occluded, do not wait for the normal
        # tracking cadence.  Force frequent reacquisition attempts so the robot
        # does not abort just because the next scheduled redetect is not due.
        state = self.last_status.state
        if state == "lost":
            return max(1, int(self.lost_redetect_every_steps))
        if state == "occluded_recent":
            return max(1, int(self.occluded_redetect_every_steps))
        return max(1, int(self.redetect_every_steps))

    def _detect_target(self, obs: RobotObservation) -> tuple[Target2D | None, list[Detection2D], TargetTrackingStatus]:
        if obs.rgbd is None:
            return None, [], TargetTrackingStatus(
                state="occluded_recent" if self.active_target is not None else "lost",
                reason="missing_rgbd",
                active_target=None if self.active_target is None else self.active_target.xy,
            )
        frame = obs.rgbd
        detections = self.detector.detect(frame.rgb)
        detections = filter_detections(
            detections,
            allowed_classes={self.target_class},
            min_confidence=float(self.detector.config.confidence),
        )
        selected = select_detection_for_target(
            detections,
            allowed_classes={self.target_class},
            min_confidence=float(self.detector.config.confidence),
            policy="nearest",
            depth_image=frame.depth_m,
            depth_min_m=self.target_cfg.depth_min_m,
            depth_max_m=self.target_cfg.depth_max_m,
        )
        if selected is None:
            return None, detections, TargetTrackingStatus(
                state="occluded_recent" if self.active_target is not None else "lost",
                reason="no_detection",
                active_target=None if self.active_target is None else self.active_target.xy,
                detection_count=len(detections),
            )

        T_body_camera = self._T_body_camera_matrix(frame)
        result = bbox_depth_to_target2d(
            selected,
            frame.depth_m,
            frame.intrinsics,
            T_body_camera,
            robot_pose=Pose2D(0.0, 0.0, 0.0),
            config=self.target_cfg,
        )
        selected_dict = {
            "class_name": selected.class_name,
            "confidence": float(selected.confidence),
            "bbox_xyxy": [float(v) for v in selected.bbox_xyxy],
        }
        if result is None:
            return None, detections, TargetTrackingStatus(
                state="occluded_recent" if self.active_target is not None else "lost",
                reason="no_valid_depth",
                active_target=None if self.active_target is None else self.active_target.xy,
                detection_count=len(detections),
                selected_detection=selected_dict,
            )
        if self.rgbd_body_exclusion_box is not None and self.rgbd_body_exclusion_box.contains_point(result.point_world):
            return None, detections, TargetTrackingStatus(
                state="occluded_recent" if self.active_target is not None else "lost",
                reason="rgbd_body_exclusion",
                active_target=None if self.active_target is None else self.active_target.xy,
                raw_target=result.target.xy,
                detection_count=len(detections),
                selected_detection=selected_dict,
                target_depth_m=float(result.depth_m),
                object_xy_b=result.object_xy_w,
            )
        return result.target, detections, TargetTrackingStatus(
            state="tracking",
            reason="measurement",
            active_target=result.target.xy,
            raw_target=result.target.xy,
            detection_count=len(detections),
            selected_detection=selected_dict,
            target_depth_m=float(result.depth_m),
            object_xy_b=result.object_xy_w,
        )

    def _T_body_camera_matrix(self, frame: Any) -> np.ndarray:
        matrix = np.asarray(frame.T_body_camera.matrix, dtype=np.float64)
        if not self.use_forward_camera_convention:
            return matrix
        translation = matrix[:3, 3]
        return make_forward_camera_T_body_camera(translation_body=translation)

    def _update_tracker(self, measurement: Target2D | None, status: TargetTrackingStatus) -> TargetTrackingStatus:
        if measurement is None:
            return self._miss_status(status.reason, base=status)

        raw_xy = measurement.xy
        rejected_jump: float | None = None
        if self.active_target is not None:
            jump = float(np.hypot(raw_xy[0] - self.active_target.x, raw_xy[1] - self.active_target.y))
            if jump > self.max_jump_m:
                rejected_jump = jump
                return self._miss_status("jump_rejected", base=replace_status(status, rejected_jump_m=rejected_jump))
            alpha = self.smoothing_alpha
            smoothed = Target2D(
                (1.0 - alpha) * self.active_target.x + alpha * raw_xy[0],
                (1.0 - alpha) * self.active_target.y + alpha * raw_xy[1],
            )
        else:
            smoothed = measurement

        self.active_target = smoothed
        self.last_seen_step = self.step_index
        return TargetTrackingStatus(
            state="tracking",
            reason="measurement_smoothed" if raw_xy != smoothed.xy else "measurement",
            active_target=smoothed.xy,
            raw_target=raw_xy,
            detection_count=status.detection_count,
            selected_detection=status.selected_detection,
            target_depth_m=status.target_depth_m,
            object_xy_b=status.object_xy_b,
            missed_steps=0,
            rejected_jump_m=rejected_jump,
        )

    def _miss_status(self, reason: str, *, base: TargetTrackingStatus | None = None) -> TargetTrackingStatus:
        if self.active_target is None or self.last_seen_step is None:
            return TargetTrackingStatus(
                state="lost",
                reason=reason,
                active_target=None,
                detection_count=0 if base is None else base.detection_count,
                selected_detection=None if base is None else base.selected_detection,
                rejected_jump_m=None if base is None else base.rejected_jump_m,
            )
        missed = max(0, self.step_index - self.last_seen_step)
        if missed <= self.occlusion_grace_steps:
            return TargetTrackingStatus(
                state="occluded_recent",
                reason=reason,
                active_target=self.active_target.xy,
                detection_count=0 if base is None else base.detection_count,
                selected_detection=None if base is None else base.selected_detection,
                missed_steps=missed,
                rejected_jump_m=None if base is None else base.rejected_jump_m,
            )
        # Drop the stale body-frame target once the grace interval expires.
        # Command-odometry propagation still keeps a world-frame target only while
        # the wrapped observation exposes an active target.
        self.active_target = None
        return TargetTrackingStatus(
            state="lost",
            reason=reason,
            active_target=None,
            detection_count=0 if base is None else base.detection_count,
            selected_detection=None if base is None else base.selected_detection,
            missed_steps=missed,
            rejected_jump_m=None if base is None else base.rejected_jump_m,
        )


@dataclass(frozen=True)
class replace_status:
    # Tiny immutable helper used to pass a modified field into _miss_status
    src: TargetTrackingStatus
    rejected_jump_m: float | None = None

    @property
    def reason(self) -> str:
        return self.src.reason

    @property
    def state(self) -> str:
        return self.src.state

    @property
    def active_target(self) -> tuple[float, float] | None:
        return self.src.active_target

    @property
    def detection_count(self) -> int:
        return self.src.detection_count

    @property
    def selected_detection(self) -> dict[str, Any] | None:
        return self.src.selected_detection


class StopUnlessTargetPolicy:
    """Return zero command until a target hint is available, then call target-nav."""

    def __init__(self, nav_policy: LidarTargetNavPolicyRuntime) -> None:
        self.nav_policy = nav_policy
        self.last_debug: Any | None = None

    @property
    def runtime(self) -> Any:
        return self.nav_policy.runtime

    @property
    def config(self) -> Any:
        return self.nav_policy.config

    def step(self, observation: RobotObservation) -> Command2D:
        if observation.target_hint is None:
            return Command2D.stop(
                timestamp_s=observation.timestamp_s,
                source="detect_follow_no_target_stop",
            )
        cmd = self.nav_policy.step(observation)
        self.last_debug = self.nav_policy.last_debug
        return cmd


def _parse_classes(value: str) -> tuple[str, ...] | None:
    if value.strip().lower() == "all":
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--iface", default="eth0", help="Network interface for Unitree SDK ChannelFactoryInitialize.")
    parser.add_argument("--steps", type=int, default=30, help="Number of bridge iterations.")
    parser.add_argument("--dt", type=float, default=0.10, help="Sleep between iterations.")
    parser.add_argument("--unitree-vibes-dir", required=True, help="Path to unitree_g1_vibes repo for live Livox mode.")
    parser.add_argument("--livox-config", required=True, help="Livox-SDK2 JSON config path.")
    parser.add_argument("--host-ip", default="192.168.123.222", help="Host IP passed to Livox2.")
    parser.add_argument("--livox-frame-time", type=float, default=0.20, help="Livox aggregation window.")
    parser.add_argument("--livox-frame-packets", type=int, default=120, help="Livox aggregation packet count.")
    parser.add_argument("--livox-timeout", type=float, default=2.0, help="Timeout waiting for live Livox frames.")
    parser.add_argument("--realsense-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--realsense-timeout-ms", type=int, default=1000, help="RealSense wait_for_frames timeout.")
    parser.add_argument("--no-realsense-spatial-filter", action="store_true", help="Disable RealSense spatial filter.")
    parser.add_argument("--no-realsense-temporal-filter", action="store_true", help="Disable RealSense temporal filter.")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path/name.")
    parser.add_argument("--yolo-confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--yolo-device", default="cpu", help="YOLO device. Default: cpu for Jetson compatibility.")
    parser.add_argument("--target-class", default="person", help="YOLO class to follow.")
    parser.add_argument("--redetect-every", type=int, default=5, help="Run YOLO every N bridge steps while actively tracking.")
    parser.add_argument("--occluded-redetect-every", type=int, default=1, help="Run YOLO every N bridge steps while target is occluded_recent. Default: every step.")
    parser.add_argument("--lost-redetect-every", type=int, default=1, help="Run YOLO every N bridge steps while target is lost. Default: every step.")
    parser.add_argument("--occlusion-grace-steps", type=int, default=50, help="Keep last target for this many missed steps.")
    parser.add_argument("--target-smoothing-alpha", type=float, default=0.35, help="Target low-pass smoothing alpha.")
    parser.add_argument("--target-max-jump-m", type=float, default=1.25, help="Reject target jumps larger than this distance.")
    parser.add_argument("--stand-off-m", type=float, default=0.80, help="Standoff from detected person.")
    parser.add_argument("--target-depth-min-m", type=float, default=0.20, help="Minimum valid target depth.")
    parser.add_argument("--target-depth-max-m", type=float, default=8.0, help="Maximum valid target depth.")
    parser.add_argument("--depth-crop-frac", type=float, default=0.50, help="Central bbox crop fraction for depth extraction.")
    parser.add_argument("--depth-percentile", type=float, default=0.50, help="Depth percentile inside bbox crop.")
    parser.add_argument("--use-config-camera-transform", action="store_true", help="Use config T_body_camera exactly instead of optical forward-camera convention.")
    parser.add_argument("--target-nav-replan-every", type=int, default=0, help="Replan cadence; 0 disables periodic replanning.")
    parser.add_argument("--target-nav-enable-replanning", action="store_true", help="Enable target-nav replanning.")
    parser.add_argument("--target-nav-inflation", type=float, default=0.35, help="A* inflation radius in meters.")
    parser.add_argument("--target-nav-max-wz", type=float, default=None, help="Override target-nav policy max yaw rate before final Unitree command scaling.")
    parser.add_argument("--target-nav-command-mode", choices=("holonomic", "heading_aligned"), default="holonomic", help="Waypoint-to-command controller. heading_aligned prefers vx+wz and damps vy for real G1 tests.")
    parser.add_argument("--heading-lookahead-m", type=float, default=0.60, help="Path lookahead distance used by heading_aligned mode to smooth local grid-cell turns.")
    parser.add_argument("--heading-slow-angle-rad", type=float, default=0.35, help="Bearing error above which heading_aligned starts reducing vx/vy.")
    parser.add_argument("--heading-rotate-in-place-angle-rad", type=float, default=0.80, help="Bearing error above which heading_aligned commands turn-first behavior.")
    parser.add_argument("--heading-rotate-vx-scale", type=float, default=0.0, help="Forward scale during turn-first heading alignment.")
    parser.add_argument("--heading-rotate-lateral-scale", type=float, default=0.0, help="Lateral scale during turn-first heading alignment.")
    parser.add_argument("--heading-lateral-scale", type=float, default=0.20, help="Policy-level lateral damping used by --target-nav-command-mode heading_aligned.")
    parser.add_argument("--heading-min-vx-scale", type=float, default=0.25, help="Minimum forward scale used by heading_aligned mode when bearing error is large.")
    parser.add_argument("--enable-command-odometry", action="store_true", help="Inject simple command-integrated pose into observations for short-horizon real tests.")
    parser.add_argument("--command-odom-max-dt", type=float, default=2.0, help="Maximum dt integrated by command odometry per update; use <=0 to disable the cap.")
    parser.add_argument("--command-odom-target-frame", choices=("body", "world"), default="body", help="Frame of fresh target hints before command-odometry injection/propagation. Current live detector emits body-frame hints.")
    parser.add_argument("--disable-target-propagation", action="store_true", help="With command odometry enabled, use simple target-frame conversion only; do not keep occluded targets fixed in command-odometry world frame.")
    parser.add_argument("--save-lidar-debug", type=Path, default=None, help="Base directory for LiDAR occupancy/debug artifacts. A timestamped run subdir is added by default.")
    parser.add_argument("--save-rgbd-debug", type=Path, default=None, help="Base directory for RGB-D snapshot/YOLO debug artifacts. A timestamped run subdir is added by default.")
    parser.add_argument("--run-id", default=None, help="Optional run id used for timestamped debug subdirectories.")
    parser.add_argument("--no-timestamp-debug-dirs", action="store_true", help="Write debug artifacts directly into the requested directories instead of a run_<timestamp> subdir.")
    parser.add_argument("--json-lines", action="store_true", help="Print one JSON object per step.")
    parser.add_argument("--expected-fsm-id", type=int, default=200, help="Expected Unitree loco fsm_id for motion.")
    parser.add_argument("--allow-non-200", action="store_true", help="Disable fsm_id equality gate; fsm_mode gate remains active.")
    parser.add_argument("--state-samples", type=int, default=3, help="Unitree loco state samples per command.")
    parser.add_argument("--state-interval-s", type=float, default=0.05, help="Delay between state samples.")
    parser.add_argument("--command-vx-scale", type=float, default=1.0, help="Final vx scale/sign applied before Unitree Move().")
    parser.add_argument("--command-vy-scale", type=float, default=1.0, help="Final vy scale/sign applied before Unitree Move().")
    parser.add_argument("--command-wz-scale", type=float, default=1.0, help="Final wz scale/sign applied before Unitree Move(); use -1 if yaw sign is inverted.")
    parser.add_argument("--allow-yaw-on-front-stop", action="store_true", help="Allow yaw-only escape commands when front obstacle stop would otherwise zero all motion.")
    parser.add_argument("--front-stop-yaw-escape-min-wz", type=float, default=0.05, help="Minimum proposed |wz| required to preserve yaw during front-obstacle stop.")
    parser.add_argument("--front-stop-yaw-escape-max-wz", type=float, default=0.60, help="Maximum |wz| allowed during front-obstacle yaw escape.")
    parser.add_argument("--front-stop-yaw-escape-max-linear", type=float, default=0.05, help="Maximum proposed linear speed norm allowed for yaw escape during front-obstacle stop.")
    parser.add_argument("--yes-i-understand-this-moves-the-robot", action="store_true", help="Enable real Unitree Move commands.")
    parser.add_argument("--allow-synthetic-robot-state-motion", action="store_true", help="Required for real motion until real RobotState provider is implemented.")
    args = parser.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.dt < 0.0:
        raise SystemExit("--dt must be non-negative")
    if args.redetect_every < 1:
        raise SystemExit("--redetect-every must be >= 1")
    if args.occluded_redetect_every < 1:
        raise SystemExit("--occluded-redetect-every must be >= 1")
    if args.lost_redetect_every < 1:
        raise SystemExit("--lost-redetect-every must be >= 1")
    if args.occlusion_grace_steps < 0:
        raise SystemExit("--occlusion-grace-steps must be >= 0")
    if not (0.0 <= args.target_smoothing_alpha <= 1.0):
        raise SystemExit("--target-smoothing-alpha must be in [0, 1]")
    if args.target_max_jump_m <= 0.0:
        raise SystemExit("--target-max-jump-m must be > 0")
    if args.realsense_timeout_ms <= 0:
        raise SystemExit("--realsense-timeout-ms must be > 0")
    if args.livox_timeout <= 0.0:
        raise SystemExit("--livox-timeout must be > 0")
    if args.target_nav_replan_every < 0:
        raise SystemExit("--target-nav-replan-every must be >= 0")
    if args.target_nav_inflation < 0.0:
        raise SystemExit("--target-nav-inflation must be >= 0")
    if args.target_nav_max_wz is not None and args.target_nav_max_wz <= 0.0:
        raise SystemExit("--target-nav-max-wz must be positive when provided")
    if args.heading_lookahead_m < 0.0:
        raise SystemExit("--heading-lookahead-m must be >= 0")
    if args.heading_slow_angle_rad < 0.0:
        raise SystemExit("--heading-slow-angle-rad must be >= 0")
    if args.heading_rotate_in_place_angle_rad < args.heading_slow_angle_rad:
        raise SystemExit("--heading-rotate-in-place-angle-rad must be >= --heading-slow-angle-rad")
    if args.heading_rotate_vx_scale < 0.0 or args.heading_rotate_lateral_scale < 0.0:
        raise SystemExit("--heading-rotate-vx-scale and --heading-rotate-lateral-scale must be >= 0")
    if args.state_samples < 1:
        raise SystemExit("--state-samples must be >= 1")
    if args.state_interval_s < 0.0:
        raise SystemExit("--state-interval-s must be non-negative")
    if args.command_odom_max_dt == 0.0:
        raise SystemExit("--command-odom-max-dt must be positive, negative, or omitted; use <0 to disable the cap")
    if args.front_stop_yaw_escape_min_wz < 0.0:
        raise SystemExit("--front-stop-yaw-escape-min-wz must be >= 0")
    if args.front_stop_yaw_escape_max_wz <= 0.0:
        raise SystemExit("--front-stop-yaw-escape-max-wz must be > 0")
    if args.front_stop_yaw_escape_max_linear < 0.0:
        raise SystemExit("--front-stop-yaw-escape-max-linear must be >= 0")

    real_motion = bool(args.yes_i_understand_this_moves_the_robot)
    if real_motion and not args.allow_synthetic_robot_state_motion:
        raise SystemExit(
            "Refusing real motion while RobotState is synthetic. Add "
            "--allow-synthetic-robot-state-motion only for supervised low-speed tests."
        )

    cfg = load_adapter_config(args.config)
    rs_adapter = RealSenseRGBDAdapter(
        RealSenseRGBDAdapterConfig.from_camera_config(
            cfg.camera,
            serial_number=args.realsense_serial,
            use_spatial_filter=not args.no_realsense_spatial_filter,
            use_temporal_filter=not args.no_realsense_temporal_filter,
        )
    )
    livox_adapter = UnitreeVibesLivox2Adapter(
        UnitreeVibesLivox2AdapterConfig(
            livox_config_path=str(args.livox_config),
            host_ip=str(args.host_ip),
            unitree_vibes_dir=str(args.unitree_vibes_dir),
            frame_time_s=float(args.livox_frame_time),
            frame_packets=int(args.livox_frame_packets),
            read_timeout_s=float(args.livox_timeout),
            adapter_config=LivoxAdapterConfig.from_lidar_config(cfg.lidar),
        )
    )

    rs_adapter.start()
    livox_adapter.start()
    wrapped_rs = _TimeoutRealSenseAdapter(rs_adapter, timeout_ms=args.realsense_timeout_ms)
    base_provider = CompositeDryRunObservationProvider(
        config=cfg,
        rgbd_adapter=wrapped_rs,
        lidar_adapter=livox_adapter,
        include_synthetic_lidar_if_missing=False,
        include_synthetic_robot_state=True,
        source_name="DetectFollowLiveObservationProvider",
    )
    target_provider = DetectTargetObservationProvider(
        base_provider=base_provider,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_iou=args.yolo_iou,
        yolo_device=None if args.yolo_device.lower() in {"auto", "none", ""} else args.yolo_device,
        target_class=args.target_class,
        redetect_every_steps=args.redetect_every,
        occluded_redetect_every_steps=args.occluded_redetect_every,
        lost_redetect_every_steps=args.lost_redetect_every,
        occlusion_grace_steps=args.occlusion_grace_steps,
        smoothing_alpha=args.target_smoothing_alpha,
        max_jump_m=args.target_max_jump_m,
        stand_off_m=args.stand_off_m,
        depth_min_m=args.target_depth_min_m,
        depth_max_m=args.target_depth_max_m,
        depth_crop_frac=args.depth_crop_frac,
        depth_percentile=args.depth_percentile,
        use_forward_camera_convention=not args.use_config_camera_transform,
        rgbd_body_exclusion_box=cfg.camera.body_exclusion_box,
    )
    odom_provider = None
    observation_provider = target_provider
    if args.enable_command_odometry:
        max_dt = None if args.command_odom_max_dt < 0.0 else float(args.command_odom_max_dt)
        odom_cls = CommandOdometryObservationProvider if args.disable_target_propagation else CommandOdometryTargetPropagationProvider
        odom_provider = odom_cls(
            base_provider=target_provider,
            estimator=CommandOdometryEstimator(CommandOdometryConfig(max_dt_s=max_dt)),
            target_hint_frame=args.command_odom_target_frame,
        )
        observation_provider = odom_provider

    nav_policy = LidarTargetNavPolicyRuntime(
        LidarTargetNavPolicyRuntimeConfig.from_runtime_config(
            cfg,
            target=Target2D(0.0, 0.0),
            enable_replanning=args.target_nav_enable_replanning,
            replan_every=args.target_nav_replan_every,
            inflation_radius_m=args.target_nav_inflation,
            max_wz=args.target_nav_max_wz,
            command_mode=args.target_nav_command_mode,
            heading_lateral_scale=float(args.heading_lateral_scale),
            heading_min_vx_scale=float(args.heading_min_vx_scale),
            heading_slow_angle_rad=float(args.heading_slow_angle_rad),
            heading_rotate_in_place_angle_rad=float(args.heading_rotate_in_place_angle_rad),
            heading_rotate_vx_scale=float(args.heading_rotate_vx_scale),
            heading_rotate_lateral_scale=float(args.heading_rotate_lateral_scale),
            path_lookahead_m=float(args.heading_lookahead_m),
        )
    )
    policy = StopUnlessTargetPolicy(nav_policy)
    supervisor = SafetySupervisor.from_runtime_config(
        cfg,
        config=SafetySupervisorConfig(
            allow_yaw_on_front_stop=bool(args.allow_yaw_on_front_stop),
            front_stop_yaw_escape_min_wz=float(args.front_stop_yaw_escape_min_wz),
            front_stop_yaw_escape_max_wz=float(args.front_stop_yaw_escape_max_wz),
            front_stop_yaw_escape_max_linear_mps=float(args.front_stop_yaw_escape_max_linear),
        ),
    )
    require_fsm_id = None if args.allow_non_200 else int(args.expected_fsm_id)
    sink = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(
            iface=args.iface,
            dry_run=not real_motion,
            require_arm=True,
            require_fsm_id=require_fsm_id,
            command_vx_scale=args.command_vx_scale,
            command_vy_scale=args.command_vy_scale,
            command_wz_scale=args.command_wz_scale,
            state_samples=args.state_samples,
            state_interval_s=args.state_interval_s,
        ),
        command_limits=cfg.command_limits,
    )
    bridge = RealBridge(
        observation_provider=observation_provider,
        policy_runtime=policy,
        safety_supervisor=supervisor,
        command_sink=sink,
        config=RealBridgeConfig(name="real_g1_follow_detect_live", send_commands=True, auto_disarm_sink_on_stop=True),
    )

    run_id = args.run_id or make_run_id()
    timestamp_debug_dirs = not bool(args.no_timestamp_debug_dirs)
    lidar_debug_dir = timestamped_output_dir(args.save_lidar_debug, run_id=run_id, enabled=timestamp_debug_dirs)
    rgbd_debug_dir = timestamped_output_dir(args.save_rgbd_debug, run_id=run_id, enabled=timestamp_debug_dirs)
    lidar_writer = LidarDebugWriter(lidar_debug_dir, grid_cfg=nav_policy.config.grid_cfg) if lidar_debug_dir else None
    rgbd_writer = RGBDDebugWriter(rgbd_debug_dir) if rgbd_debug_dir else None
    run_info = {
        "run_id": run_id,
        "timestamped_debug_dirs": timestamp_debug_dirs,
        "script": "real_g1_follow_detect_live.py",
        "config": str(args.config),
        "target_class": str(args.target_class),
        "real_motion": real_motion,
        "body_exclusion": {
            "camera": None if cfg.camera.body_exclusion_box is None else cfg.camera.body_exclusion_box.to_dict(),
            "lidar": None if cfg.lidar.body_exclusion_box is None else cfg.lidar.body_exclusion_box.to_dict(),
        },
        "command_odometry": {
            "enabled": bool(args.enable_command_odometry),
            "target_hint_frame": str(args.command_odom_target_frame),
            "target_propagation_enabled": bool(args.enable_command_odometry and not args.disable_target_propagation),
            "max_dt_s": None if args.command_odom_max_dt < 0.0 else float(args.command_odom_max_dt),
        },
        "target_nav_command": {
            "mode": str(args.target_nav_command_mode),
            "heading_lateral_scale": float(args.heading_lateral_scale),
            "heading_min_vx_scale": float(args.heading_min_vx_scale),
            "heading_lookahead_m": float(args.heading_lookahead_m),
            "heading_slow_angle_rad": float(args.heading_slow_angle_rad),
            "heading_rotate_in_place_angle_rad": float(args.heading_rotate_in_place_angle_rad),
            "heading_rotate_vx_scale": float(args.heading_rotate_vx_scale),
            "heading_rotate_lateral_scale": float(args.heading_rotate_lateral_scale),
            "target_nav_max_wz": None if args.target_nav_max_wz is None else float(args.target_nav_max_wz),
        },
        "target_redetection": {
            "redetect_every": int(args.redetect_every),
            "occluded_redetect_every": int(args.occluded_redetect_every),
            "lost_redetect_every": int(args.lost_redetect_every),
            "occlusion_grace_steps": int(args.occlusion_grace_steps),
        },
        "safety_supervisor": {
            "allow_yaw_on_front_stop": bool(args.allow_yaw_on_front_stop),
            "front_stop_yaw_escape_min_wz": float(args.front_stop_yaw_escape_min_wz),
            "front_stop_yaw_escape_max_wz": float(args.front_stop_yaw_escape_max_wz),
            "front_stop_yaw_escape_max_linear": float(args.front_stop_yaw_escape_max_linear),
        },
    }
    write_run_info(lidar_debug_dir, run_info)
    write_run_info(rgbd_debug_dir, run_info)

    print("[INFO]: live detect-and-follow bridge starting")
    print(f"[INFO]: config={args.config}")
    print(f"[INFO]: iface={args.iface}")
    print(f"[INFO]: real_motion={real_motion}")
    print(f"[INFO]: yolo_model={args.yolo_model!r} yolo_device={args.yolo_device!r} redetect_every={args.redetect_every} occluded_redetect_every={args.occluded_redetect_every} lost_redetect_every={args.lost_redetect_every}")
    print(f"[INFO]: fsm gate: require_fsm_id={require_fsm_id} state_samples={args.state_samples} state_interval_s={args.state_interval_s}")
    print(f"[INFO]: command mapping: vx_scale={args.command_vx_scale} vy_scale={args.command_vy_scale} wz_scale={args.command_wz_scale}")
    print(f"[INFO]: target-nav command: mode={args.target_nav_command_mode} heading_lateral_scale={args.heading_lateral_scale} heading_min_vx_scale={args.heading_min_vx_scale} lookahead={args.heading_lookahead_m} slow_angle={args.heading_slow_angle_rad} rotate_angle={args.heading_rotate_in_place_angle_rad} max_wz={args.target_nav_max_wz}")
    print(f"[INFO]: command odometry: enabled={bool(args.enable_command_odometry)} target_frame={args.command_odom_target_frame} target_propagation={bool(args.enable_command_odometry and not args.disable_target_propagation)} max_dt={None if args.command_odom_max_dt < 0.0 else args.command_odom_max_dt}")
    print(f"[INFO]: safety yaw escape: allow={bool(args.allow_yaw_on_front_stop)} min_wz={args.front_stop_yaw_escape_min_wz} max_wz={args.front_stop_yaw_escape_max_wz} max_linear={args.front_stop_yaw_escape_max_linear}")
    print(f"[INFO]: run_id={run_id}")
    if lidar_debug_dir:
        print(f"[INFO]: LiDAR debug artifacts: {lidar_debug_dir}")
    if rgbd_debug_dir:
        print(f"[INFO]: RGB-D debug artifacts: {rgbd_debug_dir}")
    if not real_motion:
        print("[INFO]: dry-run mode: no Unitree Move command will be sent")
    else:
        print("[WARN]: REAL MOTION ENABLED; operator must be ready to stop the robot")
        if args.allow_non_200:
            print("[WARN]: --allow-non-200 enabled: fsm_id equality gate disabled; fsm_mode gate remains active")
        if args.allow_synthetic_robot_state_motion:
            print("[WARN]: RobotState is still synthetic for SafetySupervisor; UnitreeLocoAdapter state gate remains active")

    status = bridge.follow()
    print(f"[INFO]: bridge command: follow -> {status.health.value}: {status.message}")

    exit_code = 0
    try:
        for i in range(args.steps):
            result = bridge.step()
            if odom_provider is not None:
                odom_provider.update_from_bridge_result(result)
            row = _result_to_dict(result)
            tracking = {} if result.observation is None else dict(result.observation.metadata.get("target_tracking", {}))
            command_status = None if result.command_result is None else result.command_result.status
            command_health = None if command_status is None else command_status.health.value
            command_message = None if command_status is None else command_status.message
            command_details = {} if command_status is None else dict(command_status.details)

            if lidar_writer is not None:
                lidar_writer.write_step(i, result, policy=policy)
            if rgbd_writer is not None and result.observation is not None and result.observation.rgbd is not None:
                rgbd_writer.write_frame(
                    step=i,
                    frame=result.observation.rgbd,
                    detections=target_provider.last_detections,
                    source=result.observation.metadata.get("source"),
                    extra_metadata={"target_tracking": tracking},
                )

            payload = {
                **row,
                "step": i,
                "target_tracking": tracking,
                "command_health": command_health,
                "command_message": command_message,
                "command_details": command_details,
                "command_odometry": None if odom_provider is None else odom_provider.snapshot().to_dict(),
            }
            if args.json_lines:
                print(json.dumps(payload, sort_keys=True, default=_json_default))
            else:
                print(
                    f"[INFO]: step={i} mode={row['mode']} action={row['action']} health={row['status_health']} "
                    f"target_state={tracking.get('state')} target={tracking.get('active_target')} "
                    f"proposed={row['proposed_command']} safe={row['safe_command']} accepted={row['command_accepted']} "
                    f"command_health={command_health} command_message={command_message!r} "
                    f"reasons={row['safety_reasons']} warnings={row['safety_warnings']}"
                )
                if command_details:
                    print(f"[INFO]: command_details={json.dumps(command_details, sort_keys=True, default=_json_default)}")

            if result.status.health.value == "fault":
                exit_code = 1
                break
            if result.command_result is not None and not result.command_result.accepted:
                exit_code = 1
                if real_motion:
                    break
            if args.dt > 0.0 and i + 1 < args.steps:
                time.sleep(args.dt)
    finally:
        stop_status = bridge.stop()
        if odom_provider is not None:
            odom_provider.set_stop(source="script_final_stop")
        print(f"[INFO]: bridge command: stop -> {stop_status.health.value}: {stop_status.message}")
        sink.close()
        livox_adapter.stop()
        rs_adapter.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
