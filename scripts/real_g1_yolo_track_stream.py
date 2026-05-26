#!/usr/bin/env python3
"""Publish RealSense + YOLO/track detections as DetectionStreamFrame JSON.

This producer is the first implementation of the asynchronous detection-stream
architecture.  It owns RGB-D capture and detection/tracking, writes one atomic
``latest_tracks.json`` file per step, and optionally saves RGB-D/debug overlays.
The follow controller should consume the JSON stream instead of running YOLO in
its control loop.

The first backend uses Ultralytics on CPU/GPU.  On Jetson, this script can later
be extended to load a TensorRT engine while preserving the same output schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from adapters import (
    RGBDDebugWriter,
    RealSenseRGBDAdapter,
    RealSenseRGBDAdapterConfig,
    load_adapter_config,
    make_run_id,
    rgbd_debug_frame_stats,
    timestamped_output_dir,
    write_run_info,
)
from g1_nav_rl.perception.detection_stream import DetectionStreamFrame, write_detection_stream_frame_atomic
from g1_nav_rl.perception.detection_stream_producer import (
    DetectionStreamProjectionConfig,
    make_detection_stream_frame_from_detections,
)
from g1_nav_rl.perception.target_from_bbox import Detection2D


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


def _parse_class_filter(raw: str) -> set[str] | None:
    raw = str(raw).strip()
    if not raw or raw.lower() in {"any", "all", "*"}:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _load_ultralytics_model(model_path: str) -> Any:  # pragma: no cover - optional dependency
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ultralytics is required for real_g1_yolo_track_stream.py. "
            "Install the real extra or run this script in the detection environment."
        ) from exc
    return YOLO(model_path)


def _extract_detections_from_ultralytics(
    results: Any,
    *,
    allowed_classes: set[str] | None,
    min_confidence: float,
) -> tuple[list[Detection2D], list[int | str | None], list[str]]:
    """Convert Ultralytics result objects into Detection2D + track IDs."""

    detections: list[Detection2D] = []
    track_ids: list[int | str | None] = []
    statuses: list[str] = []
    allowed_norm = None if allowed_classes is None else {str(c).lower() for c in allowed_classes}

    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
        conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
        cls = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
        ids_raw = getattr(boxes, "id", None)
        if ids_raw is None:
            ids = [None] * len(xyxy)
        else:
            ids_np = ids_raw.detach().cpu().numpy() if hasattr(ids_raw, "detach") else np.asarray(ids_raw)
            ids = [None if v is None or not np.isfinite(v) else int(v) for v in ids_np]
        for i, (bbox, score, class_id) in enumerate(zip(xyxy, conf, cls)):
            confidence = float(score)
            if confidence < float(min_confidence):
                continue
            class_idx = int(class_id)
            class_name = str(names.get(class_idx, class_idx))
            if allowed_norm is not None and class_name.lower() not in allowed_norm:
                continue
            detections.append(
                Detection2D(
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
            )
            track_ids.append(ids[i] if i < len(ids) else None)
            statuses.append("tracked" if (i < len(ids) and ids[i] is not None) else "detected")
    return detections, track_ids, statuses


def _run_ultralytics(
    model: Any,
    rgb: np.ndarray,
    *,
    mode: str,
    confidence: float,
    iou: float,
    device: str | None,
    tracker_config: str | None,
) -> Any:  # pragma: no cover - optional dependency
    kwargs: dict[str, Any] = {
        "source": rgb,
        "conf": float(confidence),
        "iou": float(iou),
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    if mode == "track":
        kwargs["persist"] = True
        if tracker_config:
            kwargs["tracker"] = tracker_config
        return model.track(**kwargs)
    return model.predict(**kwargs)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--output", type=Path, default=Path("outputs/runtime/latest_tracks.json"), help="Atomic latest-track JSON path.")
    parser.add_argument("--steps", type=int, default=0, help="Number of frames to publish. 0 means run until interrupted.")
    parser.add_argument("--dt", type=float, default=0.0, help="Optional sleep between frames; 0 uses camera/inference cadence.")
    parser.add_argument("--timeout-ms", type=int, default=1000, help="RealSense wait_for_frames timeout.")
    parser.add_argument("--no-spatial-filter", action="store_true", help="Disable RealSense spatial depth filter.")
    parser.add_argument("--no-temporal-filter", action="store_true", help="Disable RealSense temporal depth filter.")
    parser.add_argument("--use-config-camera-transform", action="store_true", help="Use camera.T_body_camera from YAML instead of explicit forward optical convention.")

    parser.add_argument("--backend", choices=["ultralytics"], default="ultralytics", help="Detection backend. TensorRT backend will be added later.")
    parser.add_argument("--mode", choices=["track", "predict"], default="track", help="Use YOLO tracking or one-frame detection.")
    parser.add_argument("--yolo-model", default=None, help="YOLO model/engine path. Defaults to config perception model.")
    parser.add_argument("--yolo-device", default=None, help="YOLO device, e.g. cpu, cuda:0, 0. Defaults to config perception device.")
    parser.add_argument("--yolo-confidence", type=float, default=None, help="YOLO confidence threshold. Defaults to config perception threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--tracker-config", default=None, help="Optional Ultralytics tracker yaml, e.g. bytetrack.yaml or botsort.yaml.")
    parser.add_argument("--target-class", default=None, help="Comma-separated classes to publish. Defaults to config target_class. Use 'any' for all.")

    parser.add_argument("--stand-off-m", type=float, default=None, help="Published target standoff. Defaults to config perception stand_off_m.")
    parser.add_argument("--depth-min-m", type=float, default=0.20, help="Minimum valid bbox depth.")
    parser.add_argument("--depth-max-m", type=float, default=8.0, help="Maximum valid bbox depth.")
    parser.add_argument("--depth-crop-frac", type=float, default=0.50, help="Central bbox crop fraction for depth.")
    parser.add_argument("--depth-percentile", type=float, default=0.50, help="Depth percentile in bbox crop.")
    parser.add_argument("--min-bbox-area-px", type=float, default=0.0, help="Reject boxes smaller than this area.")
    parser.add_argument("--min-depth-valid-ratio", type=float, default=0.0, help="Reject boxes with too few valid depth pixels.")

    parser.add_argument("--save-rgbd-debug", type=str, default="", help="Optional RGB-D/debug overlay output directory.")
    parser.add_argument("--save-every", type=int, default=1, help="Save RGB-D debug every N frames when enabled.")
    parser.add_argument("--run-id", type=str, default="", help="Optional run id for debug folder naming.")
    parser.add_argument("--no-timestamp-debug-dirs", action="store_true", help="Write directly into debug dir without run subfolder.")
    parser.add_argument("--json-lines", action="store_true", help="Print per-frame JSON status lines.")
    args = parser.parse_args()

    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    if args.dt < 0.0:
        raise SystemExit("--dt must be >= 0")
    if args.timeout_ms <= 0:
        raise SystemExit("--timeout-ms must be > 0")
    if args.save_every <= 0:
        raise SystemExit("--save-every must be > 0")

    cfg = load_adapter_config(args.config)
    model_path = str(args.yolo_model or cfg.perception.yolo_model)
    yolo_device_raw = args.yolo_device if args.yolo_device is not None else cfg.perception.yolo_device
    yolo_device = None if str(yolo_device_raw).lower() in {"", "auto", "none"} else str(yolo_device_raw)
    confidence = float(args.yolo_confidence if args.yolo_confidence is not None else cfg.perception.yolo_confidence_threshold)
    target_class_raw = args.target_class if args.target_class is not None else cfg.perception.target_class
    allowed_classes = _parse_class_filter(str(target_class_raw))
    stand_off = float(args.stand_off_m if args.stand_off_m is not None else cfg.perception.stand_off_m)

    projection_cfg = DetectionStreamProjectionConfig(
        target_class=None if allowed_classes is None or len(allowed_classes) != 1 else next(iter(allowed_classes)),
        min_confidence=confidence,
        min_bbox_area_px=float(args.min_bbox_area_px),
        min_depth_valid_ratio=float(args.min_depth_valid_ratio),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        depth_crop_frac=float(args.depth_crop_frac),
        depth_percentile=float(args.depth_percentile),
        stand_off_m=stand_off,
        use_forward_camera_convention=not bool(args.use_config_camera_transform),
        metadata={"producer_script": "real_g1_yolo_track_stream.py"},
    )

    run_id = args.run_id or make_run_id()
    debug_dir = timestamped_output_dir(args.save_rgbd_debug, run_id=run_id, enabled=not args.no_timestamp_debug_dirs)
    debug_writer = RGBDDebugWriter(debug_dir) if debug_dir else None
    if debug_dir:
        write_run_info(
            debug_dir,
            {
                "run_id": run_id,
                "script": "real_g1_yolo_track_stream.py",
                "output": str(args.output),
                "mode": str(args.mode),
                "model": model_path,
                "device": yolo_device,
                "target_class": target_class_raw,
                "allowed_classes": None if allowed_classes is None else sorted(allowed_classes),
                "projection_config": projection_cfg.__dict__,
            },
        )

    rs_cfg = RealSenseRGBDAdapterConfig.from_camera_config(
        cfg.camera,
        serial_number=args.serial,
        use_spatial_filter=not args.no_spatial_filter,
        use_temporal_filter=not args.no_temporal_filter,
    )
    rs_adapter = RealSenseRGBDAdapter(rs_cfg)
    model = _load_ultralytics_model(model_path)

    print("[INFO]: detection stream producer starting")
    print(f"[INFO]: output={args.output}")
    print(f"[INFO]: config={args.config}")
    print(f"[INFO]: model={model_path!r} device={yolo_device!r} mode={args.mode}")
    print(f"[INFO]: classes={target_class_raw!r} confidence={confidence:.3f} stand_off_m={stand_off:.3f}")
    if debug_dir:
        print(f"[INFO]: RGB-D debug artifacts: {debug_dir}")

    exit_code = 0
    frame_index = 0
    try:
        rs_adapter.start()
        while args.steps == 0 or frame_index < int(args.steps):
            loop_start = time.monotonic()
            rgbd = rs_adapter.read_frame(timeout_ms=args.timeout_ms)
            yolo_start = time.monotonic()
            yolo_results = _run_ultralytics(
                model,
                rgbd.rgb,
                mode=str(args.mode),
                confidence=confidence,
                iou=float(args.yolo_iou),
                device=yolo_device,
                tracker_config=args.tracker_config,
            )
            yolo_elapsed = time.monotonic() - yolo_start
            raw_detections, track_ids, statuses = _extract_detections_from_ultralytics(
                yolo_results,
                allowed_classes=allowed_classes,
                min_confidence=confidence,
            )
            stream_frame = make_detection_stream_frame_from_detections(
                frame=rgbd,
                detections=raw_detections,
                config=projection_cfg,
                frame_id=frame_index,
                source="real_g1_yolo_track_stream",
                track_ids=track_ids,
                statuses=statuses,
                body_exclusion_box=cfg.camera.body_exclusion_box,
                metadata={
                    "backend": str(args.backend),
                    "mode": str(args.mode),
                    "model": model_path,
                    "device": yolo_device,
                    "yolo_elapsed_s": float(yolo_elapsed),
                },
            )
            write_detection_stream_frame_atomic(stream_frame, args.output, indent=2)

            if debug_writer is not None and frame_index % int(args.save_every) == 0:
                debug_writer.write_frame(
                    step=frame_index,
                    frame=rgbd,
                    detections=stream_frame.tracks,
                    source="real_g1_yolo_track_stream",
                    extra_metadata={
                        "stream_output": str(args.output),
                        "stream_frame": stream_frame.to_dict(),
                        "raw_detection_count": len(raw_detections),
                        "published_track_count": len(stream_frame.tracks),
                    },
                )

            payload = {
                "step": frame_index,
                "timestamp_s": float(stream_frame.timestamp_s),
                "raw_detections": len(raw_detections),
                "published_tracks": len(stream_frame.tracks),
                "track_ids": [track.track_id for track in stream_frame.tracks],
                "target_body_xy": [track.target_body_xy for track in stream_frame.tracks],
                "yolo_elapsed_s": float(yolo_elapsed),
                "loop_elapsed_s": float(time.monotonic() - loop_start),
                "output": str(args.output),
            }
            if args.json_lines:
                print(json.dumps(payload, sort_keys=True, default=_json_default), flush=True)
            else:
                print(
                    f"[INFO]: step={frame_index} raw={len(raw_detections)} tracks={len(stream_frame.tracks)} "
                    f"ids={[track.track_id for track in stream_frame.tracks]} "
                    f"targets={[track.target_body_xy for track in stream_frame.tracks]} "
                    f"yolo={yolo_elapsed:.3f}s loop={payload['loop_elapsed_s']:.3f}s",
                    flush=True,
                )

            frame_index += 1
            if args.dt > 0.0:
                sleep_s = max(0.0, float(args.dt) - (time.monotonic() - loop_start))
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("[INFO]: interrupted; writing final empty stream frame", flush=True)
        empty = DetectionStreamFrame(
            timestamp_s=time.monotonic(),
            frame_id="interrupt_stop",
            source="real_g1_yolo_track_stream",
            tracks=(),
            metadata={"reason": "keyboard_interrupt"},
        )
        write_detection_stream_frame_atomic(empty, args.output, indent=2)
    finally:
        rs_adapter.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
