"""RGB-D debug artifact writer for real/sim dry-run bridge validation.

The writer saves raw RGB snapshots, metric depth arrays, depth visualizations,
YOLO overlays, and metadata from the shared ``RGBDFrame`` contract.  The module
is dependency-light at import time: Pillow and ultralytics/YOLO are imported only
when image saving or detection is explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .types import RGBDFrame


BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class RGBDDebugDetection:
    """Serializable detection record used by RGB-D debug overlays."""

    class_name: str
    bbox_xyxy: BBoxXYXY
    confidence: float = 1.0

    @classmethod
    def from_any(cls, det: Any) -> "RGBDDebugDetection":
        """Convert Detection2D-like objects or mappings into a debug record."""
        if isinstance(det, RGBDDebugDetection):
            return det
        if isinstance(det, dict):
            bbox = det.get("bbox_xyxy") or det.get("bbox")
            name = det.get("class_name", det.get("label", "object"))
            conf = det.get("confidence", det.get("conf", 1.0))
        else:
            bbox = getattr(det, "bbox_xyxy")
            name = getattr(det, "class_name", "object")
            conf = getattr(det, "confidence", 1.0)
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"detection bbox must have four elements, got {bbox!r}")
        return cls(
            class_name=str(name),
            bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            confidence=float(conf),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
        }


@dataclass(frozen=True)
class RGBDDebugArtifacts:
    """Paths written for one RGB-D debug step."""

    output_dir: str
    step: int
    paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"output_dir": self.output_dir, "step": self.step, "paths": dict(self.paths)}


def rgbd_frame_stats(frame: RGBDFrame) -> dict[str, Any]:
    """Return compact RGB/depth/intrinsics statistics for logs and JSON."""
    rgb = np.asarray(frame.rgb)
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    return {
        "timestamp_s": float(frame.timestamp_s),
        "frame_id": frame.frame_id,
        "depth_frame_id": frame.depth_frame_id,
        "rgb_shape": [int(v) for v in rgb.shape],
        "rgb_dtype": str(rgb.dtype),
        "rgb_min": int(rgb.min()) if rgb.size else None,
        "rgb_max": int(rgb.max()) if rgb.size else None,
        "rgb_mean": float(rgb.mean()) if rgb.size else None,
        "rgb_channel_mean": [float(v) for v in rgb.reshape(-1, 3).mean(axis=0)] if rgb.size else [],
        "depth_shape": [int(v) for v in depth.shape],
        "depth_dtype": str(depth.dtype),
        "depth_valid_count": int(finite.size),
        "depth_min_m": None if finite.size == 0 else float(finite.min()),
        "depth_max_m": None if finite.size == 0 else float(finite.max()),
        "depth_mean_m": None if finite.size == 0 else float(finite.mean()),
        "intrinsics": {
            "fx": float(frame.intrinsics.fx),
            "fy": float(frame.intrinsics.fy),
            "cx": float(frame.intrinsics.cx),
            "cy": float(frame.intrinsics.cy),
            "width": int(frame.intrinsics.width),
            "height": int(frame.intrinsics.height),
        },
        "metadata": dict(frame.metadata),
    }


def depth_to_uint8_visualization(
    depth_m: np.ndarray,
    *,
    min_m: float | None = None,
    max_m: float | None = None,
    percentile_min: float = 2.0,
    percentile_max: float = 98.0,
) -> np.ndarray:
    """Convert metric depth into an 8-bit grayscale image for inspection."""
    depth = np.asarray(depth_m, dtype=np.float32)
    finite_mask = np.isfinite(depth) & (depth > 0.0)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if not finite_mask.any():
        return out
    values = depth[finite_mask]
    lo = float(np.percentile(values, percentile_min)) if min_m is None else float(min_m)
    hi = float(np.percentile(values, percentile_max)) if max_m is None else float(max_m)
    if hi <= lo:
        hi = lo + 1e-6
    scaled = (depth - lo) / (hi - lo)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    out[finite_mask] = np.clip(scaled[finite_mask] * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def save_rgb_image(path: Path, rgb: np.ndarray) -> None:
    """Save an RGB uint8 image using Pillow imported lazily."""
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8 RGB image, got shape={arr.shape} dtype={arr.dtype}")
    Image.fromarray(arr, mode="RGB").save(path)


def save_depth_visualization(path: Path, depth_m: np.ndarray) -> None:
    """Save an 8-bit depth visualization using Pillow imported lazily."""
    from PIL import Image

    vis = depth_to_uint8_visualization(depth_m)
    Image.fromarray(vis, mode="L").save(path)


def draw_detections_overlay(rgb: np.ndarray, detections: Iterable[Any]) -> np.ndarray:
    """Return an RGB image with detection boxes and labels drawn on top."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for det_any in detections:
        det = RGBDDebugDetection.from_any(det_any)
        x1, y1, x2, y2 = det.bbox_xyxy
        label = f"{det.class_name} {det.confidence:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        text_xy = (x1, max(0.0, y1 - 14.0))
        draw.text(text_xy, label, fill=(255, 0, 0))
    return np.asarray(image, dtype=np.uint8)


def save_detections_overlay(path: Path, rgb: np.ndarray, detections: Iterable[Any]) -> None:
    """Save RGB overlay with detection boxes."""
    save_rgb_image(path, draw_detections_overlay(rgb, detections))


def run_yolo_detections(
    rgb: np.ndarray,
    *,
    model: str = "yolov8n.pt",
    confidence: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    allowed_classes: Sequence[str] | None = ("person",),
) -> list[RGBDDebugDetection]:
    """Run optional YOLO detection and return serializable debug detections."""
    from g1_nav_rl.perception.yolo_detector import YoloDetector, YoloDetectorConfig, filter_detections

    detector = YoloDetector(YoloDetectorConfig(model=model, confidence=confidence, iou=iou, device=device, verbose=False))
    detections = detector.detect(rgb)
    if allowed_classes is not None:
        detections = filter_detections(detections, allowed_classes={str(c) for c in allowed_classes}, min_confidence=confidence)
    return [RGBDDebugDetection.from_any(det) for det in detections]


class LazyYoloDebugDetector:
    """Lazy YOLO detector wrapper used by scripts when debug detection is enabled."""

    def __init__(
        self,
        *,
        model: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
        allowed_classes: Sequence[str] | None = ("person",),
    ) -> None:
        self.model = model
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device
        self.allowed_classes = None if allowed_classes is None else tuple(str(c) for c in allowed_classes)
        self._detector: Any | None = None
        self._filter_detections: Any | None = None

    def detect(self, rgb: np.ndarray) -> list[RGBDDebugDetection]:
        if self._detector is None:
            from g1_nav_rl.perception.yolo_detector import YoloDetector, YoloDetectorConfig, filter_detections

            self._detector = YoloDetector(
                YoloDetectorConfig(
                    model=self.model,
                    confidence=self.confidence,
                    iou=self.iou,
                    device=self.device,
                    verbose=False,
                )
            )
            self._filter_detections = filter_detections
        detections = self._detector.detect(rgb)
        if self.allowed_classes is not None:
            detections = self._filter_detections(
                detections,
                allowed_classes={str(c) for c in self.allowed_classes},
                min_confidence=self.confidence,
            )
        return [RGBDDebugDetection.from_any(det) for det in detections]


class RGBDDebugWriter:
    """Write RGB-D snapshots, depth visualization, detections, and metadata."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_frame(
        self,
        *,
        step: int,
        frame: RGBDFrame,
        detections: Iterable[Any] | None = None,
        source: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RGBDDebugArtifacts:
        detections_list = [RGBDDebugDetection.from_any(det) for det in (detections or [])]
        prefix = f"rgbd_step_{int(step):06d}"
        paths: dict[str, str] = {}

        rgb_step = self.output_dir / f"{prefix}_rgb.png"
        rgb_latest = self.output_dir / "rgb_latest.png"
        depth_npy_step = self.output_dir / f"{prefix}_depth_m.npy"
        depth_npy_latest = self.output_dir / "depth_m_latest.npy"
        depth_vis_step = self.output_dir / f"{prefix}_depth_viz.png"
        depth_vis_latest = self.output_dir / "depth_viz_latest.png"
        overlay_step = self.output_dir / f"{prefix}_yolo_overlay.png"
        overlay_latest = self.output_dir / "yolo_overlay_latest.png"
        det_step = self.output_dir / f"{prefix}_detections.json"
        det_latest = self.output_dir / "detections_latest.json"
        meta_step = self.output_dir / f"{prefix}_meta.json"
        meta_latest = self.output_dir / "rgbd_meta_latest.json"

        save_rgb_image(rgb_step, frame.rgb)
        save_rgb_image(rgb_latest, frame.rgb)
        np.save(depth_npy_step, frame.depth_m)
        np.save(depth_npy_latest, frame.depth_m)
        save_depth_visualization(depth_vis_step, frame.depth_m)
        save_depth_visualization(depth_vis_latest, frame.depth_m)
        save_detections_overlay(overlay_step, frame.rgb, detections_list)
        save_detections_overlay(overlay_latest, frame.rgb, detections_list)

        detections_payload = [det.to_dict() for det in detections_list]
        meta = rgbd_frame_stats(frame)
        meta.update(
            {
                "step": int(step),
                "source": source,
                "detection_count": len(detections_payload),
                "detections": detections_payload,
            }
        )
        if extra_metadata:
            meta["extra"] = dict(extra_metadata)

        det_payload = {
            "step": int(step),
            "source": source,
            "detections": detections_payload,
            "detection_count": len(detections_payload),
        }
        det_step.write_text(json.dumps(det_payload, indent=2, sort_keys=True), encoding="utf-8")
        det_latest.write_text(json.dumps(det_payload, indent=2, sort_keys=True), encoding="utf-8")
        meta_step.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        meta_latest.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        for name, path in {
            "rgb": rgb_step,
            "rgb_latest": rgb_latest,
            "depth_m": depth_npy_step,
            "depth_m_latest": depth_npy_latest,
            "depth_viz": depth_vis_step,
            "depth_viz_latest": depth_vis_latest,
            "yolo_overlay": overlay_step,
            "yolo_overlay_latest": overlay_latest,
            "detections": det_step,
            "detections_latest": det_latest,
            "metadata": meta_step,
            "metadata_latest": meta_latest,
        }.items():
            paths[name] = str(path)
        return RGBDDebugArtifacts(output_dir=str(self.output_dir), step=int(step), paths=paths)

    def write_step(
        self,
        step: int,
        result: Any,
        *,
        detector: Any | None = None,
    ) -> dict[str, Any] | None:
        """Write debug artifacts from a BridgeStepResult-like object if RGB-D exists."""
        obs = getattr(result, "observation", None)
        if obs is None or getattr(obs, "rgbd", None) is None:
            return None
        frame = obs.rgbd
        detections: list[Any] = []
        detector_error: str | None = None
        if detector is not None:
            try:
                detections = list(detector.detect(frame.rgb))
            except Exception as exc:  # pragma: no cover - optional YOLO/hardware path
                detector_error = f"{type(exc).__name__}: {exc}"
        artifacts = self.write_frame(
            step=int(step),
            frame=frame,
            detections=detections,
            source=getattr(obs, "metadata", {}).get("source"),
            extra_metadata={"detector_error": detector_error} if detector_error else None,
        )
        payload = artifacts.to_dict()
        if detector_error:
            payload["detector_error"] = detector_error
        return payload


__all__ = [
    "LazyYoloDebugDetector",
    "RGBDDebugArtifacts",
    "RGBDDebugDetection",
    "RGBDDebugWriter",
    "depth_to_uint8_visualization",
    "draw_detections_overlay",
    "rgbd_frame_stats",
    "run_yolo_detections",
    "save_depth_visualization",
    "save_detections_overlay",
    "save_rgb_image",
]
