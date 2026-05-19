"""Optional YOLO detector adapter for Phase-5 perception targets.

The core navigation/perception geometry code must not depend on YOLO.  This
module keeps the optional ``ultralytics`` import local and converts YOLO results
into the shared ``Detection2D`` dataclass used by ``target_from_bbox``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from g1_nav_rl.perception.target_from_bbox import Detection2D


@dataclass(frozen=True)
class YoloDetectorConfig:
    """Minimal YOLO runtime options."""

    model: str = "yolov8n.pt"
    confidence: float = 0.25
    iou: float = 0.45
    device: str | None = None
    verbose: bool = False


class YoloDetector:
    """Thin wrapper around ultralytics.YOLO.

    This class intentionally exposes only the small detector API needed by our
    target-provider pipeline: RGB image in, list[Detection2D] out.
    """

    def __init__(self, config: YoloDetectorConfig | None = None) -> None:
        self.config = config or YoloDetectorConfig()
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "ultralytics is required for YoloDetector. Install it separately "
                "or run the smoke test with --skip_yolo / --fallback_synthetic_bbox."
            ) from exc
        self._model = YOLO(self.config.model)

    def detect(self, rgb_image: Any) -> list[Detection2D]:
        """Run YOLO on an RGB image and return Detection2D records.

        Parameters
        ----------
        rgb_image:
            HxWx3 RGB uint8/float image. If float in [0,1], it is converted to
            uint8. The input is not assumed to be BGR.
        """
        rgb = _as_rgb_uint8(rgb_image)
        kwargs: dict[str, Any] = {
            "conf": float(self.config.confidence),
            "iou": float(self.config.iou),
            "verbose": bool(self.config.verbose),
        }
        if self.config.device:
            kwargs["device"] = self.config.device
        results = self._model.predict(source=rgb, **kwargs)
        detections: list[Detection2D] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
            conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
            cls = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
            for bbox, score, class_id in zip(xyxy, conf, cls):
                class_idx = int(class_id)
                class_name = str(names.get(class_idx, class_idx))
                detections.append(
                    Detection2D(
                        class_name=class_name,
                        bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=float(score),
                    )
                )
        return detections


def filter_detections(
    detections: Iterable[Detection2D],
    *,
    allowed_classes: set[str] | None = None,
    min_confidence: float = 0.0,
) -> list[Detection2D]:
    """Return detections matching class/confidence constraints."""
    allowed_norm = {c.lower() for c in allowed_classes} if allowed_classes is not None else None
    out: list[Detection2D] = []
    for det in detections:
        if allowed_norm is not None and det.class_name.lower() not in allowed_norm:
            continue
        if float(det.confidence) < float(min_confidence):
            continue
        out.append(det)
    return out


def _as_rgb_uint8(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got shape={arr.shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr_f = arr.astype(np.float32, copy=False)
    if np.nanmax(arr_f) <= 1.0:
        arr_f = arr_f * 255.0
    arr_u8 = np.nan_to_num(arr_f, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr_u8)
