"""Perception-to-navigation target adapters."""


from .detection_stream import (
    DETECTION_STREAM_SCHEMA_VERSION,
    DetectionStreamClient,
    DetectionStreamFrame,
    DetectionStreamHealth,
    DetectionStreamReadResult,
    DetectionStreamStatus,
    TrackDetection,
    write_detection_stream_frame_atomic,
)

from .target_from_bbox import (
    CameraIntrinsics,
    DepthEstimate,
    Detection2D,
    TargetFromBBoxConfig,
    TargetFromBBoxResult,
    bbox_depth_to_target2d,
    center_crop_bbox_xyxy,
    clamp_bbox_xyxy,
    estimate_depth_in_bbox,
    make_forward_camera_T_body_camera,
    pose2d_to_T_world_body,
    select_detection_for_target,
    target_from_detection,
    target_xy_with_standoff,
    transform_point,
    unproject_pixel_to_camera,
)

__all__ = [
    "DETECTION_STREAM_SCHEMA_VERSION",
    "DetectionStreamClient",
    "DetectionStreamFrame",
    "DetectionStreamHealth",
    "DetectionStreamReadResult",
    "DetectionStreamStatus",
    "TrackDetection",
    "write_detection_stream_frame_atomic",
    "CameraIntrinsics",
    "DepthEstimate",
    "Detection2D",
    "TargetFromBBoxConfig",
    "TargetFromBBoxResult",
    "bbox_depth_to_target2d",
    "center_crop_bbox_xyxy",
    "clamp_bbox_xyxy",
    "estimate_depth_in_bbox",
    "make_forward_camera_T_body_camera",
    "pose2d_to_T_world_body",
    "select_detection_for_target",
    "target_from_detection",
    "target_xy_with_standoff",
    "transform_point",
    "unproject_pixel_to_camera",
]

try:
    from .yolo_detector import YoloDetector, YoloDetectorConfig, filter_detections
except Exception:  # optional ultralytics dependency may be unavailable
    YoloDetector = None  # type: ignore
    YoloDetectorConfig = None  # type: ignore
    filter_detections = None  # type: ignore

__all__ += ["YoloDetector", "YoloDetectorConfig", "filter_detections"]
