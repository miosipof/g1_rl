"""Compatibility shim for the shared detection-stream perception contract.

The implementation lives in :mod:`g1_nav_rl.perception.detection_stream` so it can
be used by Isaac Sim, real-robot, replay, and synthetic producers/consumers.
"""

from g1_nav_rl.perception.detection_stream import (
    DETECTION_STREAM_SCHEMA_VERSION,
    DetectionStreamClient,
    DetectionStreamFrame,
    DetectionStreamHealth,
    DetectionStreamReadResult,
    DetectionStreamStatus,
    TrackDetection,
    write_detection_stream_frame_atomic,
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
]
