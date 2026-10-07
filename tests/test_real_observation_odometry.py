from __future__ import annotations

from types import SimpleNamespace

from adapters import RobotObservation, Target2D, UnitreeOdometryObservationProvider
from adapters.real_unitree_odometry import UnitreeOdometryAdapter


class _Base:
    def read_observation(self) -> RobotObservation:
        return RobotObservation(timestamp_s=10.0, target_hint=Target2D(2.0, -0.2))


def test_odometry_wrapper_uses_atomic_pose_and_advances_parent_timestamp() -> None:
    odom = UnitreeOdometryAdapter(clock=lambda: 10.1)
    odom._on_odom(
        SimpleNamespace(
            position=[1.0, 2.0, 0.72],
            velocity=[0.0, 0.0, 0.0],
            yaw_speed=0.0,
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.3]),
        )
    )
    odom.capture_origin()
    obs = UnitreeOdometryObservationProvider(_Base(), odom).read_observation()
    assert obs.timestamp_s == 10.1
    assert obs.pose == obs.robot_state.pose
    assert obs.target_hint.xy == (2.0, -0.2)
    assert obs.metadata["unitree_odometry"]["sequence"] == 1
    assert obs.statuses[-1].name == "unitree_odometry"
