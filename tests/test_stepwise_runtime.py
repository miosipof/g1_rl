import torch

from adapters import LidarFrame, RobotObservation, StepwiseTargetNavPolicyRuntime, StepwiseTargetNavPolicyRuntimeConfig, Target2D
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.stepwise import StepwiseNavConfig
from g1_nav_rl.planning.grid_astar import GridAStarCfg


def _runtime(**kwargs):
    stepwise_cfg = StepwiseNavConfig(
        grid_cfg=OccupancyGridCfg(x_min=-1.0, x_max=4.0, y_min=-2.0, y_max=2.0, resolution=0.1),
        astar_cfg=GridAStarCfg(inflation_radius=0.0),
        forward_segment_m=0.35,
        forward_speed_mps=0.35,
        turn_speed_radps=0.50,
        replan_every_actions=0,
        blind_plan_actions=1,
        emergency_front_stop_m=0.35,
        success_radius_m=0.20,
    )
    return StepwiseTargetNavPolicyRuntime(StepwiseTargetNavPolicyRuntimeConfig(stepwise_cfg=stepwise_cfg, **kwargs))


def _obs(target=True, points=None):
    if points is None:
        points = torch.empty((0, 3), dtype=torch.float32)
    lidar = LidarFrame(timestamp_s=1.0, points_body_xyz=points)
    return RobotObservation(
        timestamp_s=1.0,
        lidar=lidar,
        target_hint=Target2D(1.0, 0.0) if target else None,
        metadata={"test": True},
    )


def test_stepwise_runtime_converts_decision_to_command2d():
    rt = _runtime()
    cmd = rt.step(_obs())

    assert cmd.source == "stepwise_target_nav_policy"
    assert cmd.metadata["policy"] == "stepwise-target-nav"
    assert cmd.duration_s is not None
    assert cmd.vx > 0.0
    assert cmd.vy == 0.0
    assert rt.last_decision is not None
    assert rt.last_decision.execute


def test_stepwise_runtime_blocks_forward_when_front_obstacle_is_close():
    rt = _runtime()
    points = torch.tensor([[0.20, 0.0, 0.3]], dtype=torch.float32)
    cmd = rt.step(_obs(points=points))

    assert cmd.is_stop
    assert rt.last_decision is not None
    assert rt.last_decision.blocked
    assert rt.last_decision.reason == "forward_blocked"


def test_stepwise_runtime_missing_target_returns_stop():
    rt = _runtime()
    cmd = rt.step(_obs(target=False))

    assert cmd.is_stop
    assert rt.last_decision is not None
    assert rt.last_decision.reason == "target_missing_stop_observe"


def test_stepwise_runtime_uses_occupancy_grid_override_for_planning():
    rt = _runtime(require_lidar=False)
    grid_cfg = rt.config.stepwise_cfg.grid_cfg
    grid = torch.zeros((grid_cfg.height, grid_cfg.width), dtype=torch.float32)
    # Insert a full-width barrier between start=(0,0) and target=(1,0).
    # If the runtime ignored the override and rebuilt from the empty LiDAR frame,
    # it would move forward through free space.
    barrier_row = int((grid_cfg.x_max - 0.5) / grid_cfg.resolution)
    grid[barrier_row, :] = 1.0

    obs = _obs(points=torch.empty((0, 3), dtype=torch.float32))
    obs.metadata["stepwise_occupancy_grid"] = grid
    cmd = rt.step(obs)

    assert cmd.is_stop
    assert rt.last_decision is not None
    assert rt.last_decision.reason in {"plan_failed", "empty_action_queue"}
