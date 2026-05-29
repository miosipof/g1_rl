from __future__ import annotations

import numpy as np

from adapters.g1_policy_runtime import RLResidualTargetNavPolicyRuntime, RLResidualTargetNavPolicyRuntimeConfig
from adapters.types import LidarFrame, RobotObservation
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.simple_nav import Pose2D, Target2D
from g1_nav_rl.planning.grid_astar import GridAStarCfg
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicyCfg


def test_rl_residual_runtime_runs_with_zero_policy() -> None:
    grid_cfg = OccupancyGridCfg(x_min=-1.0, x_max=3.0, y_min=-2.0, y_max=2.0, resolution=0.2)
    cfg = RLResidualTargetNavPolicyRuntimeConfig(
        target=Target2D(2.0, 0.0),
        grid_cfg=grid_cfg,
        astar_cfg=GridAStarCfg(inflation_radius=0.2),
        residual_policy_cfg=RLResidualPolicyCfg(path_points=4, grid_cfg=grid_cfg),
        require_lidar=False,
    )
    rt = RLResidualTargetNavPolicyRuntime(cfg)
    obs = RobotObservation(
        timestamp_s=1.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        lidar=LidarFrame(timestamp_s=1.0, points_body_xyz=np.zeros((0, 3), dtype=np.float32)),
        metadata={"target_visible": True, "target_age_s": 0.0, "tracker_state": "tracking"},
    )
    cmd = rt.step(obs)
    assert cmd.metadata["policy"] == "rl-residual-target-nav"
    assert "nominal_command" in cmd.metadata
    assert "residual" in cmd.metadata
