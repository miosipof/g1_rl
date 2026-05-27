import math

import numpy as np
import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.stepwise import (
    StepwiseActionKind,
    StepwiseNavConfig,
    StepwiseNavigator,
    forward_action_blocked,
    front_clearance_from_points,
    path_to_stepwise_actions,
    plan_stepwise_actions,
    remaining_path_blocked,
)
from g1_nav_rl.planning.grid_astar import GridAStarCfg, plan_grid_path_xy, xy_to_grid_cell


def _cfg(**kwargs):
    values = dict(
        grid_cfg=OccupancyGridCfg(x_min=-1.0, x_max=4.0, y_min=-2.0, y_max=2.0, resolution=0.1),
        astar_cfg=GridAStarCfg(inflation_radius=0.0),
        forward_segment_m=0.35,
        forward_speed_mps=0.35,
        turn_speed_radps=0.50,
        turn_angle_max_rad=math.radians(35),
        min_turn_angle_rad=math.radians(3),
        min_forward_m=0.02,
        action_duration_max_s=0.1,
        replan_every_actions=0,
        replan_every_forward_m=0.0,
        target_change_replan_m=0.50,
        path_block_check=True,
        path_block_lookahead_m=2.0,
        blind_plan_actions=0,
        emergency_front_stop_m=0.35,
        success_radius_m=0.20,
        allow_start_relocation=True,
    )
    values.update(kwargs)
    return StepwiseNavConfig(**values)


def test_path_to_stepwise_actions_builds_full_yaw_then_translate_queue():
    cfg = _cfg()
    actions = path_to_stepwise_actions([(0.0, 0.0), (0.0, 0.1), (0.0, 0.2)], cfg=cfg)

    assert len(actions) == 3
    assert actions[0].kind == StepwiseActionKind.TURN
    assert math.isclose(actions[0].angle_rad, math.pi / 2.0, abs_tol=1e-9)
    assert math.isclose(actions[0].duration_s, (math.pi / 2.0) / cfg.turn_speed_radps, abs_tol=1e-9)
    assert actions[1].kind == StepwiseActionKind.FORWARD
    assert actions[2].kind == StepwiseActionKind.FORWARD
    assert all(a.duration_s > 0.0 for a in actions)


def test_path_to_stepwise_actions_keeps_straight_path_as_forward_commands():
    cfg = _cfg()
    actions = path_to_stepwise_actions([(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)], cfg=cfg)

    assert [a.kind for a in actions] == [StepwiseActionKind.FORWARD, StepwiseActionKind.FORWARD]
    assert math.isclose(actions[0].distance_m, 0.1, abs_tol=1e-9)
    assert math.isclose(actions[0].duration_s, 0.1 / cfg.forward_speed_mps, abs_tol=1e-9)
    assert actions[0].command.vy == 0.0


def test_turn_duration_is_not_clipped_by_legacy_action_duration_cap():
    cfg = _cfg(action_duration_max_s=0.1, turn_speed_radps=0.5)
    actions = path_to_stepwise_actions([(0.0, 0.0), (0.0, 1.0)], cfg=cfg)

    turn = actions[0]
    assert turn.kind == StepwiseActionKind.TURN
    assert turn.duration_s > cfg.action_duration_max_s
    assert math.isclose(turn.duration_s, abs(turn.angle_rad) / cfg.turn_speed_radps, abs_tol=1e-9)


def test_plan_stepwise_actions_builds_command_queue_from_astar_path():
    cfg = _cfg()
    grid = torch.zeros((cfg.grid_cfg.height, cfg.grid_cfg.width), dtype=torch.float32)

    result = plan_stepwise_actions(grid, (1.0, 0.0), cfg=cfg)

    assert result.found
    assert len(result.actions) > 1
    assert all(a.kind in {StepwiseActionKind.FORWARD, StepwiseActionKind.TURN} for a in result.actions)


def test_front_guard_blocks_forward_but_not_turn():
    cfg = _cfg(emergency_front_stop_m=0.40)
    turn = path_to_stepwise_actions([(0.0, 0.0), (0.0, 1.0)], cfg=cfg)[0]
    forward = path_to_stepwise_actions([(0.0, 0.0), (1.0, 0.0)], cfg=cfg)[0]

    assert not forward_action_blocked(turn, front_min_distance_m=0.20, cfg=cfg)
    assert forward_action_blocked(forward, front_min_distance_m=0.20, cfg=cfg)
    assert not forward_action_blocked(forward, front_min_distance_m=0.60, cfg=cfg)


def test_front_clearance_from_points_uses_front_corridor_only():
    points = np.array(
        [
            [0.50, 0.10, 0.30],
            [0.20, 1.00, 0.30],
            [-0.10, 0.00, 0.30],
            [0.35, -0.20, 0.30],
        ],
        dtype=np.float32,
    )

    assert math.isclose(front_clearance_from_points(points, y_abs_max_m=0.30), 0.35, abs_tol=1e-6)


def test_navigator_executes_queue_without_replanning_every_cycle():
    cfg = _cfg(replan_every_actions=99, success_radius_m=0.05)
    nav = StepwiseNavigator(cfg)
    grid = torch.zeros((cfg.grid_cfg.height, cfg.grid_cfg.width), dtype=torch.float32)

    first = nav.decide(target_xy_b=(1.0, 0.0), occupancy_grid=grid, front_min_distance_m=1.0)
    second = nav.decide(target_xy_b=(0.9, 0.0), occupancy_grid=grid, front_min_distance_m=1.0)

    assert first.execute
    assert second.execute
    assert first.replan_attempted
    assert first.replan_reason == "initial_sync"
    assert not second.replan_attempted
    assert second.queue_len_after_pop < first.queue_len_after_pop


def test_navigator_missing_target_stops_immediately():
    cfg = _cfg()
    nav = StepwiseNavigator(cfg)
    grid = torch.zeros((cfg.grid_cfg.height, cfg.grid_cfg.width), dtype=torch.float32)

    first = nav.decide(target_xy_b=(1.0, 0.0), occupancy_grid=grid, front_min_distance_m=1.0)
    assert first.execute

    missing = nav.decide(target_xy_b=None, occupancy_grid=grid, front_min_distance_m=1.0)
    assert not missing.execute
    assert missing.target_visible is False
    assert missing.action.kind == StepwiseActionKind.STOP_OBSERVE
    assert missing.reason == "target_missing_stop_observe"


def test_navigator_front_block_stops_and_marks_replan_pending():
    cfg = _cfg(emergency_front_stop_m=0.40)
    nav = StepwiseNavigator(cfg)
    grid = torch.zeros((cfg.grid_cfg.height, cfg.grid_cfg.width), dtype=torch.float32)

    decision = nav.decide(target_xy_b=(1.0, 0.0), occupancy_grid=grid, front_min_distance_m=0.20)

    assert not decision.execute
    assert decision.blocked
    assert decision.reason == "forward_blocked"
    assert decision.replan_reason == "forward_blocked"
    assert decision.action.kind == StepwiseActionKind.STOP_OBSERVE


def test_start_relocation_remains_enabled_by_default_for_legacy_behavior():
    cfg = _cfg(astar_cfg=GridAStarCfg(inflation_radius=0.0))
    grid = torch.zeros((cfg.grid_cfg.height, cfg.grid_cfg.width), dtype=torch.float32)
    start_cell = xy_to_grid_cell((0.0, 0.0), cfg.grid_cfg)
    grid[start_cell[0], start_cell[1]] = 1.0

    result = plan_stepwise_actions(grid, (1.0, 0.0), cfg=cfg)

    assert result.found
    assert result.astar_result.start_cell != start_cell


def test_grid_astar_can_disable_start_relocation_for_strict_callers():
    cfg = _cfg(astar_cfg=GridAStarCfg(inflation_radius=0.0))
    grid = torch.zeros((cfg.grid_cfg.height, cfg.grid_cfg.width), dtype=torch.float32)
    start_cell = xy_to_grid_cell((0.0, 0.0), cfg.grid_cfg)
    grid[start_cell[0], start_cell[1]] = 1.0

    result = plan_grid_path_xy(
        grid,
        (0.0, 0.0),
        (1.0, 0.0),
        cfg.grid_cfg,
        cfg.astar_cfg,
        allow_goal_relocation=True,
        allow_start_relocation=False,
    )

    assert not result.found
    assert result.reason == "start_occupied"


def test_remaining_path_blocked_uses_inflated_current_grid():
    cfg = _cfg(astar_cfg=GridAStarCfg(inflation_radius=0.0), path_block_lookahead_m=2.0)
    grid = torch.zeros((cfg.grid_cfg.height, cfg.grid_cfg.width), dtype=torch.float32)
    plan = plan_stepwise_actions(grid, (1.0, 0.0), cfg=cfg)
    assert plan.found

    blocked_grid = torch.zeros_like(grid)
    cell = xy_to_grid_cell((0.5, 0.0), cfg.grid_cfg)
    blocked_grid[cell[0], cell[1]] = 1.0

    blocked, info = remaining_path_blocked(blocked_grid, plan, cfg=cfg, current_action_index=0)
    assert blocked
    assert info["reason"] == "inflated_occupancy_intersection"


def test_periodic_replan_stops_before_syncing_new_plan():
    cfg = _cfg(replan_every_actions=1, target_change_replan_m=0.0, path_block_check=False, success_radius_m=0.05)
    nav = StepwiseNavigator(cfg)
    grid = torch.zeros((cfg.grid_cfg.height, cfg.grid_cfg.width), dtype=torch.float32)

    first = nav.decide(target_xy_b=(1.0, 0.0), occupancy_grid=grid, front_min_distance_m=1.0)
    second = nav.decide(target_xy_b=(0.9, 0.0), occupancy_grid=grid, front_min_distance_m=1.0)
    third = nav.decide(target_xy_b=(0.9, 0.0), occupancy_grid=grid, front_min_distance_m=1.0)

    assert first.execute
    assert not second.execute
    assert second.reason == "periodic_replan_actions"
    assert third.execute
    assert third.replan_attempted
    assert third.replan_reason == "periodic_replan_actions"
