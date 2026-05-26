# ORIENTATION.md — Orientation–Velocity Alignment Plan

## Purpose

The current continuous target-following stack became too holonomic for the real G1. It often tried to solve lateral target/path error with a mixture of `vx`, `vy`, and `wz`. In simulation this was acceptable because the proxy can execute clean planar velocity commands. On the real robot, however, lateral motion is weak, gait-coupled, and hard to interpret. The robot can end up looking away from the intended path while strafing sideways, which keeps the obstacle in the frontal safety corridor and prevents clean recovery.

For the current task, the preferred behavior is:

```text
prefer forward motion + yaw
minimize lateral strafe
turn toward the planned direction before translating
```

This document describes the orientation-vs-velocity alignment approach and how it relates to the simplified controller direction in `SIMPLIFY.md`.

---

## Observed problem

In the real tests, A* frequently produced a plausible path around the obstacle, but the robot did not reliably align its body orientation with the path tangent or next useful waypoint. Several failure modes appeared:

```text
1. robot tried to move with vx + vy while keeping yaw almost unchanged
2. front obstacle safety kept slowing/stopping the robot because the robot still faced the obstacle
3. when yaw authority was increased, yaw smoothing/rate limiting sometimes emitted wrong-sign yaw
4. command odometry overestimated executed yaw and destabilized subsequent replanning
5. the robot sometimes leaked forward motion while a turn-in-place was intended
```

The main conclusion is that the path can be geometrically valid while the velocity command is dynamically inappropriate for the real G1.

---

## Current command conventions

The project convention is:

```text
+x = robot forward
+y = robot left
+wz = counter-clockwise / left yaw
```

Real calibration showed:

```text
+vx moves forward
+vy moves robot-left
+wz turns counter-clockwise
```

Therefore, the yaw sign itself is not globally inverted. Problems observed after yaw tuning were caused by control logic, rate limiting, odometry, and target jumps, not by a fixed `wz` sign error.

---

## Previous holonomic mapping

The older controller approximately used:

```text
vx = k_xy * waypoint_x_body
vy = k_xy * waypoint_y_body
wz = k_yaw * path_tangent_yaw_error
```

This means lateral waypoint error directly becomes `vy`.

For example:

```text
waypoint is forward-left
→ command forward + strafe-left
→ yaw may remain small if local path tangent is approximately forward
```

This is too holonomic for the real G1.

---

## Heading-aligned mapping

The heading-aligned controller changes the command interpretation. It computes a desired motion bearing from the selected waypoint or lookahead point:

```text
bearing = atan2(waypoint_y_body, waypoint_x_body)
```

Then it prefers yaw alignment:

```text
wz = k_heading * bearing
vx = forward speed scaled down when bearing is large
vy = small lateral correction only
```

Desired behavior:

```text
small bearing:
  move mostly forward, with small yaw/lateral correction

medium bearing:
  reduce vx, reduce vy, increase yaw importance

large bearing:
  turn in place, no translation
```

This makes the robot align its body orientation with the planned corridor before moving into it.

---

## Turn-first behavior

For the real robot, heading alignment should include turn-first thresholds:

```text
if abs(bearing) > rotate_in_place_angle:
    vx = 0
    vy = 0
    wz = sign(bearing) * yaw_command

elif abs(bearing) > slow_angle:
    vx = reduced
    vy = reduced
    wz = normal heading correction

else:
    vx = normal
    vy = small
    wz = normal heading correction
```

Reasonable starting values:

```text
slow_angle:             0.35 rad  (~20 deg)
rotate_in_place_angle:  0.80 rad  (~46 deg)
heading_lookahead_m:    0.60 m
heading_lateral_scale:  0.02–0.05
max_wz:                 0.4–0.6 rad/s effective SDK command
```

The key invariant is:

```text
if turn_in_place is active:
    final command must have vx = 0 and vy = 0
```

No safety smoother or acceleration limiter should reintroduce linear velocity during turn-in-place.

---

## Lookahead point

The immediate A* waypoint can be too close to the robot and can cause unstable bearings. A lookahead point should be selected farther along the path:

```text
heading_lookahead_m = 0.5–0.8 m
```

Instead of steering toward the next grid cell, steer toward the first path point at least `heading_lookahead_m` along the remaining path. This smooths local grid jaggedness and makes the yaw target more stable.

---

## Safety interaction

Safety should not block yaw-in-place just because there is an obstacle in front. It should distinguish:

```text
forward motion into obstacle → unsafe
turn-in-place near obstacle  → allowed escape behavior
```

Recommended last-resort rule:

```text
if front_obstacle_stop and proposed command is yaw-only:
    allow wz
    force vx = 0
    force vy = 0
else if front_obstacle_stop and proposed command has forward/lateral motion:
    stop
```

This prevents the robot from being trapped facing an obstacle when A* wants it to rotate toward a side corridor.

---

## Rate limiting and smoothing

The yaw rate/acceleration limiter must be sign-safe.

Invalid behavior seen in logs:

```text
proposed_wz > 0, safe_wz < 0
proposed_wz < 0, safe_wz > 0
```

This causes the robot to start turns in the wrong direction.

Correct rule:

```text
if proposed_wz and previous_wz have opposite signs:
    safe_wz should go to 0 first
    then ramp in the requested direction
```

It must never output old-sign yaw when the controller requests a turn in the opposite direction.

For the simplified stepwise controller, avoid continuous smoothing entirely unless needed as a final hardware guard.

---

## Command odometry caution

Command-integrated odometry was added as a stopgap, but it can damage behavior when the robot does not execute commands exactly.

Observed issue:

```text
controller thinks robot turned by integrated command
real robot turned less or differently
target propagation and planner frame become wrong
planner changes direction
robot oscillates or starts wrong turn
```

For the simplified version, disable command odometry by default.

If odometry is used at all, it should integrate the final mapped SDK command, not the pre-mapping `safe_command`:

```text
odom_vx = sdk_vx
odom_vy = sdk_vy
odom_wz = sdk_wz
```

and likely apply conservative scaling to yaw until real robot-state/SLAM odometry is available.

---

## Simplified version recommendation

For the next simplified real-robot script, avoid continuous heading-aligned velocity commands and convert the path into atomic actions:

```text
Turn(angle)
Forward(distance)
StopObserve()
```

This enforces orientation/velocity alignment structurally:

```text
turn first
then move forward
then stop and observe again
```

No `vy` is needed for the core task.

A typical action sequence:

```text
Stop
Observe RGB-D + LiDAR
Plan A* path
Turn toward first path segment
Forward 0.3–0.5 m
Stop
Observe again
```

This is easier to debug and better matched to the current fixed-target/fixed-obstacle experiment.

---

## Acceptance criteria

The orientation-aligned behavior is acceptable when logs show:

```text
1. for side path:
   waypoint/lookahead bearing has correct sign
   wz has same sign
   vx/vy are near zero if bearing is large

2. during turn-in-place:
   final SDK command has vx=0, vy=0, wz!=0

3. near obstacle:
   front stop blocks forward motion but allows yaw escape

4. no wrong-sign yaw:
   proposed_wz and final/sdk_wz do not have opposite signs

5. no lateral crabbing:
   vy remains negligible unless explicitly enabled for small corrections
```

For the simplified stepwise version, the stronger criterion is:

```text
the robot should rotate toward the intended path segment before each forward move
```
