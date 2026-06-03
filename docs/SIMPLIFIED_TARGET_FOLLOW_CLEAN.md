# Clean simplified target follower

` scripts/simplified_target_follow.py ` is now the public real-robot entrypoint.
The previous flag-heavy implementation is kept as:

```text
scripts/simplified_target_follow_advanced.py
```

Use the advanced script only for exact replay/debugging. For robot sessions, use
the clean script.

## Fixed policy

```text
YOLO stream best current target
+ Livox occupancy grid
→ A* in robot body frame
→ yaw-first / forward-only Unitree commands
→ short burst
→ stop / observe / validate / replan if LiDAR says the active path is blocked
```

Target loss alone is not a replan trigger. The script never replans to stale
body-frame target coordinates.

## Minimal dry-run

```bash
python scripts/simplified_target_follow.py \
  --unitree-vibes-dir ~/unitree_g1_vibes \
  --livox-config ~/unitree_g1_vibes/mid360_config.json \
  --host-ip 192.168.123.222 \
  --iface eth0 \
  --detection-stream-path "${STREAM_PATH}" \
  --target-class "${TARGET_CLASS}" \
  --max-cycles 5 \
  --json-lines
```

## Conservative real motion

```bash
python scripts/simplified_target_follow.py \
  --unitree-vibes-dir ~/unitree_g1_vibes \
  --livox-config ~/unitree_g1_vibes/mid360_config.json \
  --host-ip 192.168.123.222 \
  --iface eth0 \
  --detection-stream-path "${STREAM_PATH}" \
  --target-class "${TARGET_CLASS}" \
  --max-cycles 20 \
  --burst-s 2.0 \
  --sdk-forward-vx-mps 0.20 \
  --sdk-turn-wz-radps 0.25 \
  --planner-inflation-m 0.25 \
  --front-replan-m 0.75 \
  --front-stop-m 0.45 \
  --allow-non-200 \
  --save-lidar-debug "${BASE_DIR}/simplified_clean/lidar" \
  --run-id "simplified_clean_${RUN_ID}" \
  --no-timestamp-debug-dirs \
  --json-lines \
  --yes-i-understand-this-moves-the-robot
```

## Public knobs

| Argument | Meaning |
|---|---|
| `--burst-s` | Maximum open-loop motion duration before stop/observe/check. |
| `--sdk-forward-vx-mps` | Actual forward velocity sent to Unitree SDK. No hidden vx scale. |
| `--sdk-turn-wz-radps` | Actual yaw rate sent to Unitree SDK. |
| `--planner-inflation-m` | A* obstacle inflation. |
| `--front-replan-m` | Raw front obstacle distance that asks for a LiDAR-driven replan. Must be larger than `--front-stop-m`. |
| `--front-stop-m` | Emergency raw-front stop. Motion is suppressed below this distance. |
| `--goal-relocation-radius-m` | Radius for goal-only relocation if the target stand-off cell is occupied. |

Everything else is intentionally fixed inside the script.
