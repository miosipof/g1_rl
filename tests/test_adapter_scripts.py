from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_g1_follow_dryrun  # noqa: E402
import real_g1_follow_low_speed  # noqa: E402
import real_g1_probe_adapters  # noqa: E402
import real_g1_probe_livox  # noqa: E402


def test_probe_adapters_json_runs(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["real_g1_probe_adapters.py", "--json"])
    assert real_g1_probe_adapters.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "real_g1_conservative"
    assert payload["observation"]["has_lidar"] is True
    assert payload["safety_decision"]["safe_command"][0] > 0.0


def test_follow_dryrun_json_lines_runs_two_steps(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["real_g1_follow_dryrun.py", "--steps", "2", "--dt", "0", "--json-lines"])
    assert real_g1_follow_dryrun.main() == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(row["mode"] == "following" for row in rows)
    assert all(row["command_accepted"] is True for row in rows)


def test_follow_dryrun_front_obstacle_forces_stop(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["real_g1_follow_dryrun.py", "--steps", "1", "--dt", "0", "--json-lines", "--obstacle-front-m", "0.30"],
    )
    assert real_g1_follow_dryrun.main() == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert row["safe_command"] == [0.0, 0.0, 0.0]
    assert any("front_obstacle_stop" in reason for reason in row["safety_reasons"])


def test_low_speed_defaults_to_dry_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["real_g1_follow_low_speed.py", "--steps", "1", "--dt", "0"])
    assert real_g1_follow_low_speed.main() == 0
    stdout = capsys.readouterr().out
    assert "dry-run mode" in stdout
    assert "REAL MOTION ENABLED" not in stdout


def test_low_speed_refuses_real_motion_with_synthetic_observation_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["real_g1_follow_low_speed.py", "--steps", "1", "--dt", "0", "--yes-i-understand-this-moves-the-robot"],
    )
    try:
        real_g1_follow_low_speed.main()
    except SystemExit as exc:
        assert exc.code != 0
        assert "Refusing real motion with synthetic observations" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected SystemExit refusal")


def test_probe_livox_synthetic_json_runs(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["real_g1_probe_livox.py", "--json"])
    assert real_g1_probe_livox.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["source"] == "synthetic"
    assert payload["frame"]["point_count"] > 0


def test_follow_dryrun_livox_npy_runs(monkeypatch, capsys, tmp_path) -> None:
    import numpy as np

    path = tmp_path / "livox.npy"
    np.save(path, np.array([[[1.0, 0.0, 0.4], [2.0, 0.2, 0.4]]], dtype=np.float32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "real_g1_follow_dryrun.py",
            "--observation-source",
            "livox-npy",
            "--livox-npy-path",
            str(path),
            "--steps",
            "1",
            "--dt",
            "0",
            "--json-lines",
        ],
    )
    assert real_g1_follow_dryrun.main() == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert row["has_lidar"] is True
    assert row["lidar_synthetic"] is False
    assert row["has_robot_state"] is True
    assert row["command_accepted"] is True


def test_follow_dryrun_livox_npy_single_snapshot_loops_by_default(monkeypatch, capsys, tmp_path) -> None:
    import numpy as np

    path = tmp_path / "single_frame_livox.npy"
    np.save(path, np.array([[1.0, 0.0, 0.4], [2.0, 0.2, 0.4]], dtype=np.float32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "real_g1_follow_dryrun.py",
            "--observation-source",
            "livox-npy",
            "--livox-npy-path",
            str(path),
            "--steps",
            "3",
            "--dt",
            "0",
            "--json-lines",
        ],
    )
    assert real_g1_follow_dryrun.main() == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(rows) == 3
    assert all(row["has_lidar"] is True for row in rows)
    assert all(row["status_health"] == "ok" for row in rows)


def test_follow_dryrun_livox_npy_can_disable_loop(monkeypatch, capsys, tmp_path) -> None:
    import numpy as np

    path = tmp_path / "single_frame_livox.npy"
    np.save(path, np.array([[1.0, 0.0, 0.4], [2.0, 0.2, 0.4]], dtype=np.float32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "real_g1_follow_dryrun.py",
            "--observation-source",
            "livox-npy",
            "--livox-npy-path",
            str(path),
            "--no-loop-livox-npy",
            "--steps",
            "2",
            "--dt",
            "0",
            "--json-lines",
        ],
    )
    assert real_g1_follow_dryrun.main() == 1
    rows = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert rows[0]["status_health"] == "ok"
    assert rows[1]["status_health"] == "fault"
