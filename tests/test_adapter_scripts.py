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
import diagnose_frames  # noqa: E402
import real_g1_calibrate_commands  # noqa: E402


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
        ["real_g1_follow_dryrun.py", "--steps", "1", "--dt", "0", "--json-lines", "--obstacle-front-m", "0.01"],
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


def test_low_speed_help_exposes_non_200_gate(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["real_g1_follow_low_speed.py", "--help"])
    try:
        real_g1_follow_low_speed.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover - argparse help exits
        raise AssertionError("expected argparse help SystemExit")
    stdout = capsys.readouterr().out
    assert "--allow-non-200" in stdout
    assert "--expected-fsm-id" in stdout
    assert "--command-wz-scale" in stdout


def test_detect_follow_live_help_exposes_motion_gates(monkeypatch, capsys) -> None:
    import real_g1_follow_detect_live  # noqa: WPS433

    monkeypatch.setattr(sys, "argv", ["real_g1_follow_detect_live.py", "--help"])
    try:
        real_g1_follow_detect_live.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover - argparse help exits
        raise AssertionError("expected argparse help SystemExit")
    stdout = capsys.readouterr().out
    assert "--yes-i-understand-this-moves-the-robot" in stdout
    assert "--allow-synthetic-robot-state-motion" in stdout
    assert "--allow-non-200" in stdout
    assert "--redetect-every" in stdout
    assert "--command-wz-scale" in stdout


def test_diagnose_frames_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["diagnose_frames.py", "--help"])
    try:
        diagnose_frames.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover - argparse help exits
        raise AssertionError("expected argparse help SystemExit")
    stdout = capsys.readouterr().out
    assert "--target-class" in stdout
    assert "--lidar-source" in stdout
    assert "--use-config-camera-transform" in stdout


def test_real_g1_calibrate_commands_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["real_g1_calibrate_commands.py", "--help"])
    try:
        real_g1_calibrate_commands.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover - argparse help exits
        raise AssertionError("expected argparse help SystemExit")
    stdout = capsys.readouterr().out
    assert "--yaw-degrees" in stdout
    assert "--allow-outside-config-limits" in stdout
    assert "--yes-i-understand-this-moves-the-robot" in stdout
