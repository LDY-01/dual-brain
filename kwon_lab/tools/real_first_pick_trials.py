#!/usr/bin/env python3
"""Record ten operator-supervised real picks; this tool never commands motors."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.camera_roles import capture_camera_frame, load_camera_registry
from hardware.first_pick_trials import FAILURE_STAGES, summarize_first_pick_trials
from hardware.real_preflight import evaluate_real_preflight


def _capture_roles(registry, folder, label):
    saved = {}
    for role in ("wrist", "overhead"):
        entry = registry["roles"][role]
        frame, diagnostics = capture_camera_frame(
            entry["index"],
            backend=entry.get("backend", "dshow"),
            width=entry.get("width", 1280),
            height=entry.get("height", 720),
        )
        if frame is None or not diagnostics["read_ok"]:
            raise RuntimeError(f"Could not capture {role} frame: {diagnostics}")
        path = folder / f"{label}_{role}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not save frame: {path}")
        saved[role] = str(path.resolve())
    return saved


def _ask_choice(prompt, choices):
    allowed = {choice.casefold(): choice for choice in choices}
    while True:
        answer = input(f"{prompt} [{'/'.join(choices)}]: ").strip().casefold()
        if answer in allowed:
            return allowed[answer]
        print("Invalid choice; try again.")


def _save_session(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_self_test():
    trials = [
        {"success": index < 8, "lift_verified": index < 8,
         "failure_stage": None if index < 8 else "missed_grasp"}
        for index in range(10)
    ]
    passed = summarize_first_pick_trials(trials)
    failed = summarize_first_pick_trials(trials[:-1])
    report = {
        "eight_of_ten_passes": passed["passed"],
        "nine_records_is_incomplete": not failed["complete"] and not failed["passed"],
        "failure_count_preserved": passed["failure_counts"].get("missed_grasp") == 2,
    }
    report["passed"] = all(report.values())
    print(json.dumps(report, indent=2))
    return report["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-config", type=Path, default=Path("config/real_camera_roles.local.json"))
    parser.add_argument("--safety-config", type=Path, default=Path("config/real_robot_safety_limits.local.json"))
    parser.add_argument("--calibration-config", type=Path, default=Path("config/overhead_camera_calibration.local.json"))
    parser.add_argument("--workspace-config", type=Path, default=Path("config/real_workspace.local.json"))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--pass-successes", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path("results/real_first_pick"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(0 if run_self_test() else 1)
    if args.trials < 1 or not 1 <= args.pass_successes <= args.trials:
        raise SystemExit("Require trials >= 1 and 1 <= pass-successes <= trials")

    preflight = evaluate_real_preflight(
        args.camera_config,
        args.safety_config,
        args.calibration_config,
        args.workspace_config,
        probe_cameras=True,
    )
    if not preflight["motion_authorized"]:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        raise SystemExit("Preflight blocked the trial session; no motor command was issued.")
    confirmation = input(
        "This is a RECORD-ONLY tool and never moves the robot. Confirm that each pick will use "
        "a separately approved low-speed controller. Type START_RECORDING: "
    )
    if confirmation != "START_RECORDING":
        raise SystemExit("Trial recording cancelled.")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_root / timestamp
    session_dir.mkdir(parents=True, exist_ok=False)
    registry = load_camera_registry(args.camera_config)
    session = {
        "format_version": 1,
        "session_id": timestamp,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "layout_id": preflight["layout_id"],
        "tool_controls_motors": False,
        "expected_trials": args.trials,
        "pass_successes": args.pass_successes,
        "preflight": preflight,
        "trials": [],
        "summary": None,
    }
    session_file = session_dir / "session.json"
    _save_session(session_file, session)

    for trial_number in range(1, args.trials + 1):
        print(f"\nTrial {trial_number}/{args.trials}")
        pose = _ask_choice("Place the block, then record its pose", ("UPRIGHT", "TIPPED"))
        input("Keep hands clear. Press ENTER to capture BEFORE frames: ")
        before = _capture_roles(registry, session_dir, f"trial_{trial_number:02d}_before")
        input(
            "Run exactly one approved LOW-SPEED FIRST-PICK attempt with emergency stop ready. "
            "Press ENTER after the attempt: "
        )
        after = _capture_roles(registry, session_dir, f"trial_{trial_number:02d}_after")
        success = _ask_choice("Was the block securely picked and held?", ("YES", "NO")) == "YES"
        lift_verified = success or (
            _ask_choice("Was any lift observed before failure?", ("YES", "NO")) == "YES"
        )
        failure_stage = None if success else _ask_choice("Failure stage", FAILURE_STAGES)
        notes = input("Optional short note: ").strip()
        session["trials"].append({
            "trial": trial_number,
            "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "block_pose": pose,
            "success": success,
            "lift_verified": lift_verified,
            "failure_stage": failure_stage,
            "notes": notes,
            "frames": {"before": before, "after": after},
        })
        session["summary"] = summarize_first_pick_trials(
            session["trials"], args.trials, args.pass_successes
        )
        _save_session(session_file, session)

    session["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    session["summary"] = summarize_first_pick_trials(
        session["trials"], args.trials, args.pass_successes
    )
    _save_session(session_file, session)
    print(json.dumps(session["summary"], indent=2, ensure_ascii=False))
    print(f"Saved session: {session_file}")


if __name__ == "__main__":
    main()
