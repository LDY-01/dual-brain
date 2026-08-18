#!/usr/bin/env python3
"""Return the SO101 follower to a physically verified task-start home pose."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

# Allow both `python scripts/return_home.py` and `python -m scripts.return_home`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kwon_lab.hardware.joint_safety import load_safety_limits, safe_send_action


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

DEFAULT_HOME_FILE = Path("config/lens_cap_home_pose.json")
DEFAULT_SAFETY_FILE = Path("config/real_robot_safety_limits.local.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/tty.usbmodem5B140307781")
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--home-file", type=Path, default=DEFAULT_HOME_FILE)
    parser.add_argument("--safety-file", type=Path, default=DEFAULT_SAFETY_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run == args.execute:
        raise SystemExit("Choose exactly one of --dry-run or --execute.")
    if args.duration < 3 or args.hz <= 0:
        raise SystemExit("Use --duration >= 3 and a positive --hz.")
    if not args.home_file.is_file():
        raise SystemExit(
            f"Home pose file not found: {args.home_file}. "
            "First create it with scripts/capture_home_pose.py --save."
        )
    home_data = json.loads(args.home_file.read_text())
    home_pose = home_data["pose"]
    missing = set(JOINTS) - set(home_pose)
    if missing:
        raise SystemExit(f"Home pose file is missing joints: {sorted(missing)}")

    # Fail before connecting to hardware if the pillar-side absolute limit has
    # not been physically measured. This applies to dry-run as well so the
    # exact guard used for execution is always visible and testable.
    try:
        safety_limits = load_safety_limits(args.safety_file, require_configured=True)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"SAFETY BLOCK: {exc}") from exc

    config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras={},
        # Keeps every individual home command safely bounded as well.
        max_relative_target=5.0,
    )
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    try:
        observation = robot.get_observation()
        start = np.array([observation[f"{joint}.pos"] for joint in JOINTS], dtype=float)
        target = np.array([home_pose[joint] for joint in JOINTS], dtype=float)
        print("Current:", dict(zip(JOINTS, np.round(start, 2), strict=True)))
        print("Home:   ", dict(zip(JOINTS, np.round(target, 2), strict=True)))

        if args.dry_run:
            return

        steps = max(1, round(args.duration * args.hz))
        print(f"Returning home over {args.duration:.1f}s. Watch the arm; cut 12V power for any abnormal motion.")
        for step in range(1, steps + 1):
            alpha = step / steps
            pose = start + alpha * (target - start)
            _, interventions = safe_send_action(
                robot,
                {f"{joint}.pos": float(value) for joint, value in zip(JOINTS, pose, strict=True)},
                safety_limits,
            )
            for item in interventions:
                print(
                    "SAFETY CLAMP: shoulder_pan "
                    f"{item['requested_deg']:.2f} -> {item['applied_deg']:.2f} deg"
                )
            time.sleep(1 / args.hz)
        print("Home return complete.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
