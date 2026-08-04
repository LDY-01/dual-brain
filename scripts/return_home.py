#!/usr/bin/env python3
"""Return the SO101 follower to a physically verified task-start home pose."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

DEFAULT_HOME_FILE = Path("config/lens_cap_home_pose.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/tty.usbmodem5B140307781")
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--home-file", type=Path, default=DEFAULT_HOME_FILE)
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
            robot.send_action({f"{joint}.pos": float(value) for joint, value in zip(JOINTS, pose, strict=True)})
            time.sleep(1 / args.hz)
        print("Home return complete.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
