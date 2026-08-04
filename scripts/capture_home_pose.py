#!/usr/bin/env python3
"""Save the SO101 follower's currently verified task-start pose as home."""

import argparse
import json
from datetime import datetime
from pathlib import Path

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
DEFAULT_OUTPUT = Path("config/lens_cap_home_pose.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/tty.usbmodem5B140307781")
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    robot = SO101Follower(
        SO101FollowerConfig(port=args.port, id=args.robot_id, cameras={}, max_relative_target=5.0)
    )
    robot.connect(calibrate=False)
    try:
        observation = robot.get_observation()
        pose = {joint: round(float(observation[f"{joint}.pos"]), 2) for joint in JOINTS}
    finally:
        robot.disconnect()

    print(json.dumps(pose, indent=2))
    if not args.save:
        print("Read only. These numbers are the encoder reading of the pose you physically set.")
        return

    confirmation = input("Confirm that the arm is visibly at the collision-free task-start pose. Type SAVE: ")
    if confirmation != "SAVE":
        raise SystemExit("Home pose was not saved.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    contents = {
        "task": "Place the lens cap into the open box.",
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pose": pose,
    }
    args.output.write_text(json.dumps(contents, indent=2) + "\n")
    print(f"Saved verified home pose to {args.output}")


if __name__ == "__main__":
    main()
