#!/usr/bin/env python3
"""Read a manually positioned SO101 and save a pillar-side pan limit.

This tool never commands motion. It disables motor torque, asks the operator to
place the arm at a visibly clear boundary, and only then reads the encoder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig


DEFAULT_OUTPUT = Path("config/real_robot_safety_limits.local.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Physical follower port, for example COM3")
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument(
        "--layout-id",
        default="UNSET",
        help="Stable name for this physical robot/pillar layout.",
    )
    parser.add_argument(
        "--blocked-direction",
        default="auto",
        choices=("auto", "increasing", "decreasing"),
        help="Whether moving toward the pillar increases or decreases shoulder_pan degrees",
    )
    parser.add_argument("--margin-deg", type=float, default=10.0)
    parser.add_argument("--clearance-cm", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def build_payload(args: argparse.Namespace, measured_deg: float, blocked_direction: str) -> dict:
    if blocked_direction == "increasing":
        minimum, maximum = None, measured_deg - args.margin_deg
    else:
        minimum, maximum = measured_deg + args.margin_deg, None
    return {
        "format_version": 1,
        "units": "degrees",
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "layout_id": args.layout_id,
        "layout_policy": {
            "camera_pillar_side": "right",
            "task_workspace": "opposite_side_of_camera_pillar",
            "measured_physical_clearance_cm": args.clearance_cm,
        },
        "joints": {
            "shoulder_pan": {
                "min_deg": minimum,
                "max_deg": maximum,
                "blocked_physical_direction": "right_toward_camera_pillar",
                "blocked_encoder_direction": blocked_direction,
                "measured_clear_boundary_deg": round(measured_deg, 3),
                "safety_margin_deg": args.margin_deg,
            }
        },
        "enforcement": {
            "fail_if_unconfigured": True,
            "max_relative_target_deg": 5.0,
        },
    }


def main() -> None:
    args = parse_args()
    if args.margin_deg <= 0:
        raise SystemExit("--margin-deg must be positive")
    if args.clearance_cm < 5:
        raise SystemExit("Keep at least 5 cm of visible clearance from the pillar")

    print("No automatic movement will be commanded. Connecting only to read the encoder.")
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            cameras={},
            use_degrees=True,
            max_relative_target=5.0,
        )
    )
    robot.connect(calibrate=False)
    try:
        robot.bus.disable_torque()
        blocked_direction = args.blocked_direction
        if blocked_direction == "auto":
            input(
                "Torque is OFF. Put the arm in a comfortably clear central pose, then press ENTER: "
            )
            central = float(robot.get_observation()["shoulder_pan.pos"])
            input(
                "Move the shoulder pan a SMALL amount toward the right pillar while staying far away, "
                "then press ENTER: "
            )
            probe = float(robot.get_observation()["shoulder_pan.pos"])
            delta = probe - central
            if abs(delta) < 2.0:
                raise SystemExit(
                    f"Direction probe changed only {delta:.2f} deg. Nothing was saved; retry with a "
                    "clearer but still safe manual movement."
                )
            blocked_direction = "increasing" if delta > 0 else "decreasing"
            print(
                f"Detected pillar direction: {blocked_direction} "
                f"({central:.2f} -> {probe:.2f} deg)."
            )
        input(
            "Torque is OFF. Manually place the whole arm at the right-side SAFE boundary "
            "with at least 5 cm clearance (do not touch the pillar), then press ENTER: "
        )
        observation = robot.get_observation()
        measured = float(observation["shoulder_pan.pos"])
    finally:
        robot.disconnect()

    payload = build_payload(args, measured, blocked_direction)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not args.save:
        print("Read only; no file was written. Add --save after checking the direction and value.")
        return
    confirmation = input(
        "Confirm: arm never touched the pillar, >=5 cm clearance was visible, and direction is correct. "
        "Type SAVE_LIMIT: "
    )
    if confirmation != "SAVE_LIMIT":
        raise SystemExit("Safety limit was not saved.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved local safety limit to {args.output}")


if __name__ == "__main__":
    main()
