#!/usr/bin/env python3
"""Run ACT against live SO-101 observations without sending motor targets."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.real_preflight import evaluate_real_preflight
from hardware.joint_safety import JointLimit, SafetyLimits, load_safety_limits
from hardware.real_so101_adapter import (
    ACTPolicySession,
    JOINT_KEYS,
    JOINT_NAMES,
    JointMapping,
    OpenCVDualCameraSource,
    RealSO101Adapter,
    create_lerobot_follower,
    load_adapter_config,
    policy_action_to_robot,
    prepare_policy_image,
    robot_state_to_policy,
)
from skills.block_reacquisition import (
    load_real_camera_matrices,
    locate_overhead_block,
    locate_overhead_target,
)
from skills.vision_supervision import observe_colors


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def scene_snapshot(wrist, overhead, calibration_path):
    wrist_colors = observe_colors(wrist)
    report = {
        "wrist_red_visible": bool(wrist_colors.red_block.visible),
        "calibration_available": Path(calibration_path).is_file(),
        "overhead_block_visible": False,
        "overhead_target_visible": False,
        "block_pose_class": "UNKNOWN",
        "block_pose_confidence": 0.0,
        "block_table_xy": None,
        "target_table_xy": None,
        "suggested_state": "CALIBRATION_REQUIRED",
    }
    if not report["calibration_available"]:
        overhead_colors = observe_colors(overhead)
        report["overhead_block_visible"] = bool(overhead_colors.red_block.visible)
        report["overhead_target_visible"] = bool(overhead_colors.green_target.visible)
        return report
    block_matrices, target_matrix = load_real_camera_matrices(calibration_path)
    block = locate_overhead_block(overhead, block_matrices)
    target = locate_overhead_target(overhead, target_matrix)
    report.update(
        {
            "overhead_block_visible": bool(block.visible),
            "overhead_target_visible": bool(target.visible),
            "block_pose_class": block.pose_class,
            "block_pose_confidence": float(block.pose_confidence),
            "block_table_xy": block.table_xy,
            "target_table_xy": target.table_xy,
        }
    )
    if not target.visible:
        report["suggested_state"] = "SEARCH_TARGET"
    elif not block.visible:
        report["suggested_state"] = "SEARCH_BLOCK"
    elif block.pose_class == "UNKNOWN" or not block.reachable:
        report["suggested_state"] = "CLEAR_VIEW"
    else:
        report["suggested_state"] = "PICK_CANDIDATE"
    return _json_safe(report)


class _FakeBus:
    def __init__(self, robot):
        self.robot = robot
        self.connected = False
        self.disconnect_disable_torque = None

    def connect(self):
        self.connected = True

    def disconnect(self, disable_torque):
        self.disconnect_disable_torque = disable_torque
        self.connected = False

    def sync_read(self, register):
        if register != "Present_Position":
            raise AssertionError(register)
        return {
            name: self.robot.observation[f"{name}.pos"] for name in JOINT_NAMES
        }


class _FakeRobot:
    def __init__(self, observation):
        self.observation = dict(observation)
        self.bus = _FakeBus(self)
        self.connected = False
        self.send_calls = []

    def connect(self, calibrate=False):
        assert calibrate is False
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        if not (self.connected or self.bus.connected):
            raise RuntimeError("fake robot not connected")
        return dict(self.observation)

    def send_action(self, action):
        self.send_calls.append(dict(action))
        self.observation.update(action)
        return dict(action)


class _FakeCameras:
    def __init__(self):
        self.connected = False
        self.wrist = np.full((720, 1280, 3), 245, dtype=np.uint8)
        self.overhead = self.wrist.copy()
        cv2.rectangle(self.wrist, (580, 320), (700, 440), (220, 15, 15), -1)
        cv2.rectangle(self.overhead, (500, 280), (560, 340), (220, 15, 15), -1)
        cv2.circle(self.overhead, (850, 380), 70, (15, 180, 15), -1)

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def read_rgb(self, role):
        if not self.connected:
            raise RuntimeError("fake cameras not connected")
        return (self.wrist if role == "wrist" else self.overhead).copy()


def run_self_test(adapter_config):
    config = load_adapter_config(adapter_config)
    initial_policy = np.array([0.0, -0.5, 0.6, 0.8, 0.2, 0.8])
    robot_observation = policy_action_to_robot(initial_policy, config)
    recovered = robot_state_to_policy(robot_observation, config)
    image = prepare_policy_image(np.zeros((720, 1280, 3), dtype=np.uint8))
    safety = SafetyLimits(JointLimit(-45.0, 30.0), Path("self_test"))

    with tempfile.TemporaryDirectory() as folder:
        audit = Path(folder) / "shadow.jsonl"
        shadow_robot = _FakeRobot(robot_observation)
        shadow = RealSO101Adapter(
            config,
            mode="shadow",
            robot=shadow_robot,
            cameras=_FakeCameras(),
            safety_limits=safety,
            preflight_report={"motion_authorized": False},
            audit_path=audit,
        )
        with shadow:
            obs = shadow._get_obs()
            requested = np.array([1.2, -0.4, 0.7, 0.9, 0.1, 1.0])
            shadow.step(requested)
            shadow_scene = scene_snapshot(
                obs["pixels"], shadow.render_overhead(), Path(folder) / "missing.json"
            )
            matrix = [[0.0001, 0.0, 0.18], [0.0, 0.0001, -0.10], [0.0, 0.0, 1.0]]
            calibration = Path(folder) / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "planes": {
                            "target_table": {"pixel_to_table_homography": matrix},
                            "upright_top_6cm": {"pixel_to_table_homography": matrix},
                            "tipped_top_4cm": {"pixel_to_table_homography": matrix},
                        },
                    }
                ),
                encoding="utf-8",
            )
            calibrated_scene = scene_snapshot(
                obs["pixels"], shadow.render_overhead(), calibration
            )
        shadow_never_sent = not shadow_robot.send_calls and shadow.goal_position_writes == 0

        active_without_preflight_blocked = False
        try:
            RealSO101Adapter(
                config,
                mode="active",
                robot=_FakeRobot(robot_observation),
                cameras=_FakeCameras(),
                safety_limits=safety,
                preflight_report={"motion_authorized": False},
                active_confirmation=config.confirmation_token,
            )
        except RuntimeError:
            active_without_preflight_blocked = True

        active_unverified_mapping_blocked = False
        try:
            RealSO101Adapter(
                config,
                mode="active",
                robot=_FakeRobot(robot_observation),
                cameras=_FakeCameras(),
                safety_limits=safety,
                preflight_report={"motion_authorized": True},
                active_confirmation=config.confirmation_token,
            )
        except RuntimeError:
            active_unverified_mapping_blocked = True

        verified_config = replace(
            config,
            joint_mappings={
                name: JointMapping(
                    item.robot_units_per_policy_rad,
                    item.robot_zero,
                    True,
                )
                for name, item in config.joint_mappings.items()
            },
        )
        active_robot = _FakeRobot(robot_observation)
        active_without_confirmation_blocked = False
        try:
            RealSO101Adapter(
                verified_config,
                mode="active",
                robot=_FakeRobot(robot_observation),
                cameras=_FakeCameras(),
                safety_limits=safety,
                preflight_report={"motion_authorized": True},
                active_confirmation=None,
            )
        except RuntimeError:
            active_without_confirmation_blocked = True
        active = RealSO101Adapter(
            verified_config,
            mode="active",
            robot=active_robot,
            cameras=_FakeCameras(),
            safety_limits=safety,
            preflight_report={"motion_authorized": True},
            active_confirmation=verified_config.confirmation_token,
        )
        with active:
            active.step(np.array([1.2, -0.4, 0.7, 0.9, 0.1, 1.0]))
        active_sent_once = len(active_robot.send_calls) == 1 and active.goal_position_writes == 1
        pan_was_clamped = active_robot.send_calls[0]["shoulder_pan.pos"] <= 5.0

    report = {
        "mapping_round_trip_max_error": float(np.max(np.abs(recovered - initial_policy))),
        "policy_image_shape": list(image.shape),
        "shadow_goal_position_writes": shadow.goal_position_writes,
        "shadow_robot_send_calls": len(shadow_robot.send_calls),
        "shadow_bus_disconnect_wrote_torque": shadow_robot.bus.disconnect_disable_torque,
        "shadow_audit_recorded": len(shadow.audit_records) == 1,
        "shadow_scene_without_calibration_is_blocked": (
            shadow_scene["suggested_state"] == "CALIBRATION_REQUIRED"
        ),
        "shadow_calibrated_scene_reaches_pick_candidate": (
            calibrated_scene["suggested_state"] == "PICK_CANDIDATE"
        ),
        "active_without_preflight_blocked": active_without_preflight_blocked,
        "active_unverified_mapping_blocked": active_unverified_mapping_blocked,
        "active_without_confirmation_blocked": active_without_confirmation_blocked,
        "active_verified_path_sent_once": active_sent_once,
        "active_relative_pan_clamp_applied": pan_was_clamped,
    }
    report["passed"] = bool(
        report["mapping_round_trip_max_error"] < 1e-12
        and report["policy_image_shape"] == [480, 640, 3]
        and shadow_never_sent
        and report["shadow_bus_disconnect_wrote_torque"] is False
        and report["shadow_audit_recorded"]
        and report["shadow_scene_without_calibration_is_blocked"]
        and report["shadow_calibrated_scene_reaches_pick_candidate"]
        and active_without_preflight_blocked
        and active_unverified_mapping_blocked
        and active_without_confirmation_blocked
        and active_sent_once
        and pan_was_clamped
    )
    print(json.dumps(report, indent=2))
    return report["passed"]


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=Path("config/real_robot_adapter.local.json"),
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("config/real_camera_roles.local.json"),
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=Path("config/real_robot_safety_limits.local.json"),
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("config/overhead_camera_calibration.local.json"),
    )
    parser.add_argument(
        "--workspace-config",
        type=Path,
        default=Path("config/real_workspace.local.json"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/real_policy_shadow"),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(0 if run_self_test(args.adapter_config) else 1)
    if args.cycles < 1:
        raise SystemExit("--cycles must be at least 1")

    config = load_adapter_config(args.adapter_config)
    preflight = evaluate_real_preflight(
        args.camera_config,
        args.safety_config,
        args.calibration_config,
        args.workspace_config,
        probe_cameras=True,
    )
    try:
        safety = load_safety_limits(args.safety_config, require_configured=False)
        safety_config_available = True
    except (OSError, ValueError, json.JSONDecodeError):
        # Shadow never dispatches the action, so it remains useful before the
        # new-site physical pan boundary is measured. The missing guard stays
        # visible in both preflight and the summary and still blocks active mode.
        safety = SafetyLimits(JointLimit(None, None), args.safety_config.resolve())
        safety_config_available = False
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    adapter = RealSO101Adapter(
        config,
        mode="shadow",
        robot=create_lerobot_follower(config),
        cameras=OpenCVDualCameraSource(args.camera_config),
        safety_limits=safety,
        preflight_report=preflight,
        audit_path=output / "actions.jsonl",
    )
    policy = ACTPolicySession(config, device=args.device)
    scene_reports = []
    started = time.perf_counter()
    with adapter:
        obs = adapter._get_obs()
        for index in range(args.cycles):
            cycle_started = time.perf_counter()
            action = policy.select_action(obs)
            scene = scene_snapshot(
                obs["pixels"], adapter.render_overhead(), args.calibration_config
            )
            scene["cycle"] = index + 1
            scene_reports.append(scene)
            if index % max(1, args.snapshot_every) == 0:
                save_rgb(output / f"wrist_{index + 1:03d}.jpg", obs["pixels"])
                save_rgb(
                    output / f"overhead_{index + 1:03d}.jpg",
                    adapter.render_overhead(),
                )
            obs, _, _, _, _ = adapter.step(action)
            delay = 1.0 / config.control_hz - (time.perf_counter() - cycle_started)
            if delay > 0:
                time.sleep(delay)
    if adapter.goal_position_writes != 0:
        raise RuntimeError("Shadow invariant violated: a Goal Position was written")
    summary = {
        "format_version": 1,
        "mode": "shadow",
        "cycles": args.cycles,
        "elapsed_seconds": time.perf_counter() - started,
        "goal_position_writes": adapter.goal_position_writes,
        "robot_send_action_calls": 0,
        "preflight_motion_authorized": preflight["motion_authorized"],
        "joint_mapping_verified": config.mapping_verified,
        "safety_config_available": safety_config_available,
        "suggested_state_counts": {
            state: sum(item["suggested_state"] == state for item in scene_reports)
            for state in sorted({item["suggested_state"] for item in scene_reports})
        },
        "scene_reports": scene_reports,
        "interpretation": (
            "Policy and safety actions were computed from live observations; "
            "no Goal Position command was sent. This does not test physical grasp success."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved shadow evidence: {output}")


if __name__ == "__main__":
    main()
