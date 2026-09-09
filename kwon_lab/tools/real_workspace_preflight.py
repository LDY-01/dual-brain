#!/usr/bin/env python3
"""Run the real-workspace startup gate without commanding robot motion."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.real_preflight import (
    evaluate_real_preflight,
    validate_motion_authorization_report,
)


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_self_test():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        camera = root / "cameras.json"
        safety = root / "safety.json"
        calibration = root / "calibration.json"
        workspace = root / "workspace.json"
        layout_id = "self_test_layout"
        _write(camera, {
            "format_version": 1,
            "roles": {
                "wrist": {"index": 1, "backend": "dshow"},
                "overhead": {"index": 2, "backend": "dshow"},
            },
            "startup_policy": {
                "require_distinct_indices": True,
                "require_view_confirmation_after_usb_change": True,
                "view_confirmation_max_age_s": 300,
                "block_robot_motion_when_incomplete": True,
            },
        })
        _write(safety, {
            "format_version": 1,
            "units": "degrees",
            "layout_id": layout_id,
            "joints": {
                "shoulder_pan": {"min_deg": -90.0, "max_deg": 45.0, "verified_collision_free": True},
                "shoulder_lift": {"min_deg": -90.0, "max_deg": 90.0, "verified_collision_free": True},
                "elbow_flex": {"min_deg": -90.0, "max_deg": 90.0, "verified_collision_free": True},
                "wrist_flex": {"min_deg": -90.0, "max_deg": 90.0, "verified_collision_free": True},
                "wrist_roll": {"min_deg": -90.0, "max_deg": 90.0, "verified_collision_free": True},
                "gripper": {"min_deg": -10.0, "max_deg": 100.0, "verified_collision_free": True},
            },
            "enforcement": {
                "fail_if_unconfigured": True,
                "require_all_joint_envelopes_verified": True,
                "max_relative_target_deg": 5.0,
            },
        })
        matrix = [[0.001, 0, -0.4], [0, -0.001, 0.36], [0, 0, 1]]
        _write(calibration, {
            "format_version": 1,
            "layout_id": layout_id,
            "camera_index": 2,
            "planes": {
                "target_table": {
                    "pixel_to_table_homography": matrix,
                    "max_error_m": 0.004,
                    "reference_points": [
                        {"table_xy_m": [0.08, -0.18]},
                        {"table_xy_m": [0.08, 0.26]},
                        {"table_xy_m": [0.34, -0.18]},
                        {"table_xy_m": [0.34, 0.26]},
                    ],
                },
                "upright_top_6cm": {"pixel_to_table_homography": matrix},
                "tipped_top_4cm": {"pixel_to_table_homography": matrix},
            },
        })
        operator = {
            "layout_locked": True,
            "camera_pillar_secured": True,
            "robot_base_secured": True,
            "workspace_clear": True,
            "task_workspace_marked": True,
            "emergency_stop_ready": True,
            "camera_views_visually_confirmed": True,
        }
        _write(workspace, {
            "format_version": 1,
            "layout_id": layout_id,
            "table": {"surface": "white_matte_ceramic", "surface_confirmed": True},
            "task_workspace": {
                "x_bounds_m": [0.10, 0.32],
                "y_bounds_m": [-0.16, 0.24],
                "recovery_guard_margin_m": 0.02,
            },
            "operator_checks": operator,
        })
        camera_status = {
            "dual_camera_ready": True,
            "roles": {
                "wrist": {"registered": True, "available": True},
                "overhead": {"registered": True, "available": True},
            },
        }
        passed = evaluate_real_preflight(
            camera, safety, calibration, workspace,
            probe_cameras=False,
            camera_status_override=camera_status,
        )
        safety_payload = json.loads(safety.read_text(encoding="utf-8"))
        safety_payload["joints"]["gripper"]["verified_collision_free"] = False
        _write(safety, safety_payload)
        incomplete_joint_envelope = evaluate_real_preflight(
            camera,
            safety,
            calibration,
            workspace,
            probe_cameras=False,
            camera_status_override=camera_status,
        )
        safety_payload["joints"]["gripper"]["verified_collision_free"] = True
        _write(safety, safety_payload)
        oversized_workspace_payload = {
            "format_version": 1,
            "layout_id": layout_id,
            "table": {"surface": "white_matte_ceramic", "surface_confirmed": True},
            "task_workspace": {
                "x_bounds_m": [0.10, 0.40],
                "y_bounds_m": [-0.16, 0.24],
                "recovery_guard_margin_m": 0.02,
            },
            "operator_checks": operator,
        }
        _write(workspace, oversized_workspace_payload)
        uncovered_workspace = evaluate_real_preflight(
            camera, safety, calibration, workspace,
            probe_cameras=False,
            camera_status_override=camera_status,
        )
        operator["task_workspace_marked"] = False
        _write(workspace, {
            "format_version": 1,
            "layout_id": layout_id,
            "table": {"surface": "white_matte_ceramic", "surface_confirmed": True},
            "task_workspace": {
                "x_bounds_m": [0.10, 0.32],
                "y_bounds_m": [-0.16, 0.24],
                "recovery_guard_margin_m": 0.02,
            },
            "operator_checks": operator,
        })
        unmarked_workspace = evaluate_real_preflight(
            camera, safety, calibration, workspace,
            probe_cameras=False,
            camera_status_override=camera_status,
        )
        operator["task_workspace_marked"] = True
        operator["emergency_stop_ready"] = False
        _write(workspace, {
            "format_version": 1,
            "layout_id": layout_id,
            "table": {"surface": "white_matte_ceramic", "surface_confirmed": True},
            "task_workspace": {
                "x_bounds_m": [0.10, 0.32],
                "y_bounds_m": [-0.16, 0.24],
                "recovery_guard_margin_m": 0.02,
            },
            "operator_checks": operator,
        })
        blocked = evaluate_real_preflight(
            camera, safety, calibration, workspace,
            probe_cameras=False,
            camera_status_override=camera_status,
        )
        valid_report_accepted = validate_motion_authorization_report(
            passed, expected_layout_id=layout_id
        )
        stale_report = dict(passed)
        stale_report["checked_at"] = "2000-01-01T00:00:00+09:00"
        stale_report_blocked = False
        try:
            validate_motion_authorization_report(
                stale_report, expected_layout_id=layout_id
            )
        except ValueError:
            stale_report_blocked = True
        nonblocking_required_report = json.loads(json.dumps(passed))
        nonblocking_required_report["checks"][0]["blocking"] = False
        nonblocking_required_report_blocked = False
        try:
            validate_motion_authorization_report(
                nonblocking_required_report, expected_layout_id=layout_id
            )
        except ValueError:
            nonblocking_required_report_blocked = True
        report = {
            "valid_setup_authorized": passed["motion_authorized"],
            "incomplete_joint_envelope_blocked": not incomplete_joint_envelope[
                "motion_authorized"
            ],
            "uncovered_workspace_blocked": not uncovered_workspace["motion_authorized"],
            "unmarked_task_workspace_blocked": not unmarked_workspace["motion_authorized"],
            "missing_emergency_stop_blocked": not blocked["motion_authorized"],
            "valid_report_accepted": valid_report_accepted,
            "stale_report_blocked": stale_report_blocked,
            "nonblocking_required_report_blocked": (
                nonblocking_required_report_blocked
            ),
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
    parser.add_argument("--output", type=Path, default=Path("results/real_preflight/latest.json"))
    parser.add_argument("--skip-live-camera", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(0 if run_self_test() else 1)
    report = evaluate_real_preflight(
        args.camera_config,
        args.safety_config,
        args.calibration_config,
        args.workspace_config,
        probe_cameras=not args.skip_live_camera,
    )
    if args.skip_live_camera:
        report["motion_authorized"] = False
        report["policy"] += "; --skip-live-camera always forces motion_authorized=false"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["motion_authorized"] else 2)


if __name__ == "__main__":
    main()
