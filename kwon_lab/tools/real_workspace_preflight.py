#!/usr/bin/env python3
"""Run the real-workspace startup gate without commanding robot motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.real_preflight import evaluate_real_preflight


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
        })
        _write(safety, {
            "format_version": 1,
            "units": "degrees",
            "layout_id": layout_id,
            "joints": {"shoulder_pan": {"min_deg": -90.0, "max_deg": 45.0}},
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
        report = {
            "valid_setup_authorized": passed["motion_authorized"],
            "uncovered_workspace_blocked": not uncovered_workspace["motion_authorized"],
            "unmarked_task_workspace_blocked": not unmarked_workspace["motion_authorized"],
            "missing_emergency_stop_blocked": not blocked["motion_authorized"],
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
