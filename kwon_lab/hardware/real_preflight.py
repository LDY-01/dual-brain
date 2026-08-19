"""Fail-closed commissioning checks before any physical SO-101 motion."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np

from hardware.camera_roles import camera_registry_status, load_camera_registry
from hardware.joint_safety import load_safety_limits


REQUIRED_PLANES = ("target_table", "upright_top_6cm", "tipped_top_4cm")
REQUIRED_OPERATOR_CHECKS = (
    "layout_locked",
    "camera_pillar_secured",
    "robot_base_secured",
    "workspace_clear",
    "emergency_stop_ready",
    "camera_views_visually_confirmed",
)


def _read_json(path):
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def _check(name, passed, detail, *, blocking=True):
    return {
        "name": name,
        "passed": bool(passed),
        "blocking": bool(blocking),
        "detail": detail,
    }


def load_workspace_profile(path):
    payload = _read_json(path)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported workspace profile format_version")
    layout_id = payload.get("layout_id")
    if not isinstance(layout_id, str) or not layout_id or layout_id == "UNSET":
        raise ValueError("Workspace profile requires a non-placeholder layout_id")
    checks = payload.get("operator_checks")
    if not isinstance(checks, dict):
        raise ValueError("Workspace profile requires operator_checks")
    return payload


def validate_overhead_calibration(path, *, expected_layout_id, overhead_entry):
    payload = _read_json(path)
    errors = []
    if payload.get("format_version") != 1:
        errors.append("unsupported format_version")
    if payload.get("layout_id") != expected_layout_id:
        errors.append(
            f"layout_id mismatch: calibration={payload.get('layout_id')!r}, "
            f"workspace={expected_layout_id!r}"
        )
    if overhead_entry is None:
        errors.append("overhead camera role is not registered")
    elif payload.get("camera_index") != overhead_entry.get("index"):
        errors.append("calibration camera_index does not match registered overhead camera")
    planes = payload.get("planes", {})
    for name in REQUIRED_PLANES:
        plane = planes.get(name)
        if not isinstance(plane, dict):
            errors.append(f"missing calibration plane {name}")
            continue
        matrix = np.asarray(plane.get("pixel_to_table_homography"), dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            errors.append(f"invalid 3x3 homography for {name}")
    target = planes.get("target_table", {})
    max_error = target.get("max_error_m")
    if not isinstance(max_error, (int, float)):
        errors.append("target_table max_error_m is missing")
    elif max_error > 0.010:
        errors.append(f"target_table max error is {max_error*1000:.1f} mm (>10 mm)")
    return {"valid": not errors, "errors": errors, "payload": payload}


def evaluate_real_preflight(
    camera_config,
    safety_config,
    calibration_config,
    workspace_config,
    *,
    probe_cameras=True,
    camera_status_override=None,
):
    """Return an auditable gate report; never commands or connects robot motors."""
    checks = []
    workspace = None
    layout_id = None
    try:
        workspace = load_workspace_profile(workspace_config)
        layout_id = workspace["layout_id"]
        checks.append(_check("workspace_profile", True, f"layout_id={layout_id}"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        checks.append(_check("workspace_profile", False, str(exc)))

    if workspace is not None:
        surface = workspace.get("table", {}).get("surface")
        checks.append(
            _check(
                "table_surface",
                surface == "white_matte_ceramic"
                and workspace.get("table", {}).get("surface_confirmed") is True,
                f"surface={surface!r}",
            )
        )
        for name in REQUIRED_OPERATOR_CHECKS:
            passed = workspace["operator_checks"].get(name) is True
            checks.append(_check(f"operator:{name}", passed, "confirmed" if passed else "not confirmed"))

    registry = None
    camera_status = None
    try:
        registry = load_camera_registry(camera_config)
        camera_status = camera_status_override or camera_registry_status(
            camera_config, probe=probe_cameras
        )
        checks.append(
            _check(
                "dual_camera_ready",
                camera_status["dual_camera_ready"],
                camera_status["roles"],
            )
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        checks.append(_check("dual_camera_ready", False, str(exc)))

    try:
        limits = load_safety_limits(safety_config, require_configured=True)
        safety_payload = _read_json(safety_config)
        safety_layout = safety_payload.get("layout_id")
        layout_matches = layout_id is not None and safety_layout == layout_id
        checks.append(
            _check(
                "shoulder_pan_limit",
                layout_matches,
                {
                    "configured": True,
                    "layout_id": safety_layout,
                    "workspace_layout_id": layout_id,
                    "min_deg": limits.shoulder_pan.min_deg,
                    "max_deg": limits.shoulder_pan.max_deg,
                },
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        checks.append(_check("shoulder_pan_limit", False, str(exc)))

    try:
        overhead_entry = registry["roles"].get("overhead") if registry else None
        calibration = validate_overhead_calibration(
            calibration_config,
            expected_layout_id=layout_id,
            overhead_entry=overhead_entry,
        )
        checks.append(
            _check(
                "overhead_calibration",
                calibration["valid"],
                calibration["errors"] or "all required planes valid; max error <=10 mm",
            )
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        checks.append(_check("overhead_calibration", False, str(exc)))

    motion_authorized = all(
        item["passed"] for item in checks if item["blocking"]
    )
    return {
        "format_version": 1,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "layout_id": layout_id,
        "motion_authorized": motion_authorized,
        "checks": checks,
        "camera_status": camera_status,
        "policy": "fail_closed; this report does not itself command robot motion",
    }
