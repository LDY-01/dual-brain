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
    "task_workspace_marked",
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


def _convex_hull(points):
    """Return a counter-clockwise 2-D convex hull without extra dependencies."""
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) < 3:
        return []

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _inside_convex_hull(point, hull, tolerance=1e-9):
    if len(hull) < 3:
        return False
    signs = []
    for index, first in enumerate(hull):
        second = hull[(index + 1) % len(hull)]
        cross = ((second[0] - first[0]) * (point[1] - first[1])
                 - (second[1] - first[1]) * (point[0] - first[0]))
        if abs(cross) > tolerance:
            signs.append(cross > 0)
    return not signs or all(sign == signs[0] for sign in signs)


def _expanded_workspace_corners(workspace):
    task = workspace.get("task_workspace")
    if not isinstance(task, dict):
        raise ValueError("Workspace profile requires task_workspace")
    x_bounds = np.asarray(task.get("x_bounds_m"), dtype=float)
    y_bounds = np.asarray(task.get("y_bounds_m"), dtype=float)
    margin = task.get("recovery_guard_margin_m")
    if (x_bounds.shape != (2,) or y_bounds.shape != (2,)
            or not np.isfinite(x_bounds).all() or not np.isfinite(y_bounds).all()
            or x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]
            or not isinstance(margin, (int, float)) or margin < 0):
        raise ValueError("task_workspace requires ordered finite bounds and a nonnegative guard")
    return [
        (x_bounds[0] - margin, y_bounds[0] - margin),
        (x_bounds[0] - margin, y_bounds[1] + margin),
        (x_bounds[1] + margin, y_bounds[0] - margin),
        (x_bounds[1] + margin, y_bounds[1] + margin),
    ]


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
    _expanded_workspace_corners(payload)
    return payload


def validate_overhead_calibration(
    path, *, expected_layout_id, overhead_entry, workspace=None
):
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
    if workspace is not None:
        reference_points = target.get("reference_points", [])
        try:
            table_points = np.asarray(
                [item["table_xy_m"] for item in reference_points], dtype=float
            )
            if (table_points.ndim != 2 or table_points.shape[1] != 2
                    or len(table_points) < 4 or not np.isfinite(table_points).all()):
                raise ValueError
            hull = _convex_hull(table_points)
            outside = [
                [float(value) for value in corner]
                for corner in _expanded_workspace_corners(workspace)
                if not _inside_convex_hull(corner, hull)
            ]
            if outside:
                errors.append(
                    "calibration reference-point hull does not cover task workspace "
                    f"plus recovery guard; outside corners={outside}"
                )
        except (KeyError, TypeError, ValueError, IndexError):
            errors.append(
                "target_table reference_points are missing or invalid; "
                "workspace coverage cannot be verified"
            )
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
            workspace=workspace,
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
