"""Fail-closed commissioning checks before any physical SO-101 motion."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

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
PREFLIGHT_MAX_AGE_S = 15 * 60


def required_preflight_check_names():
    return {
        "workspace_profile",
        "table_surface",
        *(f"operator:{name}" for name in REQUIRED_OPERATOR_CHECKS),
        "dual_camera_ready",
        "joint_absolute_limits",
        "overhead_calibration",
    }


def validate_motion_authorization_report(
    report: Mapping[str, object],
    *,
    expected_layout_id: str,
    max_age_s: float = PREFLIGHT_MAX_AGE_S,
    now: datetime | None = None,
):
    """Validate that an active-mode authorization is current and complete."""
    if not isinstance(report, Mapping):
        raise ValueError("Preflight report must be an object")
    if report.get("format_version") != 1:
        raise ValueError("Preflight report has an unsupported format_version")
    if report.get("layout_id") != expected_layout_id:
        raise ValueError(
            "Preflight layout_id does not match the active safety envelope "
            f"({report.get('layout_id')!r} != {expected_layout_id!r})"
        )
    if report.get("motion_authorized") is not True:
        raise ValueError("Preflight report does not authorize robot motion")

    checked_at_raw = report.get("checked_at")
    if not isinstance(checked_at_raw, str):
        raise ValueError("Preflight report checked_at is missing")
    try:
        checked_at = datetime.fromisoformat(checked_at_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Preflight report checked_at is invalid") from exc
    if checked_at.tzinfo is None:
        raise ValueError("Preflight report checked_at must include a timezone")
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        raise ValueError("Preflight validation clock must include a timezone")
    age = current.astimezone(checked_at.tzinfo) - checked_at
    if age < timedelta(seconds=-60):
        raise ValueError("Preflight report timestamp is unexpectedly in the future")
    if age > timedelta(seconds=float(max_age_s)):
        raise ValueError(
            f"Preflight report is stale ({age.total_seconds():.0f}s > {max_age_s:.0f}s)"
        )

    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("Preflight report checks are missing")
    by_name = {}
    for item in checks:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise ValueError("Preflight report contains an invalid check")
        if not isinstance(item.get("passed"), bool) or not isinstance(
            item.get("blocking"), bool
        ):
            raise ValueError("Preflight check passed/blocking fields must be boolean")
        if item["name"] in by_name:
            raise ValueError(f"Preflight report contains duplicate check {item['name']!r}")
        by_name[item["name"]] = item
    required = required_preflight_check_names()
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError(f"Preflight report is missing required checks: {missing}")
    invalid_required = sorted(
        name
        for name in required
        if by_name[name]["blocking"] is not True or by_name[name]["passed"] is not True
    )
    if invalid_required:
        raise ValueError(
            "Preflight required checks must be blocking and passed: "
            f"{invalid_required}"
        )
    failed = sorted(
        name for name, item in by_name.items()
        if item["blocking"] and not item["passed"]
    )
    if failed:
        raise ValueError(f"Preflight report has failed blocking checks: {failed}")
    return True


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
                "joint_absolute_limits",
                layout_matches and limits.fully_configured,
                {
                    "configured": limits.fully_configured,
                    "layout_id": safety_layout,
                    "workspace_layout_id": layout_id,
                    "joints": {
                        name: {
                            "min_deg": limit.min_deg,
                            "max_deg": limit.max_deg,
                            "verified_collision_free": limit.verified_collision_free,
                        }
                        for name, limit in limits.joints.items()
                    },
                },
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        checks.append(_check("joint_absolute_limits", False, str(exc)))

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
