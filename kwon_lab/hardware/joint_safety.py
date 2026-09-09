"""Fail-closed absolute joint limits for the physical SO101.

These limits supplement LeRobot's per-command ``max_relative_target`` guard.
The relative guard limits one command; this module prevents repeated bounded
commands from accumulating outside the physically commissioned task envelope.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)
SHOULDER_PAN_KEY = "shoulder_pan.pos"


@dataclass(frozen=True)
class JointLimit:
    min_deg: float | None
    max_deg: float | None
    verified_collision_free: bool = False

    @property
    def configured(self) -> bool:
        return (
            self.min_deg is not None
            and self.max_deg is not None
            and self.verified_collision_free
        )

    def clamp(self, value: float) -> float:
        result = float(value)
        if self.min_deg is not None:
            result = max(result, self.min_deg)
        if self.max_deg is not None:
            result = min(result, self.max_deg)
        return result


@dataclass(frozen=True)
class SafetyLimits:
    joints: dict[str, JointLimit]
    source_path: Path
    layout_id: str | None = None

    @property
    def shoulder_pan(self) -> JointLimit:
        return self.joints["shoulder_pan"]

    @property
    def fully_configured(self) -> bool:
        return all(
            name in self.joints and self.joints[name].configured for name in JOINT_NAMES
        )


def unbounded_safety_limits(source_path: str | Path, layout_id: str | None = None):
    """Build limits for read-only shadow calculations; never valid for active mode."""
    return SafetyLimits(
        {name: JointLimit(None, None) for name in JOINT_NAMES},
        Path(source_path),
        layout_id,
    )


def _optional_number(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number or null")
    return result


def load_safety_limits(path: str | Path, *, require_configured: bool = True) -> SafetyLimits:
    """Load and strictly validate a real-robot joint-envelope file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Real-robot safety limit file not found: {source}. "
            "Measure the collision-free joint envelope before enabling robot motion."
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported safety-limit config format_version")
    if payload.get("units") != "degrees":
        raise ValueError("Real-robot safety limits must use degrees")
    enforcement = payload.get("enforcement")
    if not isinstance(enforcement, dict):
        raise ValueError("Safety config must contain enforcement")
    if enforcement.get("fail_if_unconfigured") is not True:
        raise ValueError("enforcement.fail_if_unconfigured must remain true")
    if enforcement.get("require_all_joint_envelopes_verified") is not True:
        raise ValueError(
            "enforcement.require_all_joint_envelopes_verified must remain true"
        )
    layout_id = payload.get("layout_id")
    if layout_id is not None and (not isinstance(layout_id, str) or not layout_id.strip()):
        raise ValueError("Safety config layout_id must be a non-empty string")
    joints = payload.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("Safety config must contain a joints object")

    parsed: dict[str, JointLimit] = {}
    missing = []
    incomplete = []
    for name in JOINT_NAMES:
        entry = joints.get(name)
        if not isinstance(entry, dict):
            missing.append(name)
            parsed[name] = JointLimit(None, None)
            continue
        minimum = _optional_number(entry.get("min_deg"), f"joints.{name}.min_deg")
        maximum = _optional_number(entry.get("max_deg"), f"joints.{name}.max_deg")
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"joints.{name}.min_deg must be smaller than max_deg")
        verified = entry.get("verified_collision_free") is True
        parsed[name] = JointLimit(minimum, maximum, verified)
        if minimum is None or maximum is None or not verified:
            incomplete.append(name)

    if require_configured and (missing or incomplete):
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if incomplete:
            detail.append(f"incomplete={incomplete}")
        raise ValueError(
            "All six absolute joint envelopes must have min_deg, max_deg and "
            "verified_collision_free=true before "
            "robot motion: " + ", ".join(detail)
        )
    if require_configured and (not layout_id or layout_id == "UNSET"):
        raise ValueError("Safety config requires the commissioned workspace layout_id")
    return SafetyLimits(parsed, source.resolve(), layout_id)


def enforce_action_limits(
    action: Mapping[str, float], limits: SafetyLimits
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Clamp every configured joint and record each intervention."""
    safe_action = {key: float(value) for key, value in action.items()}
    interventions: list[dict[str, float | str]] = []
    for name in JOINT_NAMES:
        key = f"{name}.pos"
        if key not in safe_action:
            continue
        requested = safe_action[key]
        applied = limits.joints[name].clamp(requested)
        safe_action[key] = applied
        if not math.isclose(applied, requested, abs_tol=1e-12):
            interventions.append(
                {
                    "joint": name,
                    "requested_deg": requested,
                    "applied_deg": applied,
                }
            )
    return safe_action, interventions


def assert_positions_within_limits(
    positions: Mapping[str, float], limits: SafetyLimits
) -> None:
    """Refuse powered motion if any current joint is outside its safe envelope."""
    violations = []
    for name in JOINT_NAMES:
        key = name if name in positions else f"{name}.pos"
        if key not in positions:
            violations.append(f"{name}=missing")
            continue
        value = float(positions[key])
        if not math.isfinite(value):
            violations.append(f"{name}=non_finite")
            continue
        if not math.isclose(limits.joints[name].clamp(value), value, abs_tol=1e-12):
            violations.append(f"{name}={value:.2f}deg")
    if violations:
        raise RuntimeError(
            "Current joint position is outside the configured safe envelope "
            f"({', '.join(violations)}). Cut/disable torque and reposition the arm "
            "manually; powered recovery is blocked."
        )


def assert_position_within_limits(position_deg: float, limits: SafetyLimits) -> None:
    """Backward-compatible shoulder-pan-only assertion for inspection tools."""
    value = float(position_deg)
    if not math.isclose(limits.shoulder_pan.clamp(value), value, abs_tol=1e-12):
        raise RuntimeError(
            "Current shoulder_pan position is already outside the configured safe range "
            f"({value:.2f} deg). Cut/disable torque and reposition the arm manually; "
            "powered recovery is blocked."
        )


def safe_send_action(robot, action: Mapping[str, float], limits: SafetyLimits):
    """Apply every absolute guard immediately before a physical motor command."""
    current = robot.bus.sync_read("Present_Position")
    assert_positions_within_limits(current, limits)
    safe_action, interventions = enforce_action_limits(action, limits)
    sent = robot.send_action(safe_action)
    return sent, interventions
