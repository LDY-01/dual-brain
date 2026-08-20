"""Fail-closed software joint limits for the physical SO101.

These limits are intentionally kept outside LeRobot/site-packages so they are
easy to audit and reverse.  They supplement, rather than replace, LeRobot's
per-command ``max_relative_target`` limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SHOULDER_PAN_KEY = "shoulder_pan.pos"


@dataclass(frozen=True)
class JointLimit:
    min_deg: float | None
    max_deg: float | None

    def clamp(self, value: float) -> float:
        result = float(value)
        if self.min_deg is not None:
            result = max(result, self.min_deg)
        if self.max_deg is not None:
            result = min(result, self.max_deg)
        return result


@dataclass(frozen=True)
class SafetyLimits:
    shoulder_pan: JointLimit
    source_path: Path


def _optional_number(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number or null")
    return float(value)


def load_safety_limits(path: str | Path, *, require_configured: bool = True) -> SafetyLimits:
    """Load and strictly validate a real-robot joint-limit file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Real-robot safety limit file not found: {source}. "
            "Measure the collision-free boundary before enabling robot motion."
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported safety-limit config format_version")
    if payload.get("units") != "degrees":
        raise ValueError("Real-robot safety limits must use degrees")
    joints = payload.get("joints")
    if not isinstance(joints, dict) or not isinstance(joints.get("shoulder_pan"), dict):
        raise ValueError("Safety config must contain joints.shoulder_pan")
    pan = joints["shoulder_pan"]
    minimum = _optional_number(pan.get("min_deg"), "joints.shoulder_pan.min_deg")
    maximum = _optional_number(pan.get("max_deg"), "joints.shoulder_pan.max_deg")
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError("shoulder_pan min_deg must be smaller than max_deg")
    if require_configured and minimum is None and maximum is None:
        raise ValueError(
            "Shoulder-pan safety limit is not measured. Robot motion remains blocked."
        )
    return SafetyLimits(JointLimit(minimum, maximum), source.resolve())


def enforce_action_limits(
    action: Mapping[str, float], limits: SafetyLimits
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Clamp a LeRobot action and return an audit record of each intervention."""
    safe_action = {key: float(value) for key, value in action.items()}
    interventions: list[dict[str, float]] = []
    if SHOULDER_PAN_KEY not in safe_action:
        return safe_action, interventions
    requested = safe_action[SHOULDER_PAN_KEY]
    applied = limits.shoulder_pan.clamp(requested)
    safe_action[SHOULDER_PAN_KEY] = applied
    if applied != requested:
        interventions.append(
            {
                "joint": "shoulder_pan",
                "requested_deg": requested,
                "applied_deg": applied,
            }
        )
    return safe_action, interventions


def assert_position_within_limits(position_deg: float, limits: SafetyLimits) -> None:
    """Refuse powered recovery if the arm already starts beyond the safe boundary."""
    applied = limits.shoulder_pan.clamp(float(position_deg))
    if applied != float(position_deg):
        raise RuntimeError(
            "Current shoulder_pan position is already outside the configured safe range "
            f"({position_deg:.2f} deg). Cut/disable torque and reposition the arm manually; "
            "powered recovery is blocked."
        )


def safe_send_action(robot, action: Mapping[str, float], limits: SafetyLimits):
    """Apply the absolute guard immediately before a physical motor command."""
    current = robot.bus.sync_read("Present_Position")
    assert_position_within_limits(float(current["shoulder_pan"]), limits)
    safe_action, interventions = enforce_action_limits(action, limits)
    sent = robot.send_action(safe_action)
    return sent, interventions
