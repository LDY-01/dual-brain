"""Compare named virtual conditions without changing guards or claiming calibration."""

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import json

from .dynamics import DynamicsSettings, run_dynamics
from .planner import PaperSettings, PenStroke, WritingPlan, plan_text
from .simulation import PaperFrame, SimulationSettings, ToolMount, TransitSettings, calibration_cases


def evaluation_profiles():
    tool, frame, transit = ToolMount(), PaperFrame(), TransitSettings()
    angle = math.radians(10)
    profiles = {"baseline": (tool, frame, transit)}
    for mm in (50, 55, 70, 80):
        profiles[f"tip_{mm}mm"] = (replace(tool, profile_id=f"virtual-tip-{mm}mm", tip_offset_m=(mm / 1000, 0., 0.)), frame, transit)
    profiles["paper_world_y_plus_5mm"] = (tool, replace(frame, origin_m=(.200, -.035, .005)), transit)
    profiles["paper_world_x_minus_5mm"] = (tool, replace(frame, origin_m=(.195, -.040, .005)), transit)
    profiles["paper_yaw_10deg"] = (tool, replace(frame, x_axis=(-math.sin(angle), math.cos(angle), 0.),
                                               y_axis=(math.cos(angle), math.sin(angle), 0.)), transit)
    profiles["work_area_35mm"] = (tool, replace(frame, working_rect_mm=(8., 8., 35., 35.)), transit)
    profiles["travel_30mm"] = (tool, frame, replace(transit, travel_height_mm=30.))
    return profiles


def extended_cases():
    paper = PaperSettings(character_mm=18)
    return {
        "wide_line": WritingPlan("line", paper, (PenStroke("line", 0, ((12., 12.), (140., 40.))),), 1),
        "outside_row": WritingPlan("line", paper, (PenStroke("line", 0, ((12., 90.), (40., 90.))),), 1),
        "long_sentence": plan_text("안녕하세요. 반갑습니다.", paper),
    }


def landscape_profiles():
    """Bounded virtual paper placements; keep tool, ready pose and all guards."""
    frame, tool, transit = PaperFrame(), ToolMount(), TransitSettings()
    origins = ((.200, -.040), (.200, -.090), (.190, -.080), (.180, -.090),
               (.170, -.090), (.160, -.090), (.150, -.090), (.160, -.070), (.150, -.070))
    profiles = {"legacy_candidate": (tool, frame, transit)}
    for x, y in origins:
        name = f"full_a4_origin_{round(x * 1000)}_{round(y * 1000)}mm"
        profiles[name] = (tool, replace(frame, origin_m=(x, y, .005), working_rect_mm=(8., 8., 289., 202.)), transit)
    return profiles


def landscape_sequence_cases():
    paper = PaperSettings(width_mm=297, height_mm=210, character_mm=28)
    return {"call_row_1_28mm": plan_text("네", paper, (11.75, 11.75, 285.25, 198.25)),
            "sentence_row_2_28mm": plan_text("안녕하세요", paper, (11.75, 53.25, 285.25, 198.25))}


def landscape_extent_cases():
    paper = PaperSettings(width_mm=297, height_mm=210, character_mm=28)
    positions = {"upper_left": (11.75, 11.75), "upper_right": (253.25, 11.75),
                 "lower_left": (11.75, 170.25), "lower_right": (253.25, 170.25),
                 "center": (134.5, 91.), "row_3": (11.75, 94.75), "row_4": (11.75, 136.25)}
    return {name: plan_text("네", paper, (x, y, 285.25, 198.25)) for name, (x, y) in positions.items()}


def summarize(report):
    keys = ("kind", "status", "phase", "failure", "trajectory_complete", "simulation_steps",
            "actual_simulated_duration_s", "max_tracking_error_mm", "max_path_error_mm",
            "max_tip_speed_mm_s", "max_joint_speed_rad_s", "dynamics_validated",
            "motion_authorized", "physical_calibration_required", "physical_dynamics_validated",
            "pressure_validated", "stop_test_passed", "preflight")
    return {key: report.get(key) for key in keys}


def run_evaluation(output, progress=None, profiles=None, cases=None):
    """Keep the same ready joints across variants: no automatic pose recalibration."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    profiles = evaluation_profiles() if profiles is None else profiles
    cases = calibration_cases() if cases is None else cases
    settings, dynamics = SimulationSettings(), DynamicsSettings()
    result = {"created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
              "kind": "writing_virtual_condition_evaluation", "mode": "simulation_only",
              "ready_pose_policy": "unchanged_baseline_joints_revalidated_per_condition",
              "settings": asdict(settings), "dynamics_settings": asdict(dynamics),
              "motion_authorized": False, "physical_calibration_required": True,
              "physical_dynamics_validated": False, "pressure_validated": False, "profiles": {}}
    for name, (tool, frame, transit) in profiles.items():
        runs = dict(cases)
        if name == "baseline":
            runs.update(extended_cases())
        profile = {"tool": asdict(tool), "paper_frame": asdict(frame), "transit": asdict(transit), "cases": {}}
        for case, plan in runs.items():
            report, _ = run_dynamics(plan, tool, frame, settings, transit, dynamics)
            profile["cases"][case] = {"text": plan.text, "paper": asdict(plan.paper), **summarize(report)}
            if progress:
                progress(name, case, profile["cases"][case])
        profile["passed_cases"] = sum(r["status"] == "passed" for r in profile["cases"].values())
        profile["total_cases"] = len(runs)
        result["profiles"][name] = profile
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    result["total_cases"] = sum(p["total_cases"] for p in result["profiles"].values())
    result["passed_cases"] = sum(p["passed_cases"] for p in result["profiles"].values())
    result["rejected_cases"] = result["total_cases"] - result["passed_cases"]
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result
