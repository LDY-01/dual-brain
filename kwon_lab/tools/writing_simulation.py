"""Run simulation-only writing preflight and optionally render diagnostic replay."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kwon_lab.writing.planner import PaperSettings, plan_text
from kwon_lab.writing.simulation import PaperFrame, SimulationSettings, ToolMount, TransitSettings, calibration_cases, preflight
from kwon_lab.writing.dynamics import DynamicsSettings, run_dynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="네")
    parser.add_argument("--character-mm", type=float, default=28)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/writing/simulation"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--suite", action="store_true", help="Line, square, call, three glyphs and fixed sentence")
    parser.add_argument("--writing-only", action="store_true", help="Legacy geometry check without approach/return validation")
    parser.add_argument("--dynamics", action="store_true", help="Drive the MuJoCo actuators with real physics timesteps")
    parser.add_argument("--stop-after-s", type=float, help="Request a simulated hold stop after this trajectory time")
    parser.add_argument("--time-scale", type=float, help="Dynamics schedule multiplier, default 2 (slower than preflight)")
    parser.add_argument("--write-example-config", type=Path)
    args = parser.parse_args()
    if args.dynamics and args.writing_only:
        parser.error("--dynamics requires the full-cycle preflight, not --writing-only")
    if (args.stop_after_s is not None or args.time_scale is not None) and not args.dynamics:
        parser.error("--stop-after-s and --time-scale require --dynamics")
    if args.write_example_config:
        config = {"tool": asdict(ToolMount()), "paper_frame": asdict(PaperFrame()),
                  "settings": asdict(SimulationSettings()), "transit": asdict(TransitSettings()),
                  "dynamics": asdict(DynamicsSettings())}
        args.write_example_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_example_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(args.write_example_config)
        return 0
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(config, dict) or set(config) - {"tool", "paper_frame", "settings", "transit", "dynamics"}:
            raise ValueError("Unknown or invalid simulation configuration")
        tool = ToolMount(**config.get("tool", {}))
        frame = PaperFrame(**config.get("paper_frame", {}))
        settings = SimulationSettings(**config.get("settings", {}))
        transit = None if args.writing_only else TransitSettings(**config.get("transit", {}))
        dynamics_config = config.get("dynamics", {})
        if args.time_scale is not None:
            dynamics_config = {**dynamics_config, "time_scale": args.time_scale}
        dynamics = DynamicsSettings(**dynamics_config)
        tool.validate()
        frame.validate(PaperSettings())
        settings.validate()
        dynamics.validate()
        if args.stop_after_s is not None and (not math.isfinite(args.stop_after_s) or args.stop_after_s < 0):
            raise ValueError("stop-after-s must be finite and nonnegative")
        if transit is not None:
            transit.validate(PaperSettings(), settings)
        cases = calibration_cases() if args.suite else {"custom": plan_text(args.text, PaperSettings(character_mm=args.character_mm))}
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, plan in cases.items():
        try:
            if args.dynamics:
                report, model = run_dynamics(plan, tool, frame, settings, transit, dynamics, stop_after_s=args.stop_after_s)
            else:
                report, model = preflight(plan, tool, frame, settings, transit=transit)
        except KeyboardInterrupt:
            print("Stopped. No hardware commands were sent.")
            return 130
        folder = args.output / name if args.suite else args.output
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if args.render and model is not None and report["samples"]:
            from kwon_lab.writing.simulation_preview import render_replay
            render_replay(model, report, folder)
        results[name] = {"status": report["status"], "samples": len(report["samples"]),
                         "scope": report["scope"], "phase": report["phase"],
                         "approach_validated": report["approach_validated"],
                         "departure_validated": report["departure_validated"],
                         "ready_return_validated": report["ready_return_validated"],
                         "simulated_duration_s": report.get("simulated_duration_s"),
                         "max_sampled_tip_speed_mm_s": report.get("max_sampled_tip_speed_mm_s"),
                         "min_transport_tip_height_mm": report.get("min_transport_tip_height_mm"),
                         "phase_sample_counts": report.get("phase_sample_counts"),
                         "failure": report["failure"], "report": str(path),
                         "max_position_error_mm": report.get("max_position_error_mm"),
                         "max_angle_error_deg": report.get("max_angle_error_deg"), "motion_authorized": False}
        if args.dynamics:
            results[name].update({key: report.get(key) for key in
                                 ("dynamics_validated", "physical_dynamics_validated", "pressure_validated",
                                  "simulation_steps", "actual_simulated_duration_s", "max_tracking_error_mm",
                                  "max_path_error_mm", "max_joint_error_rad", "max_axis_error_deg",
                                  "max_joint_speed_rad_s", "max_tip_speed_mm_s", "max_actuator_force_model_units",
                                  "stop_test_passed", "stop_request_trajectory_time_s", "stop_settling_time_s", "stop_max_displacement_mm")})
            for key in ("simulated_duration_s", "max_sampled_tip_speed_mm_s", "min_transport_tip_height_mm",
                        "phase_sample_counts", "max_position_error_mm", "max_angle_error_deg"):
                results[name].pop(key)
        print(json.dumps({"case": name, **results[name]}, ensure_ascii=False))
    summary = {"created_at": report["created_at"], "mode": "simulation_only", "kind": report["kind"],
                   "format_version": report["format_version"],
                   "paper_coordinate_convention": report["paper_coordinate_convention"],
                   "scope": report["scope"], "transit": report["transit"],
                   "motion_authorized": False, "physical_calibration_required": True,
                   "dynamics_validated": all(r.get("dynamics_validated", False) for r in results.values()),
                   "physical_dynamics_validated": False, "pressure_validated": False,
                   "dynamics_settings": report.get("dynamics_settings"), "tool": report["tool"], "paper_frame": report["paper_frame"],
                   "settings": report["settings"], "total_cases": len(results),
                   "passed_cases": sum(r["status"] == "passed" for r in results.values()), "cases": results}
    if args.dynamics:
        summary["model"] = report.get("model")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if all(r["status"] == "passed" for r in results.values()):
        return 0
    return 130 if all(r["status"] == "stopped" for r in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
