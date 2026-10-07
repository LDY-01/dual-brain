"""Airborne ready-pose approach/exit preflight, without any hardware I/O."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kwon_lab.writing import simulation
from kwon_lab.writing.planner import PaperSettings, PenStroke, WritingPlan
from kwon_lab.writing.simulation import (
    PaperFrame, PenKinematics, SimulationSettings, ToolMount, TransitSettings,
    calibration_cases, preflight,
)


class WritingTransitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = calibration_cases()["call"]
        cls.transit = TransitSettings()
        cls.report, cls.model = preflight(cls.plan, transit=cls.transit)

    def test_transit_config_rejects_invalid_pose_height_and_clearance(self):
        for transit in (replace(self.transit, ready_q_rad=(0, 0, 0, 0)),
                        replace(self.transit, ready_q_rad=(True, 0, 0, 0, 0)),
                        replace(self.transit, ready_q_rad=(float("nan"), 0, 0, 0, 0)),
                        replace(self.transit, travel_height_mm=float("inf")),
                        replace(self.transit, travel_height_mm=1001),
                        replace(self.transit, travel_height_mm=12),
                        replace(self.transit, travel_height_mm=7, minimum_clearance_mm=4),
                        replace(self.transit, minimum_clearance_mm=0)):
            with self.subTest(transit=transit), self.assertRaises(ValueError):
                transit.validate(self.plan.paper, SimulationSettings())

    def test_transit_pose_cannot_be_confused_with_an_ik_seed(self):
        with self.assertRaises(ValueError):
            preflight(self.plan, transit=self.transit, initial_q=np.zeros(5))

    def test_seven_full_cycles_pass_and_return_to_configured_pose(self):
        phases = {"initialization": 0, "approach": 1, "writing": 2, "departure": 3, "return": 4}
        for name, plan in calibration_cases().items():
            with self.subTest(case=name):
                report, _ = preflight(plan, transit=self.transit)
                self.assertEqual(report["status"], "passed", report["failure"])
                self.assertEqual(report["scope"], "full_cycle")
                self.assertEqual(report["phase"], "complete")
                for flag in ("initial_pose_validated", "approach_validated", "departure_validated", "ready_return_validated"):
                    self.assertTrue(report[flag])
                for flag in ("motion_authorized", "joint_command_stream_available", "dynamics_validated"):
                    self.assertFalse(report[flag])
                self.assertTrue(report["physical_calibration_required"])
                samples = report["samples"]
                np.testing.assert_allclose(samples[0]["arm_q_rad"], self.transit.ready_q_rad, atol=1e-12)
                np.testing.assert_allclose(samples[-1]["arm_q_rad"], self.transit.ready_q_rad, atol=1e-12)
                self.assertEqual(samples[0]["op"], "ready")
                self.assertEqual(samples[-1]["op"], "return_ready")
                order = [phases[s["phase"]] for s in samples]
                self.assertEqual(order, sorted(order))
                self.assertTrue(all(report["phase_sample_counts"].values()))
                self.assertLessEqual(report["max_position_error_mm"], .75)
                self.assertLessEqual(report["max_interpolated_position_error_mm"], .75)
                self.assertGreaterEqual(report["min_transport_tip_height_mm"], 12)
                self.assertLessEqual(report["max_sampled_tip_speed_mm_s"], 20 + 1e-9)
                json.dumps(report, allow_nan=False)

    def test_initial_joint_limit_violation_fails_before_any_sample(self):
        report, _ = preflight(self.plan, transit=replace(self.transit, ready_q_rad=(4, 0, 0, 0, 0)))
        self.assertEqual(report["failure"]["reason"], "initial_pose_joint_limits")
        self.assertEqual(report["samples"], [])
        self.assertFalse(report["initial_pose_validated"])

    def test_initial_low_pose_is_rejected_instead_of_teleporting_to_first_point(self):
        writing, _ = preflight(self.plan)
        low_q = writing["samples"][0]["arm_q_rad"]
        report, _ = preflight(self.plan, transit=replace(self.transit, ready_q_rad=tuple(low_q)))
        self.assertEqual(report["failure"]["reason"], "insufficient_clearance")
        self.assertEqual(report["failure"]["phase"], "initialization")
        self.assertEqual(report["samples"], [])
        self.assertFalse(report["approach_validated"])

    def test_initial_direction_is_checked(self):
        q = list(self.transit.ready_q_rad)
        q[3] -= .2
        report, _ = preflight(self.plan, transit=replace(self.transit, ready_q_rad=tuple(q)))
        self.assertEqual(report["failure"]["reason"], "tracking_error")
        self.assertGreater(report["failure"]["angle_error_deg"], 3)
        self.assertEqual(report["samples"], [])

    def test_initial_collision_is_checked(self):
        report, _ = preflight(self.plan, transit=self.transit, tool=replace(ToolMount(), shaft_radius_m=.04))
        self.assertEqual(report["failure"]["reason"], "collision")
        self.assertEqual(report["failure"]["phase"], "initialization")
        self.assertFalse(report["motion_authorized"])

    def test_unreachable_approach_height_rejects_the_job(self):
        report, _ = preflight(self.plan, transit=replace(self.transit, travel_height_mm=30))
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["failure"]["phase"], "approach")
        self.assertEqual(report["failure"]["reason"], "ik_not_converged")
        self.assertFalse(report["trajectory_complete"])
        self.assertFalse(any(s["phase"] == "writing" for s in report["samples"]))

    def test_writable_line_is_rejected_if_departure_is_unreachable(self):
        plan = WritingPlan("exit control", PaperSettings(),
                           (PenStroke("line", 0, ((12., 12.), (140., 40.))),), 1)
        writing, _ = preflight(plan)
        full, _ = preflight(plan, transit=self.transit)
        self.assertEqual(writing["status"], "passed")
        self.assertEqual(full["status"], "rejected")
        self.assertEqual(full["failure"]["phase"], "departure")
        self.assertEqual(full["failure"]["reason"], "ik_not_converged")
        self.assertTrue(full["approach_validated"])
        self.assertFalse(full["departure_validated"])
        self.assertFalse(full["ready_return_validated"])
        self.assertFalse(full["trajectory_complete"])
        self.assertFalse(full["motion_authorized"])

    def test_stop_during_each_phase_is_fail_closed(self):
        original = simulation._targets
        for wanted in ("approach", "writing", "departure", "return"):
            phase = "initialization"

            def observed_targets(*args):
                nonlocal phase
                for row in original(*args):
                    phase = row[3]
                    yield row
                phase = "return"

            with self.subTest(phase=wanted), patch.object(simulation, "_targets", observed_targets):
                report, _ = preflight(self.plan, transit=self.transit, stop_requested=lambda: phase == wanted)
                self.assertEqual(report["status"], "stopped")
                self.assertEqual(report["failure"]["phase"], wanted)
                self.assertFalse(report["trajectory_complete"])
                self.assertFalse(report["ready_return_validated"])
                self.assertFalse(report["joint_command_stream_available"])

    def test_point_budget_covers_approach_writing_departure_and_return(self):
        samples = self.report["samples"]
        for phase in ("approach", "writing", "departure", "return"):
            budget = next(i for i, s in enumerate(samples) if s["phase"] == phase)
            with self.subTest(phase=phase):
                report, _ = preflight(self.plan, transit=self.transit,
                                      settings=replace(SimulationSettings(), max_points=budget))
                self.assertEqual(report["failure"]["reason"], "point_budget_exceeded")
                self.assertEqual(report["failure"]["phase"], phase)
                self.assertEqual(len(report["samples"]), budget)
                self.assertFalse(report["trajectory_complete"])

    def test_transit_intervals_have_clearance_and_bounded_joint_speed(self):
        kin = PenKinematics(self.model)
        frame, settings = PaperFrame(), SimulationSettings()
        samples = self.report["samples"]
        for previous, current in zip(samples, samples[1:]):
            a, b = np.array(previous["arm_q_rad"]), np.array(current["arm_q_rad"])
            delta = float(np.max(np.abs(b - a)))
            dt = current["time_s"] - previous["time_s"]
            self.assertLessEqual(delta, settings.max_joint_step_rad)
            self.assertLessEqual(delta, settings.joint_speed_rad_s * dt + 1e-9)
            distance_mm = np.linalg.norm(np.array(current["target_world_m"]) - previous["target_world_m"]) * 1000
            self.assertLessEqual(distance_mm, settings.cartesian_step_mm + 1e-9)
            self.assertLessEqual(distance_mm, settings.tip_speed_mm_s * dt + 1e-9)
            tips = [np.asarray(previous["tip_world_m"])]
            for t in (.25, .5, .75, 1):
                tip, _ = kin.pose(a + t * (b - a))
                tips.append(tip)
            for start, end in zip(tips, tips[1:]):
                self.assertLessEqual(float(np.linalg.norm(end - start)) * 1000,
                                     settings.tip_speed_mm_s * dt / 4 + 1e-9)
            if current["phase"] == "writing":
                continue
            minimum = (self.plan.paper.pen_lift_mm - settings.position_tolerance_mm
                       if current["op"] in ("approach_lower", "departure_lift") else self.transit.minimum_clearance_mm)
            for t in (.25, .5, .75, 1):
                tip, _ = kin.pose(a + t * (b - a))
                self.assertGreaterEqual(frame.paper_mm(tip)[2], minimum - 1e-9)
                self.assertFalse(kin.collisions())
            self.assertEqual(current["stroke"], -1)

    def test_writing_only_mode_makes_no_approach_or_exit_claims(self):
        report, _ = preflight(self.plan)
        self.assertEqual(report["scope"], "writing_only")
        self.assertIsNone(report["transit"])
        self.assertEqual({s["phase"] for s in report["samples"]}, {"writing"})
        for flag in ("initial_pose_validated", "approach_validated", "departure_validated", "ready_return_validated"):
            self.assertFalse(report[flag])

    def test_cli_defaults_to_full_cycle_and_supports_explicit_writing_only(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            for flag, scope in (([], "full_cycle"), (["--writing-only"], "writing_only")):
                result = subprocess.run([sys.executable, "-X", "utf8", "kwon_lab/tools/writing_simulation.py",
                                         "--output", temp, *flag], cwd=root, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((Path(temp) / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["scope"], scope)
                self.assertEqual(report["ready_return_validated"], scope == "full_cycle")

    def test_cli_example_includes_transit_and_invalid_transit_fails_cleanly(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            command = [sys.executable, "-X", "utf8", "kwon_lab/tools/writing_simulation.py"]
            result = subprocess.run([*command, "--write-example-config", str(path)], cwd=root, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["transit"]["travel_height_mm"], 16)
            config["transit"]["minimum_clearance_mm"] = 20
            path.write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run([*command, "--config", str(path), "--output", str(Path(temp) / "job")],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((Path(temp) / "job").exists())


if __name__ == "__main__":
    unittest.main()
