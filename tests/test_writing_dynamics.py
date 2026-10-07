"""Real mj_step integration and fail-closed virtual-actuator/hold-stop tests."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from kwon_lab.writing import dynamics as module
from kwon_lab.writing.dynamics import DynamicsSettings, interpolate_reference, run_dynamics
from kwon_lab.writing.simulation import PaperFrame, TransitSettings, calibration_cases, preflight


class WritingDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = calibration_cases()["call"]
        cls.report, cls.model = run_dynamics(cls.plan)
        cls.stop, _ = run_dynamics(cls.plan, stop_after_s=5)

    def test_settings_reject_nonfinite_boolean_and_unsafe_bounds(self):
        base = DynamicsSettings()
        invalid = [replace(base, timestep_s=True), replace(base, timestep_s=float("nan")),
                   replace(base, timestep_s=.1), replace(base, time_scale=.5), replace(base, time_scale=11),
                   replace(base, tracking_tolerance_mm=3), replace(base, joint_tracking_tolerance_rad=1),
                   replace(base, max_joint_speed_rad_s=3), replace(base, max_tip_speed_mm_s=101),
                   replace(base, max_stop_displacement_mm=6), replace(base, stop_timeout_s=.05),
                   replace(base, stop_stable_s=.01, timestep_s=.01, settle_s=.005),
                   replace(base, final_hold_s=.05), replace(base, stop_tip_speed_mm_s=31),
                   replace(base, stop_joint_speed_rad_s=1), replace(base, max_steps=True),
                   replace(base, max_steps=0)]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()

    def test_invalid_stop_time_is_rejected(self):
        for time in (-1, True, float("nan"), float("inf"), "5"):
            with self.subTest(time=time), self.assertRaises(ValueError):
                run_dynamics(self.plan, stop_after_s=time)

    def test_reference_interpolation_retains_incoming_phase_and_duplicate_times(self):
        samples = [{"arm_q_rad": [q] * 5, "target_world_m": [q] * 3,
                    "op": phase, "stroke": -1, "phase": phase}
                   for q, phase in ((0, "initialization"), (0, "approach"), (2, "writing"))]
        q, xyz, op, _, phase = interpolate_reference(samples, [0, 0, 2], .5)
        np.testing.assert_allclose(q, [.5] * 5)
        np.testing.assert_allclose(xyz, [.5] * 3)
        self.assertEqual((op, phase), ("writing", "writing"))
        q, *_ = interpolate_reference(samples, [0, 0, 2], 10)
        np.testing.assert_allclose(q, [2] * 5)

    def test_seven_dynamics_cycles_pass_without_claiming_physical_validation(self):
        for name, plan in calibration_cases().items():
            with self.subTest(case=name):
                report, _ = run_dynamics(plan)
                self.assertEqual(report["status"], "passed", report["failure"])
                self.assertTrue(report["dynamics_validated"])
                self.assertTrue(report["trajectory_complete"])
                self.assertTrue(report["ready_return_validated"])
                for key in ("physical_dynamics_validated", "pressure_validated", "pen_mass_and_compliance_calibrated",
                            "motion_authorized", "joint_command_stream_available"):
                    self.assertFalse(report[key])
                self.assertTrue(report["physical_calibration_required"])
                self.assertLessEqual(report["max_tracking_error_mm"], 1)
                self.assertLessEqual(report["max_path_error_mm"], 1.75)
                self.assertLessEqual(report["max_joint_speed_rad_s"], .75)
                self.assertLessEqual(report["max_tip_speed_mm_s"], 30)
                self.assertLessEqual(report["max_axis_error_deg"], 3)
                self.assertLessEqual(report["max_joint_error_rad"], .02)
                self.assertLessEqual(report["max_actuator_force_model_units"], 2.94 + 1e-9)
                self.assertEqual(report["model"]["gravity_m_s2"], [0, 0, -9.81])
                self.assertEqual(report["model"]["kp"], [998.22] * 6)
                self.assertEqual(report["model"]["pen_proxy_mass_kg"], 0)
                json.dumps(report, allow_nan=False)

    def test_actual_trajectory_uses_physics_instead_of_copying_reference_positions(self):
        report = self.report
        self.assertEqual(report["status"], "passed", report["failure"])
        samples = report["samples"]
        self.assertGreater(len(samples), 4000)
        self.assertTrue(any(np.max(np.abs(np.array(s["arm_q_rad"]) - s["reference_q_rad"])) > .0001 for s in samples))
        self.assertTrue(any(np.linalg.norm(s["arm_qvel_rad_s"]) > .01 for s in samples))
        np.testing.assert_allclose(np.diff([s["time_s"] for s in samples]), .005, atol=1e-9)
        self.assertGreaterEqual(report["actual_simulated_duration_s"], report["execution_reference_duration_s"] + .8 - 1e-8)
        np.testing.assert_allclose(samples[-1]["arm_q_rad"], TransitSettings().ready_q_rad, atol=.002)

    def test_original_time_schedule_is_rejected_without_loosening_speed_limit(self):
        report, _ = run_dynamics(self.plan, dynamics=replace(DynamicsSettings(), time_scale=1))
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["failure"]["reason"], "actual_tip_overspeed")
        self.assertGreater(report["max_tip_speed_mm_s"], 30)
        self.assertFalse(report["dynamics_validated"])
        self.assertEqual(report["dynamics_settings"]["max_tip_speed_mm_s"], self.report["dynamics_settings"]["max_tip_speed_mm_s"])

    def test_weaker_actuators_cannot_be_false_success(self):
        original = module.preflight

        def weak_model(*args, **kwargs):
            report, model = original(*args, **kwargs)
            model.actuator_forcerange[:] *= .01
            return report, model

        with patch.object(module, "preflight", weak_model):
            report, _ = run_dynamics(self.plan)
        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["trajectory_complete"])
        self.assertFalse(report["dynamics_validated"])
        self.assertIsNotNone(report["failure"])
        self.assertLess(report["model"]["force_ranges_model_units"][0][1], .03)

    def test_failed_preflight_never_advances_physics(self):
        with patch.object(module.mujoco, "mj_step") as step:
            report, _ = run_dynamics(self.plan, frame=replace(PaperFrame(), origin_m=(2, 0, .005)))
        step.assert_not_called()
        self.assertEqual(report["failure"]["reason"], "preflight_not_passed")
        self.assertEqual(report["samples"], [])

    def test_writing_only_preflight_cannot_authorize_dynamics(self):
        reference, model = preflight(self.plan)
        with patch.object(module, "preflight", return_value=(reference, model)), patch.object(module.mujoco, "mj_step") as step:
            report, _ = run_dynamics(self.plan)
        step.assert_not_called()
        self.assertEqual(report["failure"]["reason"], "preflight_not_passed")

    def test_hold_stop_settles_without_resetting_velocity_or_resuming_strokes(self):
        report = self.stop
        self.assertEqual(report["status"], "stopped", report["failure"])
        self.assertTrue(report["stop_settled"])
        self.assertTrue(report["stop_test_passed"])
        self.assertFalse(report["trajectory_complete"])
        self.assertFalse(report["dynamics_validated"])
        self.assertLessEqual(report["stop_settling_time_s"], 1)
        self.assertLessEqual(report["stop_max_displacement_mm"], 2)
        samples = report["samples"]
        index = next(i for i, s in enumerate(samples) if s["phase"] == "stopping")
        self.assertTrue(any(s["op"] == "stroke" for s in samples[:index]))
        self.assertTrue(all(s["op"] == "hold_stop" for s in samples[index:]))
        np.testing.assert_allclose(report["stop_command_q_rad"], samples[index - 1]["arm_q_rad"], atol=1e-12)
        self.assertGreater(np.linalg.norm(samples[index]["arm_qvel_rad_s"]), 1e-5)
        for sample in samples[index:]:
            np.testing.assert_allclose(sample["command_q_rad"], report["stop_command_q_rad"], atol=1e-12)
        self.assertLessEqual(samples[-1]["tip_speed_mm_s"], 2)
        self.assertLessEqual(samples[-1]["joint_speed_rad_s"], .02)

    def test_stop_at_zero_trajectory_time_produces_no_strokes(self):
        report, _ = run_dynamics(self.plan, stop_after_s=0)
        self.assertEqual(report["status"], "stopped", report["failure"])
        self.assertTrue(report["stop_test_passed"])
        self.assertFalse(any(s["op"] == "stroke" for s in report["samples"]))

    def test_stop_timeout_is_not_reported_as_settled(self):
        report, _ = run_dynamics(self.plan, dynamics=replace(DynamicsSettings(), stop_timeout_s=.1), stop_after_s=5)
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["failure"]["reason"], "stop_not_settled")
        self.assertFalse(report["stop_test_passed"])

    def test_stop_displacement_guard_rejects_excess_motion(self):
        report, _ = run_dynamics(self.plan, dynamics=replace(DynamicsSettings(), max_stop_displacement_mm=.001), stop_after_s=5)
        self.assertEqual(report["failure"]["reason"], "stop_displacement_exceeded")
        self.assertFalse(report["stop_test_passed"])

    def test_step_budget_bounds_the_physics_loop(self):
        report, _ = run_dynamics(self.plan, dynamics=replace(DynamicsSettings(), max_steps=1))
        self.assertEqual(report["failure"]["reason"], "dynamics_step_budget_exceeded")
        self.assertEqual(report["simulation_steps"], 1)
        self.assertFalse(report["trajectory_complete"])

    def test_nonfinite_state_or_engine_reset_is_rejected_and_json_remains_finite(self):
        def bad_step(model, data):
            data.qpos[0] = float("nan")
            data.time = float("nan")

        with patch.object(module.mujoco, "mj_step", bad_step):
            report, _ = run_dynamics(self.plan)
        self.assertEqual(report["failure"]["reason"], "nonfinite_state_or_engine_reset")
        self.assertEqual(report["samples"], [])
        json.dumps(report, allow_nan=False)

    def test_numerical_warning_rejects_rollout(self):
        original = mujoco.mj_step

        def warn(model, data):
            original(model, data)
            data.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number = 1

        with patch.object(module.mujoco, "mj_step", warn):
            report, _ = run_dynamics(self.plan)
        self.assertEqual(report["failure"]["reason"], "mujoco_numerical_warning")
        self.assertFalse(report["dynamics_validated"])

    def test_keyboard_interrupt_aborts_simulation_without_claiming_tested_hold_stop(self):
        with patch.object(module.mujoco, "mj_step", side_effect=KeyboardInterrupt):
            report, _ = run_dynamics(self.plan)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["failure"]["reason"], "keyboard_interrupt_simulation_aborted")
        self.assertFalse(report["stop_test_passed"])

    def test_cli_dynamics_success_stop_and_invalid_mode_combinations(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, "-X", "utf8", "kwon_lab/tools/writing_simulation.py"]
        with tempfile.TemporaryDirectory() as temp:
            for args, code, status in ((["--dynamics"], 0, "passed"),
                                       (["--dynamics", "--stop-after-s", "5"], 130, "stopped")):
                result = subprocess.run([*command, *args, "--output", temp], cwd=root,
                                        capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, code, result.stderr)
                report = json.loads((Path(temp) / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], status)
                self.assertEqual(report["kind"], "writing_dynamics_rollout")
            for args in (["--dynamics", "--writing-only"], ["--stop-after-s", "5"],
                         ["--dynamics", "--time-scale", ".5"], ["--dynamics", "--stop-after-s", "nan"]):
                result = subprocess.run([*command, *args, "--output", str(Path(temp) / "invalid")],
                                        cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse((Path(temp) / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
