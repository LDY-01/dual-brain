"""Simulation-only contracts; no serial devices, cameras or model APIs."""

from dataclasses import replace
import json
import math
import unittest

import mujoco
import numpy as np

from kwon_lab.writing.planner import PaperSettings, PenStroke, WritingPlan, plan_text
from kwon_lab.writing.dynamics import DynamicsSettings
from kwon_lab.writing.simulation import (
    PaperFrame, PenKinematics, SimulationSettings, ToolMount, TransitSettings, build_scene,
    calibration_cases, paper_camera_pose, preflight,
)


class WritingSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_scene()

    def test_configuration_requires_uncalibrated_simulation(self):
        for tool in (replace(ToolMount(), calibrated=True), replace(ToolMount(), simulation_only=False),
                     replace(ToolMount(), calibrated=0), replace(ToolMount(), simulation_only=1)):
            with self.assertRaises(ValueError):
                tool.validate()

    def test_configuration_rejects_invalid_numbers_axes_and_geometry(self):
        invalid = [replace(ToolMount(), tip_offset_m=(math.nan, 0, 0)),
                   replace(ToolMount(), axis_local=(0, 0, 0)),
                   replace(ToolMount(), tip_offset_m=(True, 0, 0)),
                   replace(ToolMount(), shaft_radius_m=-1),
                   replace(ToolMount(), nib_length_m=1),
                   replace(ToolMount(), gripper_rad=math.inf)]
        for tool in invalid:
            with self.assertRaises(ValueError):
                tool.validate()

    def test_page_coordinates_are_right_down_away_and_mm_to_m(self):
        frame = PaperFrame()
        frame.validate(PaperSettings())
        np.testing.assert_allclose(frame.world([10, 20, 8]), [.220, -.030, .013])
        np.testing.assert_allclose(frame.paper_mm([.220, -.030, .013]), [10, 20, 8])
        np.testing.assert_allclose(frame.rotation.T @ frame.rotation, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(frame.rotation), -1)
        np.testing.assert_allclose(frame.normal, [0, 0, 1])
        rotated = replace(frame, x_axis=(1., 0., 0.), y_axis=(0., -1., 0.))
        np.testing.assert_allclose(rotated.world([10, 20, 8]), [.210, -.060, .013])

    def test_paper_geom_has_proper_rotation_and_top_surface_at_origin(self):
        frame = PaperFrame()
        self.assertAlmostEqual(np.linalg.det(frame.geometry_rotation), 1)
        kin = PenKinematics(self.model)
        kin.pose(np.zeros(5))
        geom = self.model.geom("writing_paper").id
        rotation = kin.data.geom_xmat[geom].reshape(3, 3)
        np.testing.assert_allclose(rotation, frame.geometry_rotation, atol=1e-9)
        top_center = kin.data.geom_xpos[geom] + rotation[:, 2] * self.model.geom_size[geom, 2]
        np.testing.assert_allclose(top_center, frame.world([105, 148.5, 0]), atol=1e-9)
        self.assertGreater(frame.world([0, 0, 8])[2], top_center[2])

    def test_camera_projects_page_right_and_down_without_mirroring(self):
        paper = PaperSettings()
        for frame in (PaperFrame(), replace(PaperFrame(), x_axis=(1., 0., 0.), y_axis=(0., -1., 0.))):
            frame.validate(paper)
            position, rotation = paper_camera_pose(frame, paper)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
            right, up, backward = rotation.T
            self.assertGreater(float(right @ frame.x_axis), .99)
            self.assertAlmostEqual(float(right @ frame.y_axis), 0)
            self.assertLess(float(up @ frame.y_axis), -.8)
            self.assertGreater(float(backward @ frame.normal), .8)
            self.assertGreater(float((position - frame.world([105, 118.8, 0])) @ frame.normal), 0)

    def test_scene_camera_matches_paper_aligned_pose(self):
        position, rotation = paper_camera_pose(PaperFrame(), PaperSettings())
        kin = PenKinematics(self.model)
        kin.pose(np.zeros(5))
        camera = self.model.camera("writing_view").id
        np.testing.assert_allclose(kin.data.cam_xpos[camera], position, atol=1e-9)
        np.testing.assert_allclose(kin.data.cam_xmat[camera].reshape(3, 3), rotation, atol=1e-9)

    def test_paper_frame_rejects_invalid_axes_and_working_area(self):
        for frame in (replace(PaperFrame(), x_axis=(2., 0., 0.)),
                      replace(PaperFrame(), y_axis=PaperFrame().x_axis),
                      replace(PaperFrame(), working_rect_mm=(0, 0, 210, 297)),
                      replace(PaperFrame(), origin_m=(0, math.nan, 0))):
            with self.assertRaises(ValueError):
                frame.validate(PaperSettings())

    def test_settings_reject_unsafe_sampling_and_invalid_budgets(self):
        for settings in (replace(SimulationSettings(), cartesian_step_mm=3),
                         replace(SimulationSettings(), max_joint_step_rad=.2),
                         replace(SimulationSettings(), angle_tolerance_deg=180),
                         replace(SimulationSettings(), joint_speed_rad_s=0),
                         replace(SimulationSettings(), max_points=True),
                         replace(SimulationSettings(), ik_iterations=0)):
            with self.assertRaises(ValueError):
                settings.validate()

    def test_tip_site_uses_gripper_local_mount_not_world_offset(self):
        kin = PenKinematics(self.model)
        tip, axis = kin.pose([.2, -.4, .5, .8, .3])
        grip = self.model.site("gripperframe").id
        rotation = kin.data.site_xmat[grip].reshape(3, 3)
        expected = kin.data.site_xpos[grip] + rotation @ np.array(ToolMount().tip_offset_m)
        np.testing.assert_allclose(tip, expected, atol=1e-9)
        np.testing.assert_allclose(axis, rotation @ np.array(ToolMount().axis_local), atol=1e-9)
        self.assertGreater(np.linalg.norm(tip - kin.data.site_xpos[grip]), .05)

    def test_pen_tip_jacobian_matches_finite_difference(self):
        kin = PenKinematics(self.model)
        q = np.array([.2, -.4, .5, .8, .3])
        tip, axis = kin.pose(q)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, kin.data, jacp, jacr, kin.tip_id)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        axis_jac = -skew @ jacr[:, kin.dofs]
        for index in range(5):
            perturbed = q.copy()
            perturbed[index] += 1e-6
            new_tip, new_axis = kin.pose(perturbed)
            np.testing.assert_allclose((new_tip - tip) / 1e-6, jacp[:, kin.dofs[index]], atol=1e-6)
            np.testing.assert_allclose((new_axis - axis) / 1e-6, axis_jac[:, index], atol=1e-6)

    def test_opposite_axis_cannot_be_false_success(self):
        kin = PenKinematics(self.model, settings=SimulationSettings(ik_iterations=1))
        q = np.zeros(5)
        tip, axis = kin.pose(q)
        result = kin.solve(tip, -axis, q)
        self.assertFalse(result["converged"])
        self.assertGreater(result["angle_error_deg"], 90)

    def test_invalid_joint_seed_and_gripper_are_rejected(self):
        kin = PenKinematics(self.model)
        with self.assertRaises(ValueError):
            kin.pose([4, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            kin.solve([.22, 0, .01], [0, 0, -1], [4, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            PenKinematics(self.model, replace(ToolMount(), gripper_rad=10))

    def test_seven_geometry_cases_pass_without_authorizing_motion(self):
        for case, plan in calibration_cases().items():
            with self.subTest(case=case):
                report, model = preflight(plan)
                self.assertEqual(report["status"], "passed", report["failure"])
                self.assertTrue(report["trajectory_complete"])
                self.assertFalse(report["motion_authorized"])
                self.assertFalse(report["joint_command_stream_available"])
                self.assertFalse(report["dynamics_validated"])
                self.assertTrue(report["physical_calibration_required"])
                self.assertIsNotNone(model)
                self.assertGreater(len(report["samples"]), 20)
                self.assertLessEqual(report["max_position_error_mm"], .75)
                self.assertLessEqual(report["max_angle_error_deg"], 3)
                self.assertLessEqual(report["max_interpolated_position_error_mm"], .75)
                self.assertLessEqual(report["max_interpolated_angle_error_deg"], 3)
                json.dumps(report, allow_nan=False)

    def test_joint_continuity_timing_and_fixed_gripper(self):
        report, _ = preflight(plan_text("네", PaperSettings(character_mm=28)))
        settings = SimulationSettings()
        samples = report["samples"]
        self.assertEqual({s["op"] for s in samples}, {"travel", "pen_down", "stroke", "pen_up"})
        for previous, current in zip(samples, samples[1:]):
            delta = np.max(np.abs(np.array(current["arm_q_rad"]) - previous["arm_q_rad"]))
            dt = current["time_s"] - previous["time_s"]
            self.assertLessEqual(delta, settings.max_joint_step_rad)
            self.assertLessEqual(delta, settings.joint_speed_rad_s * dt + 1e-9)
            self.assertEqual(len(current["arm_q_rad"]), 5)
        self.assertEqual(report["tool"]["gripper_rad"], .1)

    def test_mount_change_forces_new_preflight_not_same_path_acceptance(self):
        plan = plan_text("네", PaperSettings(character_mm=28))
        report, _ = preflight(plan, replace(ToolMount(), tip_offset_m=(.08, 0, 0), profile_id="virtual-80mm"))
        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["trajectory_complete"])
        self.assertEqual(report["failure"]["reason"], "ik_not_converged")

    def test_outside_working_area_fails_before_model_loading(self):
        report, model = preflight(plan_text("네"), frame=replace(PaperFrame(), working_rect_mm=(100, 100, 150, 150)))
        self.assertEqual(report["failure"]["reason"], "outside_candidate_work_area")
        self.assertIsNone(model)
        self.assertEqual(report["samples"], [])

    def test_unreachable_paper_placement_is_rejected(self):
        report, _ = preflight(plan_text("네"), frame=replace(PaperFrame(), origin_m=(2., 0., .005)))
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["failure"]["reason"], "ik_not_converged")

    def test_collision_rejects_whole_plan_not_partial_execution(self):
        report, _ = preflight(plan_text("네"), tool=replace(ToolMount(), shaft_radius_m=.04))
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["failure"]["reason"], "collision")
        self.assertTrue(report["failure"]["contacts"])
        self.assertFalse(report["joint_command_stream_available"])
        self.assertFalse(report["trajectory_complete"])

    def test_joint_jump_and_axis_change_are_rejected(self):
        report, _ = preflight(plan_text("네"), settings=replace(SimulationSettings(), max_joint_step_rad=.000001))
        self.assertEqual(report["failure"]["reason"], "joint_discontinuity")
        self.assertFalse(report["trajectory_complete"])
        report, _ = preflight(plan_text("네"), tool=replace(ToolMount(), axis_local=(0., 1., 0.)))
        self.assertEqual(report["failure"]["reason"], "ik_not_converged")

    def test_example_profile_matches_defaults(self):
        from dataclasses import asdict
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "kwon_lab/writing/config/virtual_mount.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        expected = {"tool": asdict(ToolMount()), "paper_frame": asdict(PaperFrame()),
                    "settings": asdict(SimulationSettings()), "transit": asdict(TransitSettings()),
                    "dynamics": asdict(DynamicsSettings())}
        self.assertEqual(config, json.loads(json.dumps(expected)))

    def test_pen_paper_penetration_is_detected(self):
        seed = [0, 0, 0, math.pi / 2, 0]
        tip, _ = PenKinematics(self.model).pose(seed)
        base = PaperFrame()
        origin = tip - np.asarray(base.x_axis) * .01 - np.asarray(base.y_axis) * .01 + base.normal * .0006
        frame = replace(base, origin_m=tuple(origin))
        model = build_scene(frame=frame)
        kin = PenKinematics(model)
        kin.pose(seed)
        self.assertTrue(any(set(c["geoms"]) == {"pen_nib", "writing_paper"} for c in kin.collisions()))

    def test_stop_is_fail_closed(self):
        report, _ = preflight(plan_text("네"), stop_requested=lambda: True)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["failure"]["reason"], "stop_requested")
        self.assertFalse(report["trajectory_complete"])
        self.assertFalse(report["joint_command_stream_available"])

    def test_stop_during_computation_and_point_budget(self):
        count = 0

        def stop():
            nonlocal count
            count += 1
            return count >= 1000

        report, _ = preflight(calibration_cases()["sentence"], stop_requested=stop)
        self.assertEqual(report["status"], "stopped")
        report, _ = preflight(plan_text("네"), settings=SimulationSettings(max_points=2))
        self.assertEqual(report["failure"]["reason"], "point_budget_exceeded")
        self.assertEqual(len(report["samples"]), 2)
        self.assertFalse(report["trajectory_complete"])

    def test_malformed_manual_strokes_are_rejected(self):
        for points in ((), ((math.nan, 10),), ((True, 10),)):
            with self.assertRaises(ValueError):
                preflight(WritingPlan("검사", PaperSettings(), (PenStroke("검", 0, points),), 1))

    def test_empty_plan_has_no_success_or_commands(self):
        report, _ = preflight(WritingPlan("빈 경로", PaperSettings(), (), 1))
        self.assertEqual(report["failure"]["reason"], "empty_trajectory")
        self.assertFalse(report["trajectory_complete"])


if __name__ == "__main__":
    unittest.main()
