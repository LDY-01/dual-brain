from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from kwon_lab.writing.evaluation import evaluation_profiles, extended_cases, landscape_extent_cases, landscape_profiles, landscape_sequence_cases, run_evaluation
from kwon_lab.writing.planner import PaperSettings
from kwon_lab.writing.simulation import calibration_cases


class ConditionEvaluationTests(unittest.TestCase):
    def test_landscape_profiles_keep_every_guard_and_explicit_sequence_positions(self):
        profiles = landscape_profiles()
        self.assertEqual(len(profiles), 10)
        baseline = profiles["legacy_candidate"]
        for name, (tool, frame, transit) in profiles.items():
            self.assertEqual(tool, baseline[0])
            self.assertEqual(transit, baseline[2])
            frame.validate(PaperSettings(width_mm=297, height_mm=210))
            if name != "legacy_candidate":
                self.assertEqual(frame.working_rect_mm, (8., 8., 289., 202.))
        cases = landscape_sequence_cases()
        for name, y in (("call_row_1_28mm", 11.75), ("sentence_row_2_28mm", 53.25)):
            points = [p for stroke in cases[name].strokes for p in stroke.points_mm]
            self.assertTrue(all(y <= p[1] <= y + 28 for p in points))
        self.assertEqual(len(landscape_extent_cases()), 7)

    def test_profiles_are_bounded_uncalibrated_and_keep_ready_joints(self):
        profiles = evaluation_profiles()
        self.assertEqual(len(profiles), 10)
        ready = profiles["baseline"][2].ready_q_rad
        for tool, frame, transit in profiles.values():
            tool.validate()
            frame.validate(PaperSettings())
            self.assertEqual(transit.ready_q_rad, ready)
            self.assertFalse(tool.calibrated)
            self.assertTrue(tool.simulation_only)

    def test_extended_cases_do_not_claim_supported_glyphs_for_line_probes(self):
        cases = extended_cases()
        self.assertEqual(set(cases), {"wide_line", "outside_row", "long_sentence"})
        self.assertEqual(cases["outside_row"].strokes[0].points_mm[0][1], 90.)
        self.assertGreater(len(cases["long_sentence"].strokes), len(calibration_cases()["sentence"].strokes))

    def test_subset_evaluation_records_clearance_rejections_and_parameters(self):
        profiles = {name: value for name, value in evaluation_profiles().items() if name in {"tip_70mm", "tip_80mm"}}
        with tempfile.TemporaryDirectory() as directory:
            result = run_evaluation(directory, profiles=profiles, cases={"call": calibration_cases()["call"]})
            self.assertEqual(result["total_cases"], 2)
            self.assertEqual(result["rejected_cases"], 2)
            self.assertEqual(json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8")), json.loads(json.dumps(result)))
            for name, profile in result["profiles"].items():
                self.assertEqual(profile["tool"], asdict(profiles[name][0]))
                case = profile["cases"]["call"]
                self.assertEqual(case["failure"]["detail"]["reason"], "insufficient_clearance")
                self.assertFalse(case["motion_authorized"])
                self.assertFalse(case["pressure_validated"])


if __name__ == "__main__":
    unittest.main()
