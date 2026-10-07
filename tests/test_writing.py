from dataclasses import replace
from math import hypot, isfinite
from pathlib import Path
import tempfile
import unittest
import unicodedata
import xml.etree.ElementTree as ET

from kwon_lab.writing import PaperSettings, plan_text
from kwon_lab.writing.preview import export_preview, to_svg
from kwon_lab.writing.review import REVIEW_CASES, export_review
from kwon_lab.writing.strokes import glyph


class WritingTests(unittest.TestCase):
    def test_default_paper_is_a4_portrait(self):
        paper = PaperSettings()
        self.assertEqual((paper.width_mm, paper.height_mm), (210, 297))
        self.assertEqual(plan_text("네").paper, paper)

    def test_all_modern_hangul_have_finite_bounded_strokes(self):
        for codepoint in range(0xAC00, 0xD7A4):
            strokes = glyph(chr(codepoint))
            self.assertTrue(strokes)
            for stroke in strokes:
                self.assertGreaterEqual(len(stroke), 2)
                self.assertTrue(all(isfinite(x) and isfinite(y) and 0 <= x <= 1
                                    and 0 <= y <= 1 for x, y in stroke))

    def test_decomposed_input_is_normalized(self):
        self.assertEqual(plan_text("안녕").to_dict(),
                         plan_text(unicodedata.normalize("NFD", "안녕")).to_dict())

    def test_wrap_and_overflow(self):
        paper = PaperSettings(width_mm=58, height_mm=58)
        self.assertEqual(plan_text("가나다라", paper).lines, 2)
        with self.assertRaises(ValueError):
            plan_text("가나다라마", paper)

    def test_spacing_and_paper_margins(self):
        plan = plan_text("괜찮아요. 함께 해봐요!")
        for stroke in plan.strokes:
            for x, y in stroke.points_mm:
                self.assertTrue(plan.paper.margin_mm <= x <= plan.paper.width_mm - plan.paper.margin_mm)
                self.assertTrue(plan.paper.margin_mm <= y <= plan.paper.height_mm - plan.paper.margin_mm)
            for first, second in zip(stroke.points_mm, stroke.points_mm[1:]):
                self.assertLessEqual(hypot(second[0] - first[0], second[1] - first[1]),
                                     plan.paper.sample_spacing_mm + 1e-9)

    def test_pen_is_up_between_strokes(self):
        plan = plan_text("안녕")
        commands = plan.commands()
        for index in range(0, len(commands), 4):
            self.assertEqual([command["op"] for command in commands[index:index + 4]],
                             ["travel", "pen_down", "stroke", "pen_up"])
            self.assertEqual(commands[index]["point_mm"][2], plan.paper.pen_lift_mm)
            self.assertEqual(commands[index + 3]["point_mm"][2], plan.paper.pen_lift_mm)
        payload = plan.to_dict()
        self.assertFalse(payload["motion_authorized"])
        self.assertTrue(payload["physical_calibration_required"])
        self.assertNotIn("joint_targets", payload)

    def test_rejects_unsupported_or_invalid_input(self):
        for text in ("", "   ", "안녕🙂", "hello", "안녕\n하세요", "안녕\t하세요", "가" * 121):
            with self.subTest(text=text), self.assertRaises(ValueError):
                plan_text(text)

    def test_rejects_invalid_settings(self):
        for changes in ({"width_mm": float("nan")}, {"character_mm": 0},
                        {"pen_lift_mm": 0}, {"sample_spacing_mm": .001},
                        {"margin_mm": 100}, {"width_mm": True}, {"line_gap_mm": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_text("안녕", replace(PaperSettings(), **changes))

    def test_exports_valid_svg_json_and_image(self):
        import json
        from PIL import Image

        plan = plan_text("안녕하세요.")
        for animated in (False, True):
            root = ET.fromstring(to_svg(plan, animate=animated))
            self.assertEqual(len(root.findall(".//{http://www.w3.org/2000/svg}path")), len(plan.strokes))
        with tempfile.TemporaryDirectory() as directory:
            paths = export_preview(plan, Path(directory))
            self.assertEqual(json.loads(paths["plan.json"].read_text(encoding="utf-8"))["text"], plan.text)
            with Image.open(paths["preview.png"]) as image:
                self.assertEqual(image.size, (1260, 1782))
                self.assertTrue(all(low < high for low, high in image.getextrema()))

    def test_review_samples_cover_glyph_structures_and_wrapping(self):
        plans = {case_id: plan_text(text, paper) for case_id, text, paper in REVIEW_CASES}
        self.assertEqual(plans["large_glyphs"].text, "가 한 글")
        self.assertEqual(plans["large_glyphs"].paper.character_mm, 24)
        self.assertEqual(plans["fixed_sentence"].text, "안녕하세요.")
        self.assertGreater(plans["wrapped_sentence"].lines, 1)
        for plan in plans.values():
            self.assertFalse(plan.to_dict()["motion_authorized"])
            self.assertTrue(plan.to_dict()["physical_calibration_required"])

    def test_review_manifest_keeps_human_and_physical_validation_pending(self):
        from datetime import datetime, timedelta
        import json
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads(export_review(root).read_text(encoding="utf-8"))
            self.assertEqual(manifest["milestone"], "M0")
            self.assertEqual(manifest["human_review_status"], "pending")
            self.assertEqual(manifest["physical_trials"], 0)
            self.assertFalse(manifest["motion_authorized"])
            self.assertTrue(manifest["physical_calibration_required"])
            self.assertEqual(datetime.fromisoformat(manifest["generated_at"]).utcoffset(),
                             timedelta(hours=9))
            self.assertEqual(len(manifest["cases"]), len(REVIEW_CASES))
            for case in manifest["cases"]:
                self.assertEqual(case["human_review_status"], "pending")
                self.assertEqual(len(case["files"]), 4)
                for filename in case["files"].values():
                    self.assertTrue((root / filename).is_file())
                with Image.open(root / case["files"]["preview.png"]) as image:
                    self.assertTrue(all(low < high for low, high in image.getextrema()))


if __name__ == "__main__":
    unittest.main()
