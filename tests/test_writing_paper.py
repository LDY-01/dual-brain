from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from kwon_lab.writing.jobs import JobBusyError, JobManager, atomic_json, default_config, failure_category, legacy_config
from kwon_lab.writing.paper import PageLedger, PaperSpaceError
from kwon_lab.writing.planner import PaperSettings, plan_text


def good_report():
    return {"kind": "writing_dynamics_rollout", "scope": "full_cycle", "status": "passed",
            "trajectory_complete": True, "dynamics_validated": True, "ready_return_validated": True,
            "motion_authorized": False, "physical_calibration_required": True,
            "physical_dynamics_validated": False, "pressure_validated": False,
            "actual_simulated_duration_s": 1., "samples": [], "failure": None}


class PaperLayoutTests(unittest.TestCase):
    def test_failure_categories_separate_reach_safety_and_recording(self):
        self.assertEqual(failure_category({"reason": "preflight_not_passed", "detail": {"reason": "ik_not_converged"}}), "robot_reach")
        self.assertEqual(failure_category({"reason": "reference_exceeds_actuator_control_range"}), "robot_reach")
        self.assertEqual(failure_category({"reason": "preflight_not_passed", "detail": {"reason": "collision"}}), "robot_validation")
        self.assertEqual(failure_category({"reason": "actual_tracking_error"}), "robot_validation")
        self.assertEqual(failure_category({"reason": "execution_or_record_error"}), "execution_record")
        self.assertIsNone(failure_category(None))

    def test_atomic_replace_retries_temporary_denial_but_not_permanent_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            import os
            original = os.replace
            attempts = []
            def temporary_lock(source, destination):
                attempts.append(1)
                if len(attempts) < 3:
                    raise PermissionError("temporary reader lock")
                original(source, destination)
            with patch("kwon_lab.writing.jobs.os.replace", side_effect=temporary_lock), patch("kwon_lab.writing.jobs.time.sleep"):
                atomic_json(target, {"saved": True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(target.read_text()), {"saved": True})
            with patch("kwon_lab.writing.jobs.os.replace", side_effect=PermissionError("permanent denial")) as denied, patch("kwon_lab.writing.jobs.time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_json(target, {"saved": False})
            self.assertEqual(denied.call_count, 7)
            self.assertEqual(json.loads(target.read_text()), {"saved": True})

    def test_layout_rect_translates_and_wraps_without_changing_a4(self):
        paper = PaperSettings(character_mm=18)
        plan = plan_text("가나다라마바사아자차", paper, (12., 40., 80., 140.))
        self.assertEqual(plan.paper, paper)
        self.assertGreater(plan.lines, 1)
        points = [p for s in plan.strokes for p in s.points_mm]
        self.assertTrue(all(12 <= x <= 80 and 40 <= y <= 140 for x, y in points))
        self.assertEqual(plan.to_dict()["paper"]["width_mm"], 210)

    def test_invalid_layouts_cannot_escape_paper_or_change_font_to_fit(self):
        for rect in ((7, 8, 100, 100), (8, 8, 203, 100), (8, 8, 10, 10), (8, 8, 100, float("nan")), (True, 8, 100, 100), []):
            with self.assertRaises(ValueError):
                plan_text("네", PaperSettings(character_mm=28), rect)


class PaperJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.managers = []

    def tearDown(self):
        for manager in reversed(self.managers):
            manager.close()
        self.directory.cleanup()

    def manager(self, **kwargs):
        manager = JobManager(self.directory.name, **kwargs)
        self.managers.append(manager)
        return manager

    def finish(self, manager, job, seconds=90):
        deadline = time.monotonic() + seconds
        while manager.list()["active_job_id"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.002)
        return manager.get(job["job_id"])

    def test_landscape_full_paper_has_six_disjoint_rows_before_space_is_blocked(self):
        manager = self.manager(runner=lambda *args: good_report())
        first = self.finish(manager, manager.submit("네", 18))
        second = self.finish(manager, manager.submit("한", 18))
        a, b = first["placement"]["reserved_rect_mm"], second["placement"]["reserved_rect_mm"]
        self.assertLess(a[3], b[1])
        self.assertEqual(first["placement"]["page_id"], second["placement"]["page_id"])
        self.assertEqual(manager.page()["working_rect_mm"], [8, 8, 289, 202])
        self.assertEqual((first["paper"]["width_mm"], first["paper"]["height_mm"]), (297, 210))
        for _ in range(4):
            self.finish(manager, manager.submit("글", 18))
        before = len(manager.list()["jobs"])
        with self.assertRaises(PaperSpaceError):
            manager.submit("글", 18)
        self.assertEqual(len(manager.list()["jobs"]), before)
        self.assertEqual(manager.page()["state"], "awaiting_new_sheet")
        self.assertTrue(all(r["state"] == "occupied" for r in manager.page()["reservations"]))

    def test_28mm_sentence_fits_one_landscape_row_beyond_robot_candidate_width(self):
        manager = self.manager(config=legacy_config(), runner=lambda *args: good_report())
        preview = manager.preview("안녕하세요", 28)
        self.assertEqual(preview["placement"]["lines"], 1)
        self.assertEqual(preview["placement"]["reserved_rect_mm"], [8, 8, 289, 43.5])
        points = [p for stroke in preview["plan"]["strokes"] for p in stroke["points_mm"]]
        self.assertGreater(max(p[0] for p in points), manager.page()["robot_candidate_rect_mm"][2])
        self.assertEqual(manager.page()["state"], "ready")

    def test_blank_paper_overflow_is_distinct_and_does_not_mark_current_sheet_full(self):
        manager = self.manager(runner=lambda *args: good_report())
        before = manager.page()
        with self.assertRaises(PaperSpaceError) as error:
            manager.submit("가" * 120, 40)
        self.assertEqual(error.exception.reason, "text_does_not_fit_blank_sheet")
        self.assertEqual(manager.page(), before)
        self.assertEqual(manager.list()["jobs"], [])

    def test_actual_robot_reach_failure_is_not_paper_space_and_releases_reservation(self):
        manager = self.manager(config=legacy_config())
        before = manager.page()
        result = self.finish(manager, manager.submit("안녕하세요", 28))
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["failure_category"], "robot_reach")
        self.assertEqual(result["failure"]["detail"]["reason"], "outside_candidate_work_area")
        self.assertEqual(result["page_reservation_state"], "released_no_motion")
        self.assertEqual(manager.page()["next_row_y_mm"], before["next_row_y_mm"])
        self.assertEqual(manager.page()["state"], "ready")

    def test_portrait_ledger_is_preserved_until_explicit_landscape_confirmation(self):
        manager = self.manager(runner=lambda *args: good_report())
        self.finish(manager, manager.submit("네", 18))
        old = deepcopy(manager.pages.book)
        manager.close(); self.managers.remove(manager)
        page = old["pages"][0]
        page["paper"] = asdict(PaperSettings())
        page["working_rect_mm"] = [8., 8., 150., 80.]
        page["reservations"][0]["reserved_rect_mm"][2] = 150.
        page["config_signature"] = "a" * 64
        atomic_json(Path(self.directory.name) / "pages.json", old)
        restarted = self.manager()
        self.assertEqual(restarted.page()["state"], "confirmation_required")
        self.assertEqual(restarted.page()["paper"]["width_mm"], 210)
        new = restarted.new_page(page["page_id"], True)
        self.assertEqual(new["paper"]["width_mm"], 297)
        self.assertEqual(new["working_rect_mm"], [8, 8, 289, 202])
        self.assertEqual(restarted.pages.book["pages"][0]["paper"]["width_mm"], 210)
        self.assertEqual(restarted.pages.book["pages"][0]["reservations"][0]["state"], "occupied")

    def test_preview_is_nonexecuting_and_does_not_reserve_or_mark_page_full(self):
        calls = []
        manager = self.manager(runner=lambda *args: calls.append(1))
        initial = manager.page()
        preview = manager.preview("네", 18, initial["page_id"])
        self.assertFalse(preview["kinematics_checked"])
        self.assertFalse(preview["motion_authorized"])
        self.assertEqual(manager.page(), initial)
        self.assertEqual(calls, [])
        self.assertEqual(preview["plan"]["paper"]["character_mm"], 18)

    def test_manual_reset_archives_old_positions_and_rejects_stale_confirmation(self):
        manager = self.manager(runner=lambda *args: good_report())
        first = self.finish(manager, manager.submit("네", 18))
        old = manager.page()
        with self.assertRaises(ValueError):
            manager.new_page(old["page_id"], False)
        new = manager.new_page(old["page_id"], True)
        self.assertNotEqual(old["page_id"], new["page_id"])
        self.assertEqual(new["next_row_y_mm"], new["working_rect_mm"][1])
        self.assertEqual(new["reservations"], [])
        with self.assertRaises(PaperSpaceError):
            manager.new_page(old["page_id"], True)
        with self.assertRaises(PaperSpaceError):
            manager.submit("네", 18, old["page_id"])
        self.assertEqual(manager.get(first["job_id"])["placement"]["page_id"], old["page_id"])
        book = json.loads((Path(self.directory.name) / "pages.json").read_text(encoding="utf-8"))
        self.assertEqual(book["pages"][0]["reservations"][0]["state"], "occupied")
        self.assertEqual(book["pages"][0]["state"], "replaced")

    def test_reset_while_running_is_blocked_without_losing_reservation(self):
        entered = threading.Event()
        def runner(plan, config, stop, progress):
            entered.set()
            while not stop():
                time.sleep(.002)
            return {**good_report(), "status": "stopped", "stop_settled": True, "stop_test_passed": True}
        manager = self.manager(runner=runner)
        job = manager.submit("네", 18)
        self.assertTrue(entered.wait(2))
        with self.assertRaises(JobBusyError):
            manager.new_page(manager.page()["page_id"], True)
        manager.stop(job["job_id"])
        result = self.finish(manager, job)
        self.assertEqual(result["page_reservation_state"], "partial")
        next_plan = manager.preview("한", 18)
        self.assertGreater(next_plan["placement"]["reserved_rect_mm"][1], result["placement"]["reserved_rect_mm"][3])

    def test_unequivocal_preflight_rejection_releases_unwritten_space(self):
        report = {**good_report(), "phase": "preflight", "status": "rejected", "trajectory_complete": False,
                  "dynamics_validated": False, "preflight": {"status": "rejected"}, "failure": {"reason": "preflight_not_passed"}}
        manager = self.manager(runner=lambda *args: deepcopy(report))
        before = manager.page()["next_row_y_mm"]
        result = self.finish(manager, manager.submit("네", 18))
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["page_reservation_state"], "released_no_motion")
        self.assertEqual(manager.page()["next_row_y_mm"], before)

    def test_unknown_failure_retains_region_and_blocks_until_new_sheet(self):
        manager = self.manager(runner=lambda *args: {**good_report(), "dynamics_validated": False})
        result = self.finish(manager, manager.submit("네", 18))
        self.assertEqual(result["state"], "uncertain")
        self.assertEqual(manager.page()["state"], "inspection_required")
        self.assertEqual(result["page_reservation_state"], "uncertain")
        with self.assertRaises(PaperSpaceError):
            manager.submit("한", 18)
        manager.new_page(manager.page()["page_id"], True)
        self.assertEqual(manager.page()["state"], "ready")

    def test_completed_occupancy_survives_manager_restart(self):
        manager = self.manager(runner=lambda *args: good_report())
        self.finish(manager, manager.submit("네", 18))
        original = manager.page()
        manager.close(); self.managers.remove(manager)
        restarted = self.manager(runner=lambda *args: good_report())
        self.assertEqual(restarted.page(), original)
        second = self.finish(restarted, restarted.submit("한", 18))
        self.assertGreater(second["placement"]["reserved_rect_mm"][1], original["reservations"][0]["reserved_rect_mm"][3])

    def test_stale_valid_ledger_cannot_silently_lose_a_completed_job_region(self):
        manager = self.manager(runner=lambda *args: good_report())
        before = deepcopy(manager.pages.book)
        self.finish(manager, manager.submit("네", 18))
        manager.close(); self.managers.remove(manager)
        atomic_json(Path(self.directory.name) / "pages.json", before)
        restarted = self.manager()
        self.assertEqual(restarted.page()["state"], "inspection_required")
        self.assertEqual(restarted.page()["reason"], "job_ledger_accounting_disagrees")
        with self.assertRaises(PaperSpaceError):
            restarted.submit("네", 18)
        restarted.new_page(restarted.page()["page_id"], True)
        restarted.close(); self.managers.remove(restarted)
        confirmed = self.manager()
        self.assertEqual(confirmed.page()["state"], "ready")

    def test_rolled_back_sheet_reset_requires_new_confirmation_before_reuse(self):
        manager = self.manager(runner=lambda *args: good_report())
        before = deepcopy(manager.pages.book)
        manager.new_page(manager.page()["page_id"], True)
        self.finish(manager, manager.submit("네", 18))
        manager.close(); self.managers.remove(manager)
        atomic_json(Path(self.directory.name) / "pages.json", before)
        restarted = self.manager()
        self.assertEqual(restarted.page()["state"], "inspection_required")
        self.assertEqual(restarted.page()["reason"], "job_ledger_accounting_disagrees")

    def test_orphan_reservation_after_crash_is_not_automatically_released(self):
        manager = self.manager(runner=lambda *args: good_report())
        initial = manager.page()
        _, placement = manager.pages.placement("네", 18)
        manager.pages.reserve("a" * 32, placement)
        manager.close(); self.managers.remove(manager)
        restarted = self.manager()
        self.assertEqual(restarted.page()["state"], "inspection_required")
        self.assertGreater(restarted.page()["next_row_y_mm"], initial["next_row_y_mm"])
        self.assertEqual(restarted.page()["reservations"][0]["state"], "uncertain")

    def test_changed_profile_requires_explicit_new_sheet_confirmation(self):
        manager = self.manager(runner=lambda *args: good_report())
        self.finish(manager, manager.submit("네", 18))
        manager.close(); self.managers.remove(manager)
        config = default_config()
        config["paper_frame"]["origin_m"] = (.195, -.040, .005)
        restarted = self.manager(config=config)
        self.assertEqual(restarted.page()["state"], "confirmation_required")
        with self.assertRaises(PaperSpaceError):
            restarted.submit("한", 18)
        restarted.new_page(restarted.page()["page_id"], True)
        self.assertEqual(restarted.page()["state"], "ready")

    def test_legacy_job_history_never_assumes_a_blank_sheet(self):
        manager = self.manager(runner=lambda *args: good_report())
        self.finish(manager, manager.submit("네", 18))
        manager.close(); self.managers.remove(manager)
        (Path(self.directory.name) / "pages.json").unlink()
        restarted = self.manager()
        self.assertEqual(restarted.page()["reason"], "legacy_jobs_have_no_paper_occupancy")
        self.assertEqual(restarted.page()["state"], "confirmation_required")

    def test_reservation_storage_failure_never_starts_runner(self):
        calls = []
        manager = self.manager(runner=lambda *args: calls.append(1))
        with patch("kwon_lab.writing.jobs.atomic_json", side_effect=OSError("disk failure")), patch.object(manager.pages, "save_json", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                manager.submit("네", 18)
        self.assertEqual(calls, [])
        self.assertTrue(manager.list()["storage_fault"])

    def test_corrupt_or_overlapping_ledger_is_not_accepted(self):
        manager = self.manager(runner=lambda *args: good_report())
        self.finish(manager, manager.submit("네", 18))
        manager.close(); self.managers.remove(manager)
        path = Path(self.directory.name) / "pages.json"
        book = json.loads(path.read_text(encoding="utf-8"))
        for field, value in (("overlap", None), ("config_signature", "invalid"), ("number", True), ("guard_mm", 1.0)):
            changed = deepcopy(book)
            if field == "overlap":
                duplicate = deepcopy(changed["pages"][0]["reservations"][0]); duplicate["job_id"] = "b" * 32
                changed["pages"][0]["reservations"].append(duplicate)
            else:
                changed["pages"][0][field] = value
            atomic_json(path, changed)
            with self.assertRaises(ValueError):
                JobManager(self.directory.name)

    def test_actual_mujoco_two_rows_pass_and_actual_trace_stays_in_reservations(self):
        manager = self.manager()
        jobs = [self.finish(manager, manager.submit("네", 18)) for _ in range(2)]
        for job in jobs:
            self.assertEqual(job["state"], "completed", job["failure"])
            report = json.loads(manager.artifact(job["job_id"], "report.json").read_text(encoding="utf-8"))
            from kwon_lab.writing.simulation import PaperFrame
            frame = PaperFrame(**report["paper_frame"])
            x0, y0, x1, y1 = job["placement"]["reserved_rect_mm"]
            ink = [frame.paper_mm(s["tip_world_m"]) for s in report["samples"] if s["op"] == "stroke"]
            self.assertTrue(ink)
            self.assertTrue(all(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in ink))
        self.assertLess(jobs[0]["placement"]["reserved_rect_mm"][3], jobs[1]["placement"]["reserved_rect_mm"][1])

    def test_actual_fixed_answer_rejection_then_explicit_smaller_size_reuses_unwritten_row(self):
        manager = self.manager(config=legacy_config())
        call = self.finish(manager, manager.submit("네", 18))
        self.assertEqual(call["state"], "completed")
        rejected = self.finish(manager, manager.submit("빨간색", 18))
        self.assertEqual(rejected["state"], "failed")
        self.assertEqual(rejected["failure"]["reason"], "reference_exceeds_actuator_control_range")
        self.assertEqual(rejected["page_reservation_state"], "released_no_motion")
        self.assertEqual(manager.page()["state"], "ready")
        answer = self.finish(manager, manager.submit("빨간색", 12))
        self.assertEqual(answer["state"], "completed", answer["failure"])
        self.assertEqual(answer["placement"]["page_id"], call["placement"]["page_id"])
        self.assertEqual(answer["placement"]["reserved_rect_mm"][1], rejected["placement"]["reserved_rect_mm"][1])
        self.assertLess(call["placement"]["reserved_rect_mm"][3], answer["placement"]["reserved_rect_mm"][1])

    def test_landscape_default_changes_only_virtual_frame_not_tool_or_guards(self):
        current, original = default_config(), legacy_config()
        self.assertEqual(current["paper_frame"]["origin_m"], (.160, -.070, .005))
        self.assertEqual(current["paper_frame"]["working_rect_mm"], (8., 8., 289., 202.))
        for key in ("tool", "settings", "transit", "dynamics"):
            self.assertEqual(current[key], original[key])
        for key in ("x_axis", "y_axis"):
            self.assertEqual(current["paper_frame"][key], original["paper_frame"][key])

    def test_actual_default_28mm_call_then_greeting_complete_in_distinct_rows(self):
        manager = self.manager()
        call = self.finish(manager, manager.submit("네", 28), seconds=90)
        greeting = self.finish(manager, manager.submit("안녕하세요", 28), seconds=90)
        for job in (call, greeting):
            self.assertEqual(job["state"], "completed", job["failure"])
            self.assertEqual(job["page_reservation_state"], "occupied")
            report = json.loads(manager.artifact(job["job_id"], "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["dynamics_validated"] and report["ready_return_validated"])
            self.assertLessEqual(report["max_tracking_error_mm"], 1.)
            self.assertFalse(report["pressure_validated"] or report["motion_authorized"])
        self.assertEqual(call["placement"]["page_id"], greeting["placement"]["page_id"])
        self.assertLess(call["placement"]["reserved_rect_mm"][3], greeting["placement"]["reserved_rect_mm"][1])
        self.assertEqual(greeting["placement"]["layout_rect_mm"][1], 53.25)


if __name__ == "__main__":
    unittest.main()
