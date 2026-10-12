from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from kwon_lab.writing.jobs import JobBusyError, JobManager, RecordOwnerError, atomic_json, default_config, legacy_config


def passed_report():
    return {"kind": "writing_dynamics_rollout", "scope": "full_cycle", "status": "passed",
            "trajectory_complete": True, "dynamics_validated": True, "ready_return_validated": True,
            "motion_authorized": False, "physical_calibration_required": True,
            "physical_dynamics_validated": False, "pressure_validated": False,
            "actual_simulated_duration_s": 1., "samples": [], "failure": None}


class WritingJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.managers = []

    def tearDown(self):
        for manager in reversed(self.managers):
            manager.close()
        self.temporary.cleanup()

    def manager(self, **kwargs):
        manager = JobManager(self.temporary.name, **kwargs)
        self.managers.append(manager)
        return manager

    def wait(self, manager, job_id, seconds=15):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if manager.list()["active_job_id"] is None:
                return manager.get(job_id)
            time.sleep(.01)
        self.fail("Worker did not finish")

    def test_success_persists_plan_settings_dates_events_and_unreviewed_status(self):
        manager = self.manager(runner=lambda *args: passed_report())
        submitted = manager.submit("네")
        result = self.wait(manager, submitted["job_id"])
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["completed_strokes"], result["total_strokes"])
        self.assertEqual(result["human_review"]["status"], "unreviewed")
        self.assertTrue(result["finished_at"].endswith("+09:00"))
        self.assertFalse(result["motion_authorized"])
        for name in ("job.json", "plan.json", "config.json", "report.json"):
            self.assertIsInstance(json.loads(manager.artifact(result["job_id"], name).read_text(encoding="utf-8")), dict)
        events = [json.loads(line) for line in manager.artifact(result["job_id"], "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([event["event"] for event in events], ["accepted", "checking", "verifying", "finished"])
        snapshot = manager.get(result["job_id"])
        snapshot["human_review"]["status"] = "pass"
        self.assertEqual(manager.get(result["job_id"])["human_review"]["status"], "unreviewed")

    def test_simultaneous_submissions_and_process_ownership_are_exclusive(self):
        release = threading.Event()
        def runner(*args):
            release.wait(5)
            return passed_report()
        manager = self.manager(runner=runner)
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(manager.submit, "네") for _ in range(6)]
                accepted, busy = [], 0
                for future in futures:
                    try:
                        accepted.append(future.result())
                    except JobBusyError:
                        busy += 1
            self.assertEqual((len(accepted), busy), (1, 5))
            with self.assertRaises(RecordOwnerError):
                JobManager(self.temporary.name)
            script = "from kwon_lab.writing.jobs import JobManager,RecordOwnerError; import sys\ntry: JobManager(sys.argv[1])\nexcept RecordOwnerError: sys.exit(23)"
            completed = subprocess.run([sys.executable, "-c", script, self.temporary.name], capture_output=True, timeout=15)
            self.assertEqual(completed.returncode, 23, completed.stderr)
        finally:
            release.set()
        self.wait(manager, accepted[0]["job_id"])

    def test_stop_token_prevents_runner_from_sending_further_segments(self):
        entered, ticks = threading.Event(), []
        def runner(plan, config, stop, progress):
            progress({"phase": "writing", "simulation_steps": 1})
            entered.set()
            while not stop():
                ticks.append(1)
                time.sleep(.005)
            count = len(ticks)
            time.sleep(.02)
            self.assertEqual(len(ticks), count)
            report = passed_report()
            report.update(status="stopped", trajectory_complete=False, dynamics_validated=False,
                          stop_settled=True, stop_test_passed=True)
            return report
        manager = self.manager(runner=runner)
        job = manager.submit("네")
        self.assertTrue(entered.wait(2))
        self.assertEqual(manager.stop(job["job_id"])["state"], "stopping")
        result = self.wait(manager, job["job_id"])
        self.assertEqual(result["state"], "stopped")
        self.assertTrue(result["stop_requested"])
        self.assertTrue(result["stop_settled"])
        self.assertEqual(result["completed_strokes"], 0)
        with self.assertRaises(JobBusyError):
            manager.stop(job["job_id"])

    def test_failed_rejected_unknown_and_missing_verification_are_distinct(self):
        for status, expected in (("rejected", "failed"), ("unknown", "uncertain"), ("passed", "uncertain")):
            report = passed_report()
            report.update(status=status, dynamics_validated=False, failure={"reason": "probe"})
            manager = self.manager(runner=lambda *args, r=report: deepcopy(r))
            manager.new_page(manager.page()["page_id"], True)
            result = self.wait(manager, manager.submit("네")["job_id"])
            self.assertEqual(result["state"], expected)
            manager.close()
            self.managers.remove(manager)

    def test_backend_exception_is_failed_not_completed(self):
        def runner(*args):
            raise RuntimeError("probe backend error")
        manager = self.manager(runner=runner)
        result = self.wait(manager, manager.submit("네")["job_id"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["failure"]["reason"], "execution_or_record_error")

    def test_timeout_stops_backend_and_records_failure_instead_of_normal_stop(self):
        def runner(plan, config, stop, progress):
            while not stop():
                time.sleep(.005)
            return {**passed_report(), "status": "stopped", "stop_settled": True}
        manager = self.manager(runner=runner, timeout_s=.03)
        result = self.wait(manager, manager.submit("네")["job_id"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["failure"]["reason"], "job_wall_timeout")

    def test_late_stop_cannot_be_reported_as_verified_completion(self):
        entered, release = threading.Event(), threading.Event()
        def runner(*args):
            entered.set()
            release.wait(3)
            return passed_report()
        manager = self.manager(runner=runner)
        job = manager.submit("네")
        entered.wait(2)
        manager.stop(job["job_id"])
        release.set()
        result = self.wait(manager, job["job_id"])
        self.assertEqual(result["state"], "uncertain")
        self.assertFalse(result["stop_settled"])

    def test_restart_marks_active_records_interrupted_without_resuming(self):
        manager = self.manager(runner=lambda *args: passed_report())
        job = self.wait(manager, manager.submit("네")["job_id"])
        folder = manager.artifact(job["job_id"], "job.json").parent
        manager.close()
        self.managers.remove(manager)
        job["state"] = "executing"
        atomic_json(folder / "job.json", job)
        calls = []
        recovered = self.manager(runner=lambda *args: calls.append(1))
        result = recovered.get(job["job_id"])
        self.assertEqual(result["state"], "interrupted")
        self.assertEqual(result["verification"], "uncertain")
        self.assertEqual(calls, [])
        self.assertIsNone(recovered.list()["active_job_id"])

    def test_review_is_explicit_screen_only_and_artifact_paths_are_allowlisted(self):
        manager = self.manager(runner=lambda *args: passed_report())
        job = self.wait(manager, manager.submit("네")["job_id"])
        reviewed = manager.review(job["job_id"], "uncertain", "네")
        self.assertEqual(reviewed["human_review"]["scope"], "simulation_trace_not_physical")
        self.assertFalse(reviewed["pressure_validated"])
        self.assertTrue(manager.artifact(job["job_id"], "review.json").is_file())
        for name in ("../config.json", ".owner.lock", "missing.json"):
            with self.assertRaises(KeyError):
                manager.artifact(job["job_id"], name)
        for status, readback in (([], ""), ("pass", "a" * 121), ("unreviewed", "")):
            with self.assertRaises(ValueError):
                manager.review(job["job_id"], status, readback)

    def test_input_and_unsafe_config_do_not_create_jobs(self):
        manager = self.manager(runner=lambda *args: passed_report())
        for text, mm in (("네", True), ("네", float("nan")), ("네", 7), ("네", 41), ("abc", 28), ([], 28)):
            with self.assertRaises(ValueError):
                manager.submit(text, mm)
        self.assertEqual(manager.list()["jobs"], [])
        config = default_config()
        config["tool"]["calibrated"] = True
        with self.assertRaises(ValueError):
            JobManager(Path(self.temporary.name) / "bad", config=config)

    def test_storage_failure_blocks_additional_execution(self):
        manager = self.manager(runner=lambda *args: passed_report())
        with patch("kwon_lab.writing.jobs.atomic_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.submit("네")
        self.assertTrue(manager.list()["storage_fault"])
        with self.assertRaises(RuntimeError):
            manager.submit("네")

    def test_actual_mujoco_runner_persists_actual_trace_and_rejected_plan(self):
        manager = self.manager(config=legacy_config())
        job = self.wait(manager, manager.submit("네")["job_id"])
        self.assertEqual(job["state"], "completed")
        from PIL import Image
        with Image.open(manager.artifact(job["job_id"], "trace.png")) as image:
            self.assertEqual(image.size, (1188, 840))
        report = json.loads(manager.artifact(job["job_id"], "report.json").read_text(encoding="utf-8"))
        self.assertGreater(len(report["samples"]), 1000)
        self.assertGreater(report["max_tracking_error_mm"], 0)
        rejected = self.wait(manager, manager.submit("안녕하세요.", 18)["job_id"])
        self.assertEqual(rejected["state"], "failed")
        self.assertEqual(rejected["completed_strokes"], 0)
        self.assertNotIn("trace.png", rejected["artifacts"])

    def test_actual_mujoco_stop_settles_and_keeps_partial_trace(self):
        manager = self.manager()
        job = manager.submit("네")
        end = time.monotonic() + 10
        while manager.get(job["job_id"])["progress"].get("phase") != "writing":
            self.assertLess(time.monotonic(), end)
            time.sleep(.001)
        manager.stop(job["job_id"])
        result = self.wait(manager, job["job_id"])
        self.assertEqual(result["state"], "stopped")
        self.assertTrue(result["stop_settled"])
        self.assertTrue(result["stop_test_passed"])
        self.assertLess(result["completed_strokes"], result["total_strokes"])
        self.assertIn("trace.png", result["artifacts"])

    def test_image_failure_cannot_be_reported_as_job_completion(self):
        report = passed_report()
        report["samples"] = [{"stroke": 0}]
        def bad_image(*args):
            raise RuntimeError("render probe")
        manager = self.manager(runner=lambda *args: report, trace_writer=bad_image)
        result = self.wait(manager, manager.submit("네")["job_id"])
        self.assertEqual(result["state"], "uncertain")

    def test_physical_authorization_contract_is_rejected(self):
        manager = self.manager()
        for key, value in (("motion_authorized", True), ("failure", {"reason": "inconsistent_success"}), ("ready_return_validated", False)):
            with self.subTest(key=key):
                report = passed_report()
                report[key] = value
                manager.runner = lambda *args, r=report: r
                result = self.wait(manager, manager.submit("네")["job_id"])
                self.assertEqual(result["state"], "uncertain")
                manager.new_page(manager.page()["page_id"], True)

    def test_corrupt_history_fails_closed_and_releases_record_lease(self):
        folder = Path(self.temporary.name) / ("a" * 32)
        folder.mkdir()
        for content in ("invalid", "[]"):
            (folder / "job.json").write_text(content, encoding="utf-8")
            for _ in range(2):
                with self.assertRaises(ValueError):
                    JobManager(self.temporary.name)

    def test_nonfinite_backend_report_does_not_poison_job_state_or_active_lock(self):
        report = passed_report()
        report["actual_simulated_duration_s"] = float("nan")
        manager = self.manager(runner=lambda *args: report)
        result = self.wait(manager, manager.submit("네")["job_id"])
        self.assertEqual(result["state"], "uncertain")
        self.assertIsNone(result["simulation_time_s"])
        self.assertIsNone(manager.list()["active_job_id"])
        def invalid_progress(plan, config, stop, progress):
            progress({"phase": "writing", "simulation_time_s": float("nan")})
        manager.new_page(manager.page()["page_id"], True)
        manager.runner = invalid_progress
        result = self.wait(manager, manager.submit("네")["job_id"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["progress"], {})
        self.assertIsNone(manager.list()["active_job_id"])

    def test_close_records_shutdown_stop_before_releasing_ownership(self):
        entered = threading.Event()
        def runner(plan, config, stop, progress):
            entered.set()
            while not stop():
                time.sleep(.005)
            return {**passed_report(), "status": "stopped", "trajectory_complete": False, "stop_settled": True}
        manager = self.manager(runner=runner)
        job = manager.submit("네")
        self.assertTrue(entered.wait(2))
        manager.close()
        self.managers.remove(manager)
        result = manager.get(job["job_id"])
        self.assertEqual(result["state"], "stopped")
        self.assertTrue(result["stop_requested"])
        self.assertEqual(result["stop_cause"], "server_shutdown")


if __name__ == "__main__":
    unittest.main()
