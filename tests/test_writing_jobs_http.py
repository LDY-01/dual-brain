from copy import deepcopy
import json
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from kwon_lab.writing.jobs import JobManager
from kwon_lab.writing.web_server import make_server


class JobHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.entered, self.release = threading.Event(), threading.Event()
        def runner(plan, config, stop, progress):
            progress({"phase": "writing", "simulation_steps": 1})
            self.entered.set()
            while not self.release.is_set() and not stop():
                time.sleep(.005)
            stopped = stop()
            return {"kind": "writing_dynamics_rollout", "scope": "full_cycle", "status": "stopped" if stopped else "passed",
                    "trajectory_complete": not stopped, "dynamics_validated": not stopped, "ready_return_validated": not stopped,
                    "motion_authorized": False, "physical_calibration_required": True,
                    "physical_dynamics_validated": False, "pressure_validated": False,
                    "samples": [], "failure": None, "stop_settled": stopped, "stop_test_passed": stopped}
        self.manager = JobManager(self.directory.name, runner=runner)
        self.server = make_server(0, manager=self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def request(self, path, payload=None, headers=None, raw=None):
        body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        request = Request(self.url + path, data=body, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            content = response.read()
            return response.status, json.loads(content) if "json" in response.headers.get("Content-Type", "") else content

    def finish(self, job_id):
        end = time.monotonic() + 3
        while self.manager.list()["active_job_id"]:
            self.assertLess(time.monotonic(), end)
            time.sleep(.01)
        return self.manager.get(job_id)

    def test_start_list_stop_and_cross_client_conflict(self):
        status, job = self.request("/api/jobs", {"text": "네"})
        self.assertEqual(status, 202)
        self.assertTrue(self.entered.wait(2))
        self.assertEqual(self.request("/api/jobs", {"text": "한"})[0], 409)
        status, listed = self.request("/api/jobs")
        self.assertEqual(status, 200)
        self.assertEqual(listed["active_job_id"], job["job_id"])
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/stop", {})[0], 202)
        self.assertEqual(self.finish(job["job_id"])["state"], "stopped")
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/stop", {})[0], 409)

    def test_reload_history_explicit_review_and_artifact_download(self):
        self.release.set()
        _, job = self.request("/api/jobs", {"text": "네", "character_mm": 24})
        result = self.finish(job["job_id"])
        self.assertEqual(result["state"], "completed")
        status, review = self.request(f"/api/jobs/{job['job_id']}/review", {"status": "uncertain", "readback": "네"})
        self.assertEqual(status, 200)
        self.assertEqual(review["human_review"]["scope"], "simulation_trace_not_physical")
        status, plan = self.request(f"/api/jobs/{job['job_id']}/artifacts/plan.json")
        self.assertEqual(status, 200)
        self.assertEqual(plan["paper"]["character_mm"], 24)
        self.assertFalse(plan["motion_authorized"])

    def test_foreign_origin_host_and_unknown_artifacts_are_blocked(self):
        self.assertEqual(self.request("/api/jobs", headers={"Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.request("/api/jobs", {"text": "네"}, headers={"Host": "example.com"})[0], 403)
        self.release.set()
        _, job = self.request("/api/jobs", {"text": "네"})
        self.finish(job["job_id"])
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/artifacts/.owner.lock")[0], 404)
        self.assertEqual(self.request("/api/jobs/not-an-id")[0], 404)
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/artifacts/%2e%2e")[0], 404)

    def test_bad_payloads_and_nonfinite_constants_are_not_jobs(self):
        for payload in ({"text": "네", "calibrated": True}, {"text": "네", "character_mm": True}, {"text": "abc"}, []):
            self.assertEqual(self.request("/api/jobs", payload)[0], 400)
        self.assertEqual(self.request("/api/jobs", raw=b'{"text":"x", "character_mm":NaN}')[0], 400)
        self.assertEqual(self.request("/api/jobs", {"text": "네"}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.manager.list()["jobs"], [])

    def test_review_while_active_and_invalid_review_are_rejected(self):
        _, job = self.request("/api/jobs", {"text": "네"})
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/review", {"status": "pass"})[0], 409)
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/review", {"status": []})[0], 400)
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/stop", {"resume": True})[0], 400)

    def test_dashboard_assets_and_original_call_api_remain_available(self):
        for path in ("/jobs", "/jobs.css", "/jobs.js"):
            self.assertEqual(self.request(path)[0], 200)
        status, call = self.request("/api/call", {"input": "야"})
        self.assertEqual(status, 200)
        self.assertEqual(call["response"], "네")

    def test_page_preview_is_read_only_and_reset_requires_explicit_confirmation(self):
        status, page = self.request("/api/page")
        self.assertEqual(status, 200)
        status, preview = self.request("/api/page/preview", {"text": "네", "character_mm": 18, "page_id": page["page_id"]})
        self.assertEqual(status, 200)
        self.assertFalse(preview["kinematics_checked"])
        self.assertFalse(preview["motion_authorized"])
        self.assertEqual(self.request("/api/page")[1], page)
        for confirmed in (False, 1, "yes"):
            self.assertEqual(self.request("/api/page/reset", {"page_id": page["page_id"], "confirmed": confirmed})[0], 400)
        status, new = self.request("/api/page/reset", {"page_id": page["page_id"], "confirmed": True})
        self.assertEqual(status, 200)
        self.assertNotEqual(page["page_id"], new["page_id"])
        self.assertEqual(self.request("/api/page/reset", {"page_id": page["page_id"], "confirmed": True})[0], 409)
        self.assertEqual(self.request("/api/jobs", {"text": "네", "page_id": page["page_id"]})[0], 409)

    def test_running_reset_and_foreign_page_requests_are_blocked(self):
        page = self.request("/api/page")[1]
        _, job = self.request("/api/jobs", {"text": "네", "character_mm": 18, "page_id": page["page_id"]})
        self.assertTrue(self.entered.wait(2))
        self.assertEqual(self.request("/api/page/reset", {"page_id": page["page_id"], "confirmed": True})[0], 409)
        self.assertEqual(self.request("/api/page", headers={"Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.request("/api/page/reset", {"confirmed": True}, headers={"Host": "example.com"})[0], 403)
        self.request(f"/api/jobs/{job['job_id']}/stop", {})
        self.finish(job["job_id"])

    def test_two_rows_full_sheet_and_new_sheet_reset_use_separate_page_ids(self):
        self.release.set()
        page = self.request("/api/page")[1]
        first = self.request("/api/jobs", {"text": "네", "character_mm": 18, "page_id": page["page_id"]})[1]
        self.finish(first["job_id"])
        second = self.request("/api/jobs", {"text": "한", "character_mm": 18, "page_id": page["page_id"]})[1]
        self.finish(second["job_id"])
        self.assertLess(first["placement"]["reserved_rect_mm"][3], second["placement"]["reserved_rect_mm"][1])
        for _ in range(4):
            _, extra = self.request("/api/jobs", {"text": "글", "character_mm": 18, "page_id": page["page_id"]})
            self.finish(extra["job_id"])
        status, blocked = self.request("/api/jobs", {"text": "글", "character_mm": 18, "page_id": page["page_id"]})
        self.assertEqual(status, 409)
        self.assertEqual(blocked["reason"], "insufficient_remaining_space")
        self.assertEqual(blocked["category"], "paper_space")
        self.assertEqual(blocked["page"]["state"], "awaiting_new_sheet")
        self.assertEqual(len(self.manager.list()["jobs"]), 6)
        _, new = self.request("/api/page/reset", {"page_id": page["page_id"], "confirmed": True})
        status, third = self.request("/api/jobs", {"text": "글", "character_mm": 18, "page_id": new["page_id"]})
        self.assertEqual(status, 202)
        self.assertEqual(third["placement"]["reserved_rect_mm"][1], first["placement"]["reserved_rect_mm"][1])

    def test_landscape_blank_paper_overflow_has_paper_space_category(self):
        status, page = self.request("/api/page")
        self.assertEqual((page["paper"]["width_mm"], page["paper"]["height_mm"]), (297, 210))
        status, failure = self.request("/api/jobs", {"text": "가" * 120, "character_mm": 40})
        self.assertEqual(status, 409)
        self.assertEqual(failure["category"], "paper_space")
        self.assertEqual(failure["reason"], "text_does_not_fit_blank_sheet")
        self.assertEqual(self.manager.page()["state"], "ready")

    def test_bad_page_payloads_cannot_override_work_area_or_physical_authorization(self):
        for path, payload in (("/api/page/preview", {"text": "네", "motion_authorized": True}),
                              ("/api/page/reset", {"confirmed": True, "working_rect_mm": [0, 0, 210, 297]}),
                              ("/api/jobs", {"text": "네", "page_id": []})):
            self.assertEqual(self.request(path, payload)[0], 400)
        self.assertEqual(self.request("/api/jobs")[1]["jobs"], [])


if __name__ == "__main__":
    unittest.main()
