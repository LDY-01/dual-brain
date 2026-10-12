"""Single-owner, durable simulation jobs. No robot backend or automatic recovery."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

from .planner import PaperSettings
from .paper import PageLedger, PaperSpaceError, MIN_CHARACTER_MM, MAX_CHARACTER_MM


ACTIVE = {"planning", "checking", "executing", "stopping", "verifying"}
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
ARTIFACTS = {"job.json", "plan.json", "config.json", "report.json", "events.jsonl", "trace.png", "review.json", "placement.json"}
DEFAULT_RECORDS = Path(__file__).resolve().parents[2] / "outputs" / "writing" / "jobs"


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(7):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 6:
                    raise
                # Windows readers/sync software can briefly lock a JSON destination.
                time.sleep(.02 * 2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)


class JobBusyError(RuntimeError):
    pass


class RecordOwnerError(RuntimeError):
    pass


class RecordLease:
    """OS file lock survives neither process death nor explicit close; never PID-kill."""

    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.stream = (root / ".owner.lock").open("a+b")
        if self.stream.seek(0, 2) == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RecordOwnerError("Another job server owns this record directory") from error

    def close(self):
        self.stream.close()


def legacy_config():
    from .dynamics import DynamicsSettings
    from .simulation import ToolMount, PaperFrame, SimulationSettings, TransitSettings
    return {"tool": asdict(ToolMount()), "paper_frame": asdict(PaperFrame()),
            "settings": asdict(SimulationSettings()), "transit": asdict(TransitSettings()),
            "dynamics": asdict(DynamicsSettings())}


def default_config():
    """Virtual landscape profile verified for 28 mm call followed by greeting."""
    config = legacy_config()
    config["paper_frame"].update(origin_m=(.160, -.070, .005), working_rect_mm=(8., 8., 289., 202.))
    return config


def validate_config(config):
    from .dynamics import DynamicsSettings
    from .simulation import ToolMount, PaperFrame, SimulationSettings, TransitSettings
    if not isinstance(config, dict) or set(config) != {"tool", "paper_frame", "settings", "transit", "dynamics"}:
        raise ValueError("A complete simulation-only configuration is required")
    tool, frame = ToolMount(**config["tool"]), PaperFrame(**config["paper_frame"])
    settings, transit = SimulationSettings(**config["settings"]), TransitSettings(**config["transit"])
    dynamics = DynamicsSettings(**config["dynamics"])
    tool.validate()
    paper = PaperSettings(width_mm=297, height_mm=210)
    frame.validate(paper)
    settings.validate()
    transit.validate(paper, settings)
    dynamics.validate()


def simulation_runner(plan, config, should_stop, progress):
    from .dynamics import DynamicsSettings, run_dynamics
    from .simulation import ToolMount, PaperFrame, SimulationSettings, TransitSettings
    report, _ = run_dynamics(plan, ToolMount(**config["tool"]), PaperFrame(**config["paper_frame"]),
                             SimulationSettings(**config["settings"]), TransitSettings(**config["transit"]),
                             DynamicsSettings(**config["dynamics"]), stop_requested=should_stop, progress=progress)
    return report


def save_trace(plan, report, path):
    """Project recorded actual TCP segments, not the planned glyph or an ink model."""
    from PIL import Image, ImageDraw
    from .simulation import PaperFrame
    frame = PaperFrame(**report["paper_frame"])
    scale = 4
    image = Image.new("RGB", (round(plan.paper.width_mm * scale), round(plan.paper.height_mm * scale)), "white")
    draw = ImageDraw.Draw(image)
    margin = plan.paper.margin_mm * scale
    draw.rectangle((margin, margin, image.width - margin, image.height - margin), outline="#d9e2dd")
    for a, b in zip(report["samples"], report["samples"][1:]):
        if b["op"] == "stroke" and a["stroke"] == b["stroke"]:
            points = [tuple(frame.paper_mm(s["tip_world_m"])[:2] * scale) for s in (a, b)]
            draw.line(points, fill="#202828", width=2)
    image.save(path)


def completed_strokes(plan, report):
    if report.get("trajectory_complete") is True and report.get("status") == "passed":
        return len(plan.strokes)
    tolerance = report.get("path_tolerance_mm", 1.75)
    return len({s["stroke"] for s in report.get("samples", [])
                if s.get("op") == "pen_up" and s.get("point_mm", [0, 0, 0])[2] >= plan.paper.pen_lift_mm - tolerance})


def failure_category(failure):
    if not isinstance(failure, dict):
        return None
    reason = failure.get("reason", "")
    if reason in {"execution_or_record_error", "record_persistence_failed", "job_wall_timeout",
                  "result_contract_not_verified", "stop_request_not_observed_before_backend_completion",
                  "server_restarted_no_automatic_resume", "worker_start_failed"}:
        return "execution_record"
    if reason == "preflight_not_passed":
        return failure_category(failure.get("detail")) or "robot_validation"
    if reason in {"outside_candidate_work_area", "ik_not_converged", "reference_exceeds_actuator_control_range",
                  "tip_outside_work_area", "interpolated_tip_outside_work_area", "actual_tip_outside_work_area"}:
        return "robot_reach"
    return "robot_validation" if reason else "execution_record"


class JobManager:
    def __init__(self, records=DEFAULT_RECORDS, runner=None, config=None, timeout_s=120., trace_writer=None):
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 600:
            raise ValueError("timeout_s must be finite and in (0, 600]")
        self.config = deepcopy(config if config is not None else default_config())
        validate_config(self.config)
        self.root = Path(records).resolve()
        self.lease = RecordLease(self.root)
        self.lock = threading.RLock()
        self.jobs, self.active, self.thread = {}, None, None
        self.stop_event = threading.Event()
        self.runner, self.trace_writer = runner or simulation_runner, trace_writer or save_trace
        self.timeout_s, self.closed, self.storage_fault = timeout_s, False, False
        try:
            interrupted = []
            for folder in sorted(self.root.iterdir()):
                if not folder.is_dir() or not JOB_ID.fullmatch(folder.name):
                    continue
                if folder.is_symlink() or folder.resolve().parent != self.root:
                    raise ValueError("Job directories must remain inside the record root")
                job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
                if not isinstance(job, dict) or job.get("job_id") != folder.name or job.get("mode") != "simulation_only" or job.get("motion_authorized") is not False:
                    raise ValueError("Invalid stored job identity or safety contract")
                if job.get("state") not in ACTIVE | {"completed", "failed", "stopped", "uncertain", "interrupted"}:
                    raise ValueError("Invalid stored job state")
                self.jobs[folder.name] = job
                job.setdefault("failure_category", failure_category(job.get("failure")))
                if job["state"] in ACTIVE:
                    interrupted.append(folder.name)
                    job.update(state="interrupted", finished_at=now(), verification="uncertain",
                               failure={"reason": "server_restarted_no_automatic_resume"}, stop_settled=False,
                               failure_category="execution_record")
                    self._save(job, "recovered_as_interrupted")
            self.pages = PageLedger(self.root, self.config, atomic_json, legacy_history=bool(self.jobs), interrupted_jobs=interrupted)
            self.pages.reconcile(self.jobs.values())
        except BaseException:
            self.lease.close()
            raise

    def _folder(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id) or job_id not in self.jobs:
            raise KeyError("Unknown job")
        folder = self.root / job_id
        if folder.resolve().parent != self.root or folder.is_symlink():
            raise KeyError("Job directory escaped the record root")
        return folder

    def _save(self, job, event):
        folder = self._folder(job["job_id"])
        atomic_json(folder / "job.json", job)
        with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), "event": event, "state": job["state"], "phase": job.get("progress", {}).get("phase")}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def submit(self, text, character_mm=28, page_id=None):
        if isinstance(character_mm, bool) or not isinstance(character_mm, (int, float)) or not math.isfinite(character_mm) or not MIN_CHARACTER_MM <= character_mm <= MAX_CHARACTER_MM:
            raise ValueError("character_mm must be finite and in [8, 40]")
        if page_id is not None and (not isinstance(page_id, str) or not JOB_ID.fullmatch(page_id)):
            raise ValueError("Invalid page_id")
        with self.lock:
            if self.closed or self.storage_fault:
                raise RuntimeError("Job manager is closed or storage is unavailable")
            if self.active is not None:
                raise JobBusyError("Another simulation job is active")
            if page_id is not None and page_id != self.pages.current["page_id"]:
                raise PaperSpaceError("stale_sheet_submission")
            try:
                plan, placement = self.pages.placement(text, character_mm)
            except PaperSpaceError as error:
                if error.reason == "insufficient_remaining_space":
                    try:
                        self.pages.mark_full()
                    except OSError:
                        self.storage_fault = True
                        raise
                raise
            job_id = uuid4().hex
            folder = self.root / job_id
            folder.mkdir()
            job = {"format_version": 1, "kind": "writing_execution_job", "job_id": job_id,
                   "created_at": now(), "started_at": None, "finished_at": None, "state": "planning",
                   "mode": "simulation_only", "motion_authorized": False, "physical_calibration_required": True,
                   "physical_dynamics_validated": False, "pressure_validated": False,
                   "text": plan.text, "paper": asdict(plan.paper), "total_strokes": len(plan.strokes),
                   "completed_strokes": 0, "stroke_count_scope": "virtual_path_segments_not_ink_legibility",
                   "verification": "pending", "human_review": {"status": "unreviewed", "scope": "simulation_trace_not_physical"},
                   "stop_requested": False, "stop_settled": False, "failure": None,
                   "failure_category": None,
                   "wall_elapsed_s": None, "simulation_time_s": None, "progress": {},
                   "placement": placement, "page_reservation_state": "reserved",
                   "artifacts": ["job.json", "plan.json", "config.json", "events.jsonl", "placement.json"], "timeout_s": self.timeout_s}
            self.jobs[job_id] = job
            try:
                self.pages.reserve(job_id, placement)
                atomic_json(folder / "plan.json", plan.to_dict())
                atomic_json(folder / "config.json", self.config)
                atomic_json(folder / "placement.json", placement)
                self._save(job, "accepted")
            except OSError:
                self.storage_fault = True
                raise
            self.active = job_id
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._run, args=(job_id, plan), name=f"writing-{job_id[:8]}", daemon=False)
            try:
                self.thread.start()
            except RuntimeError:
                self.active = None
                job.update(state="failed", failure={"reason": "worker_start_failed"}, finished_at=now())
                self._save(job, "failed")
                raise
            return deepcopy(job)

    def _run(self, job_id, plan):
        started, timed_out, report = time.monotonic(), False, None
        job = self.jobs[job_id]

        def should_stop():
            nonlocal timed_out
            timed_out = time.monotonic() - started >= self.timeout_s
            return self.stop_event.is_set() or timed_out

        def progress(update):
            json.dumps(update, allow_nan=False)
            with self.lock:
                job["progress"] = dict(update)
                if job["state"] != "stopping":
                    job["state"] = "executing"
                # Persist phase boundaries, not every physics tick.
                if update.get("phase") != progress.last_phase:
                    self._save(job, "phase_changed")
                    progress.last_phase = update.get("phase")
        progress.last_phase = None
        try:
            with self.lock:
                job.update(started_at=now(), state="stopping" if self.stop_event.is_set() else "checking")
                self._save(job, "checking")
            report = self.runner(plan, deepcopy(self.config), should_stop, progress)
            json.dumps(report, allow_nan=False)
            with self.lock:
                job.update(state="verifying", simulation_time_s=report.get("actual_simulated_duration_s"))
                self._save(job, "verifying")
            folder = self._folder(job_id)
            atomic_json(folder / "report.json", report)
            artifacts = [*job["artifacts"], "report.json"]
            with self.lock:
                job["artifacts"] = artifacts
            if report.get("samples"):
                self.trace_writer(plan, report, folder / "trace.png")
                artifacts.append("trace.png")
            passed = (report.get("kind") == "writing_dynamics_rollout" and report.get("scope") == "full_cycle"
                      and report.get("status") == "passed" and report.get("trajectory_complete") is True
                      and report.get("dynamics_validated") is True and report.get("ready_return_validated") is True
                      and report.get("failure") is None
                      and report.get("motion_authorized") is False and report.get("physical_calibration_required") is True
                      and report.get("physical_dynamics_validated") is False and report.get("pressure_validated") is False)
            state = {"rejected": "failed", "stopped": "stopped"}.get(report.get("status"), "uncertain")
            if passed and not self.stop_event.is_set():
                state = "completed"
            failure = report.get("failure")
            if timed_out:
                state, failure = "failed", {"reason": "job_wall_timeout", "backend_failure": failure}
            elif passed and self.stop_event.is_set():
                state, failure = "uncertain", {"reason": "stop_request_not_observed_before_backend_completion"}
            elif state == "uncertain":
                failure = {"reason": "result_contract_not_verified", "backend_failure": failure}
            with self.lock:
                job.update(state=state, verification="passed_virtual_path" if state == "completed" else state,
                           failure=failure, artifacts=artifacts, completed_strokes=completed_strokes(plan, report),
                           stop_settled=report.get("stop_settled") is True, stop_test_passed=report.get("stop_test_passed") is True)
        except Exception as error:
            with self.lock:
                job.update(state="uncertain" if report and report.get("status") == "passed" else "failed",
                           verification="uncertain", failure={"reason": "execution_or_record_error", "message": str(error)[:500]})
                if isinstance(error, OSError):
                    self.storage_fault = True
        finally:
            with self.lock:
                job.update(finished_at=now(), wall_elapsed_s=round(time.monotonic() - started, 6))
                job["failure_category"] = failure_category(job.get("failure"))
                try:
                    job["page_reservation_state"] = self.pages.finish(job_id, job["state"], report)
                    self._save(job, "finished")
                except (OSError, ValueError, KeyError, StopIteration):
                    self.storage_fault = True
                    job.update(state="uncertain", verification="uncertain", failure={"reason": "record_persistence_failed"},
                               failure_category="execution_record")
                self.active = None

    def stop(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            self._folder(job_id)
            if self.active != job_id or job["state"] not in {"planning", "checking", "executing", "stopping"}:
                raise JobBusyError("This job is no longer stoppable")
            self.stop_event.set()
            job.update(stop_requested=True, state="stopping", stop_requested_at=now())
            try:
                self._save(job, "stop_requested")
            except OSError:
                self.storage_fault = True
                raise
            return deepcopy(job)

    def list(self):
        with self.lock:
            return {"active_job_id": self.active, "storage_fault": self.storage_fault,
                    "page": self.pages.snapshot(), "jobs": deepcopy(sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True))}

    def page(self):
        with self.lock:
            return self.pages.snapshot()

    def preview(self, text, character_mm=28, page_id=None):
        if isinstance(character_mm, bool) or not isinstance(character_mm, (int, float)) or not math.isfinite(character_mm) or not MIN_CHARACTER_MM <= character_mm <= MAX_CHARACTER_MM:
            raise ValueError("character_mm must be finite and in [8, 40]")
        if page_id is not None and (not isinstance(page_id, str) or not JOB_ID.fullmatch(page_id)):
            raise ValueError("Invalid page_id")
        with self.lock:
            if page_id is not None and page_id != self.pages.current["page_id"]:
                raise PaperSpaceError("stale_sheet_preview")
            plan, placement = self.pages.placement(text, character_mm)
            return {"placement": placement, "plan": plan.to_dict(), "kinematics_checked": False,
                    "mode": "simulation_only", "motion_authorized": False}

    def new_page(self, page_id, confirmed=False):
        if confirmed is not True:
            raise ValueError("Explicit new sheet confirmation is required")
        with self.lock:
            if self.active is not None:
                raise JobBusyError("Cannot replace the sheet while a job is active")
            if self.closed or self.storage_fault:
                raise RuntimeError("Job manager is closed or storage is unavailable")
            try:
                return self.pages.reset(page_id, self.jobs.keys())
            except OSError:
                self.storage_fault = True
                raise

    def get(self, job_id):
        with self.lock:
            self._folder(job_id)
            return deepcopy(self.jobs[job_id])

    def artifact(self, job_id, name):
        with self.lock:
            folder = self._folder(job_id)
            if name not in ARTIFACTS or name not in self.jobs[job_id]["artifacts"]:
                raise KeyError("Unknown artifact")
            path = folder / name
            if path.is_symlink() or path.resolve().parent != folder.resolve():
                raise KeyError("Invalid artifact path")
            return path

    def review(self, job_id, status, readback=""):
        if not isinstance(status, str) or status not in {"pass", "fail", "uncertain"} or not isinstance(readback, str) or len(readback) > 120:
            raise ValueError("Invalid review status or readback")
        with self.lock:
            folder = self._folder(job_id)
            job = self.jobs[job_id]
            if job["state"] in ACTIVE:
                raise JobBusyError("Review is available only after execution")
            review = {"status": status, "readback": readback, "recorded_at": now(), "scope": "simulation_trace_not_physical"}
            atomic_json(folder / "review.json", review)
            job["human_review"] = review
            job["artifacts"] = sorted(set(job["artifacts"]) | {"review.json"})
            self._save(job, "human_review_recorded")
            return deepcopy(job)

    def close(self):
        with self.lock:
            self.closed = True
            self.stop_event.set()
            if self.active is not None:
                job = self.jobs[self.active]
                if job["state"] in {"planning", "checking", "executing", "stopping"}:
                    job.update(stop_requested=True, state="stopping", stop_requested_at=now(), stop_cause="server_shutdown")
                    try:
                        self._save(job, "shutdown_stop_requested")
                    except OSError:
                        self.storage_fault = True
            thread = self.thread
        if thread and thread.is_alive():
            thread.join(15)
            if thread.is_alive():
                raise TimeoutError("Worker still stopping; record ownership remains held")
        self.lease.close()
