"""Conservative per-sheet row reservations; never detects or changes real paper."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from uuid import uuid4

from .planner import LayoutSpaceError, PaperSettings, plan_text


ID = re.compile(r"^[a-f0-9]{32}$")
PAGE_STATES = {"ready", "awaiting_new_sheet", "inspection_required", "confirmation_required", "replaced"}
RESERVATION_STATES = {"reserved", "occupied", "partial", "uncertain", "released_no_motion"}
MIN_CHARACTER_MM = 8
MAX_CHARACTER_MM = 40


class PaperSpaceError(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def timestamp():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class PageLedger:
    """Caller holds the job-manager lock; persist before any execution can begin."""

    def __init__(self, root, config, save_json, legacy_history=False, interrupted_jobs=()):
        self.path = Path(root) / "pages.json"
        self.save_json = save_json
        self.paper = PaperSettings(width_mm=297, height_mm=210)
        margin = self.paper.margin_mm
        self.rect = (margin, margin, self.paper.width_mm - margin, self.paper.height_mm - margin)
        self.robot_rect = tuple(config["paper_frame"]["working_rect_mm"])
        self.guard = (config["settings"]["position_tolerance_mm"] + config["dynamics"]["tracking_tolerance_mm"]
                      + config["dynamics"]["max_stop_displacement_mm"])
        layout_config = {"simulation": config, "paper": asdict(self.paper), "layout_policy": "full_a4_landscape_v1"}
        self.signature = hashlib.sha256(json.dumps(layout_config, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if self.path.is_symlink():
            raise ValueError("Paper ledger must not be a symlink")
        if self.path.exists():
            self.book = json.loads(self.path.read_text(encoding="utf-8"))
            try:
                self._validate()
            except (KeyError, TypeError, StopIteration) as error:
                raise ValueError("Malformed stored paper ledger") from error
            changed = False
            for page in self.book["pages"]:
                for reservation in page["reservations"]:
                    if reservation["state"] == "reserved" or (reservation["job_id"] in interrupted_jobs and reservation["state"] != "released_no_motion"):
                        reservation.update(state="uncertain", finished_at=timestamp(), reason="restart_reservation_retained")
                        if page["page_id"] == self.book["current_page_id"]:
                            page.update(state="inspection_required", reason="interrupted_execution_or_reservation")
                        changed = True
            if self.current["config_signature"] != self.signature:
                self.current.update(state="confirmation_required", reason="paper_layout_or_simulation_profile_changed")
                changed = True
            if changed:
                self._save()
        else:
            page = self._new_page(1, "confirmation_required" if legacy_history else "ready")
            if legacy_history:
                page["reason"] = "legacy_jobs_have_no_paper_occupancy"
            self.book = {"format_version": 1, "kind": "writing_paper_ledger", "mode": "simulation_only",
                         "motion_authorized": False, "current_page_id": page["page_id"], "pages": [page]}
            self._save()

    @property
    def current(self):
        return next(p for p in self.book["pages"] if p["page_id"] == self.book["current_page_id"])

    def _new_page(self, number, state):
        return {"page_id": uuid4().hex, "number": number, "created_at": timestamp(), "state": state,
                "reason": None, "paper": asdict(self.paper), "working_rect_mm": list(self.rect),
                "guard_mm": self.guard, "row_gap_mm": self.paper.line_gap_mm,
                "next_row_y_mm": self.rect[1], "config_signature": self.signature, "reservations": [],
                "motion_authorized": False, "physical_calibration_required": True}

    def _validate(self):
        b = self.book
        if (not isinstance(b, dict) or b.get("format_version") != 1 or b.get("kind") != "writing_paper_ledger"
                or b.get("mode") != "simulation_only" or b.get("motion_authorized") is not False
                or not isinstance(b.get("pages"), list) or not b["pages"]):
            raise ValueError("Invalid paper ledger")
        ids = set()
        acknowledged = b.get("acknowledged_job_ids", [])
        if not isinstance(acknowledged, list) or any(not isinstance(v, str) or not ID.fullmatch(v) for v in acknowledged):
            raise ValueError("Invalid sheet confirmation history")
        for page in b["pages"]:
            if (not isinstance(page, dict) or not isinstance(page.get("page_id"), str) or not ID.fullmatch(page["page_id"])
                    or page["page_id"] in ids or page.get("state") not in PAGE_STATES
                    or page.get("motion_authorized") is not False or page.get("physical_calibration_required") is not True):
                raise ValueError("Invalid paper identity or state")
            ids.add(page["page_id"])
            if (not isinstance(page.get("number"), int) or isinstance(page["number"], bool) or page["number"] < 1
                    or not isinstance(page.get("created_at"), str)
                    or not isinstance(page.get("config_signature"), str)
                    or not re.fullmatch(r"[a-f0-9]{64}", page["config_signature"])):
                raise ValueError("Invalid page metadata")
            paper = PaperSettings(**page["paper"])
            paper.validate()
            rect = page["working_rect_mm"]
            if not isinstance(rect, list) or len(rect) != 4 or not all(finite(v) for v in rect):
                raise ValueError("Invalid stored work area")
            x0, y0, x1, y1 = rect
            if not (paper.margin_mm <= x0 < x1 <= paper.width_mm - paper.margin_mm
                    and paper.margin_mm <= y0 < y1 <= paper.height_mm - paper.margin_mm):
                raise ValueError("Stored work area outside paper")
            if not finite(page.get("guard_mm")) or page["guard_mm"] <= 0 or not finite(page.get("row_gap_mm")) or page["row_gap_mm"] < 0:
                raise ValueError("Invalid page spacing")
            if not finite(page.get("next_row_y_mm")) or not isinstance(page.get("reservations"), list):
                raise ValueError("Invalid paper cursor")
            used, jobs = [], set()
            for r in page["reservations"]:
                if (not isinstance(r, dict) or not isinstance(r.get("job_id"), str) or not ID.fullmatch(r["job_id"])
                        or r["job_id"] in jobs or r.get("state") not in RESERVATION_STATES):
                    raise ValueError("Invalid reservation")
                jobs.add(r["job_id"])
                box = r.get("reserved_rect_mm")
                if not isinstance(box, list) or len(box) != 4 or not all(finite(v) for v in box):
                    raise ValueError("Invalid reservation bounds")
                if not (x0 <= box[0] < box[2] <= x1 and y0 <= box[1] < box[3] <= y1):
                    raise ValueError("Reservation outside work area")
                if r["state"] != "released_no_motion":
                    if any(box[1] < other[3] and other[1] < box[3] for other in used):
                        raise ValueError("Overlapping stored reservations")
                    used.append(box)
            expected = max((box[3] + page["row_gap_mm"] for box in used), default=y0)
            if abs(page["next_row_y_mm"] - expected) > 1e-7:
                raise ValueError("Paper cursor disagrees with occupied regions")
            if page["page_id"] == b.get("current_page_id") and page["config_signature"] == self.signature:
                if (tuple(rect) != self.rect or abs(page["guard_mm"] - self.guard) > 1e-7
                        or page["row_gap_mm"] != self.paper.line_gap_mm or page["paper"] != asdict(self.paper)):
                    raise ValueError("Stored work area or guard disagrees with its configuration")
        if b.get("current_page_id") not in ids:
            raise ValueError("Missing current sheet")

    def _save(self):
        self._validate()
        self.save_json(self.path, self.book)

    def snapshot(self):
        page = deepcopy(self.current)
        page["archived_pages"] = len(self.book["pages"]) - 1
        page["remaining_height_mm"] = max(0., page["working_rect_mm"][3] - page["next_row_y_mm"])
        page["robot_candidate_rect_mm"] = list(self.robot_rect)
        page["requested_paper"] = asdict(self.paper)
        page["layout_scope"] = "paper_margins_not_robot_reach"
        return page

    def reconcile(self, jobs):
        """A stale but valid ledger must not make a recorded job's region look empty."""
        if self.current["state"] in {"confirmation_required", "inspection_required"}:
            return
        known_pages = {p["page_id"] for p in self.book["pages"]}
        acknowledged = set(self.book.get("acknowledged_job_ids", []))
        reservations = {r["job_id"]: r for r in self.current["reservations"]}
        inconsistent = []
        for job in jobs:
            placement = job.get("placement")
            if placement is None or job["job_id"] in acknowledged:
                continue
            if not isinstance(placement, dict):
                raise ValueError("Invalid stored job placement")
            page_id = placement.get("page_id")
            if page_id not in known_pages:
                inconsistent.append(job["job_id"])
            elif page_id == self.current["page_id"]:
                r = reservations.get(job["job_id"])
                if (r is None or r["reserved_rect_mm"] != placement.get("reserved_rect_mm")
                        or (job["state"] == "completed" and r["state"] == "released_no_motion")):
                    inconsistent.append(job["job_id"])
        if inconsistent:
            self.current.update(state="inspection_required", reason="job_ledger_accounting_disagrees", inconsistent_job_ids=inconsistent)
            self._save()

    def placement(self, text, character_mm):
        page = self.current
        paper = replace(self.paper, character_mm=character_mm)
        try:
            plan_text(text, paper)  # Validate all characters before checking sheet capacity.
        except LayoutSpaceError as error:
            raise PaperSpaceError("text_does_not_fit_blank_sheet") from error
        if page["state"] != "ready":
            raise PaperSpaceError(page["reason"] or page["state"])
        x0, y0, x1, y1 = self.rect
        pad = self.guard
        fresh_rect = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
        try:
            plan_text(text, paper, fresh_rect)
        except LayoutSpaceError as error:
            raise PaperSpaceError("text_does_not_fit_blank_sheet") from error
        layout = (x0 + pad, page["next_row_y_mm"] + pad, x1 - pad, y1 - pad)
        try:
            plan = plan_text(text, paper, layout)
        except (LayoutSpaceError, ValueError) as error:
            raise PaperSpaceError("insufficient_remaining_space") from error
        height = plan.lines * paper.character_mm + (plan.lines - 1) * paper.line_gap_mm + 2 * pad
        placement = {"page_id": page["page_id"], "page_number": page["number"],
                     "reserved_rect_mm": [x0, page["next_row_y_mm"], x1, page["next_row_y_mm"] + height],
                     "layout_rect_mm": list(layout), "guard_mm": pad, "lines": plan.lines,
                     "config_signature": self.signature, "mode": "simulation_only", "motion_authorized": False}
        return plan, placement

    def mark_full(self):
        if self.current["state"] == "ready":
            self.current.update(state="awaiting_new_sheet", reason="insufficient_remaining_space")
            self._save()

    def reserve(self, job_id, placement):
        if placement["page_id"] != self.current["page_id"] or self.current["state"] != "ready":
            raise PaperSpaceError("sheet_changed")
        self.current["reservations"].append({"job_id": job_id, "reserved_rect_mm": placement["reserved_rect_mm"],
                                              "state": "reserved", "created_at": timestamp()})
        self.current["next_row_y_mm"] = placement["reserved_rect_mm"][3] + self.current["row_gap_mm"]
        self._save()

    def finish(self, job_id, state, report):
        r = next(r for r in self.current["reservations"] if r["job_id"] == job_id)
        no_motion = (isinstance(report, dict) and report.get("kind") == "writing_dynamics_rollout"
                     and report.get("phase") == "preflight" and report.get("motion_authorized") is False
                     and report.get("status") in {"rejected", "stopped"} and report.get("samples") == []
                     and report.get("simulation_steps", 0) == 0 and isinstance(report.get("preflight"), dict)
                     and (report["preflight"].get("status") in {"rejected", "stopped"}
                          or (isinstance(report.get("failure"), dict)
                              and report["failure"].get("reason") == "reference_exceeds_actuator_control_range")))
        if no_motion:
            r.update(state="released_no_motion", reason="preflight_confirmed_no_motion")
            used = [s for s in self.current["reservations"] if s["state"] != "released_no_motion"]
            self.current["next_row_y_mm"] = max((s["reserved_rect_mm"][3] + self.current["row_gap_mm"] for s in used), default=self.rect[1])
        elif state == "completed":
            r.update(state="occupied", reason="virtual_trajectory_completed")
        elif state == "stopped" and report and report.get("stop_settled") is True and report.get("stop_test_passed") is True:
            r.update(state="partial", reason="settled_stop_area_retained")
        else:
            r.update(state="uncertain", reason="unverified_execution_area_retained")
            self.current.update(state="inspection_required", reason="execution_result_uncertain_replace_sheet")
        if (self.current["state"] == "ready"
                and self.rect[3] - self.current["next_row_y_mm"] < MIN_CHARACTER_MM + 2 * self.guard):
            self.current.update(state="awaiting_new_sheet", reason="insufficient_remaining_space")
        r["finished_at"] = timestamp()
        self._save()
        return r["state"]

    def reset(self, expected_page_id, existing_job_ids=()):
        if expected_page_id != self.current["page_id"]:
            raise PaperSpaceError("stale_sheet_confirmation")
        self.current.update(state="replaced", replaced_at=timestamp())
        page = self._new_page(len(self.book["pages"]) + 1, "ready")
        page["confirmation"] = "human_confirmed_new_sheet_no_automatic_detection"
        self.book["pages"].append(page)
        self.book["current_page_id"] = page["page_id"]
        self.book["acknowledged_job_ids"] = sorted(set(self.book.get("acknowledged_job_ids", [])) | set(existing_job_ids))
        self._save()
        return self.snapshot()
