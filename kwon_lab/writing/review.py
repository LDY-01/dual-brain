"""Generate reproducible geometry samples for the M0 human readability review."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .planner import PaperSettings, plan_text
from .preview import export_preview


REVIEW_CASES = (
    ("large_glyphs", "가 한 글",
     PaperSettings(character_mm=24)),
    ("mixed_vowels", "과 괘 괴 궈 궤 귀 의", PaperSettings()),
    ("finals", "각 간 갇 갈 감 갑 갓 강 값 읽 앉 많",
     PaperSettings()),
    ("fixed_sentence", "안녕하세요.", PaperSettings()),
    ("wrapped_sentence", "오늘은 어떤 일을 도와드릴까요?", PaperSettings()),
)


def export_review(destination: Path) -> Path:
    """Export labelled samples, not a blind physical-writing evaluation."""
    plans = [(case_id, plan_text(text, paper)) for case_id, text, paper in REVIEW_CASES]
    cases = []
    for case_id, plan in plans:
        paths = export_preview(plan, destination / case_id)
        cases.append({
            "id": case_id,
            "text": plan.text,
            "lines": plan.lines,
            "stroke_count": len(plan.strokes),
            "paper": plan.to_dict()["paper"],
            "human_review_status": "pending",
            "files": {name: path.relative_to(destination).as_posix()
                      for name, path in paths.items()},
        })
    manifest = {
        "milestone": "M0",
        "generated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "mode": "geometry_preview_only",
        "motion_authorized": False,
        "physical_calibration_required": True,
        "human_review_status": "pending",
        "physical_trials": 0,
        "cases": cases,
    }
    path = destination / "review.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    return path
