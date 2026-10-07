"""A fixed text-call response; no language model or hardware is involved."""

from datetime import datetime, timedelta, timezone
import unicodedata
from uuid import uuid4

from .planner import PaperSettings, plan_text


def plan_call(text: str) -> dict:
    if not isinstance(text, str) or len(text) > 120:
        raise ValueError("호출어는 120자 이내의 텍스트여야 합니다.")
    normalized = unicodedata.normalize("NFC", text).strip()
    if normalized != "야":
        raise ValueError("현재 지원하는 호출어는 ‘야’입니다.")
    plan = plan_text("네", PaperSettings(character_mm=28))
    return {
        "job_id": str(uuid4()),
        "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "input": normalized,
        "response": "네",
        "response_source": "fixed_call_response",
        "mode": "geometry_preview_only",
        "plan": plan.to_dict(),
    }
