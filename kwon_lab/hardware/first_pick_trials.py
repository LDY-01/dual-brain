"""Structured summaries for the first ten physical pick trials."""

from __future__ import annotations

from collections import Counter


FAILURE_STAGES = (
    "not_detected",
    "aim_error",
    "pushed_block",
    "missed_grasp",
    "slipped_after_lift",
    "safety_guard",
    "operator_abort",
    "other",
)


def summarize_first_pick_trials(trials, expected_trials=10, pass_successes=8):
    successes = sum(bool(item.get("success")) for item in trials)
    lifts = sum(bool(item.get("lift_verified")) for item in trials)
    failures = Counter(
        item.get("failure_stage") or "unclassified"
        for item in trials
        if not item.get("success")
    )
    complete = len(trials) == expected_trials
    passed = complete and successes >= pass_successes
    if not complete:
        recommendation = "complete_remaining_trials"
    elif passed and not failures.get("safety_guard"):
        recommendation = "proceed_to_low_speed_place_validation"
    else:
        recommendation = "review_failures_before_motion_expansion_or_finetuning"
    return {
        "expected_trials": int(expected_trials),
        "recorded_trials": len(trials),
        "complete": complete,
        "successes": successes,
        "success_rate": successes / len(trials) if trials else 0.0,
        "lift_verified": lifts,
        "failure_counts": dict(sorted(failures.items())),
        "pass_threshold_successes": int(pass_successes),
        "passed": passed,
        "recommendation": recommendation,
    }
