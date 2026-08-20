"""Registry adapters for the existing camera-supervised pick/place code."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skills.block_reacquisition import (
    SIM_OVERHEAD_BLOCK_PIXEL_TO_TABLE,
    SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE,
    locate_overhead_target,
    pick_until_verified,
)
from skills.pick_place_state_machine import (
    place_verified_transport,
    transport_verified_pick,
)
from skills.registry import (
    Precondition,
    SkillContext,
    SkillExecutor,
    SkillRegistry,
    SkillSpec,
)


@dataclass
class PickPlaceRuntime:
    env: object
    frames: list = field(default_factory=list)
    block_matrix: object = field(
        default_factory=lambda: {
            name: np.asarray(matrix).copy()
            for name, matrix in SIM_OVERHEAD_BLOCK_PIXEL_TO_TABLE.items()
        }
    )
    target_matrix: object = field(
        default_factory=lambda: np.asarray(
            SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE
        ).copy()
    )
    max_pick_attempts: int = 6
    max_task_cycles: int = 3
    max_task_duration_s: float = 300.0
    stage_budgets_s: dict[str, float] = field(
        default_factory=lambda: {
            "observe": 5.0,
            "pick": 200.0,
            "transport": 45.0,
            "place": 85.0,
        }
    )
    overhead_calibration: object | None = None
    target: object | None = None


def _state(context, name, default=False):
    return bool(context.state.get(name, default))


@contextmanager
def _cooperative_env_deadline(context):
    """Check the active skill budget at every robot control step."""
    env = context.runtime.env
    original_step = env.step
    previous_deadline_check = getattr(env, "_skill_deadline_check", None)

    def guarded_step(*args, **kwargs):
        context.check_deadline()
        value = original_step(*args, **kwargs)
        context.check_deadline()
        return value

    env.step = guarded_step
    env._skill_deadline_check = context.check_deadline
    try:
        context.check_deadline()
        yield
        context.check_deadline()
    finally:
        env.step = original_step
        if previous_deadline_check is None:
            delattr(env, "_skill_deadline_check")
        else:
            env._skill_deadline_check = previous_deadline_check


def _observe_target(context, _params):
    runtime = context.runtime
    with _cooperative_env_deadline(context):
        runtime.overhead_calibration = runtime.env.render_overhead()
        runtime.target = locate_overhead_target(
            runtime.overhead_calibration, runtime.target_matrix
        )
    target = runtime.target
    return {
        "success": bool(target.visible),
        "target_visible": bool(target.visible),
        "target_pixel": target.pixel,
        "target_xy": target.table_xy,
        "target_pixels": int(target.pixels),
        "failure_reason": None if target.visible else "target_not_visible",
    }


def _observe_effects(context, result, _succeeded):
    context.state["target_acquired"] = bool(result.get("target_visible"))
    context.state["task_done"] = False


def _pick(context, params, *, recovery=False):
    runtime = context.runtime
    if recovery and _state(context, "holding"):
        return {
            "success": True,
            "already_holding": True,
            "attempts": 0,
            "failure_reason": None,
        }
    with _cooperative_env_deadline(context):
        result = pick_until_verified(
            runtime.env,
            frames=runtime.frames,
            matrix=runtime.block_matrix,
            max_attempts=int(
                params.get("max_pick_attempts", runtime.max_pick_attempts)
            ),
            recovery=recovery,
        )
    result["recovery_pick"] = bool(recovery)
    return result


def _pick_effects(context, result, _succeeded):
    context.state["holding"] = bool(result and result.get("success"))
    context.state["at_target"] = False
    context.state["recovery_grasp"] = bool(
        result and result.get("success") and result.get("recovery_pick")
    )


def _transport(context, params):
    runtime = context.runtime
    duration = 3.0 if _state(context, "recovery_grasp") else 2.0
    with _cooperative_env_deadline(context):
        return transport_verified_pick(
            runtime.env,
            runtime.target.table_xy,
            runtime.overhead_calibration,
            frames=runtime.frames,
            on_transport_step=params.get("on_transport_step"),
            duration=duration,
        )


def _transport_effects(context, result, _succeeded):
    ready = bool(result and result.get("ready_for_place"))
    context.state["holding"] = ready
    context.state["at_target"] = ready


def _place(context, params):
    runtime = context.runtime
    with _cooperative_env_deadline(context):
        return place_verified_transport(
            runtime.env,
            runtime.target.pixel,
            runtime.overhead_calibration,
            frames=runtime.frames,
            target_matrix=runtime.target_matrix,
            block_matrix=runtime.block_matrix,
            on_place_step=params.get("on_place_step"),
            conservative=_state(context, "recovery_grasp"),
        )


def _place_effects(context, result, _succeeded):
    if not result:
        context.state["task_done"] = False
        context.state["holding"] = False
        context.state["at_target"] = False
        return
    done = bool(
        result
        and result.get("next_state") == "DONE"
        and result.get("camera_place_confirmed")
    )
    released = bool(result and result.get("released"))
    dropped = bool(result and result.get("drop_detected"))
    context.state["task_done"] = done
    context.state["holding"] = not done and not released and not dropped
    context.state["at_target"] = context.state["holding"]
    if done:
        context.state["recovery_grasp"] = False


def build_pick_place_registry() -> SkillRegistry:
    target_ready = Precondition(
        "target_acquired",
        lambda context: _state(context, "target_acquired"),
        "target_not_acquired",
    )
    not_holding = Precondition(
        "not_holding",
        lambda context: not _state(context, "holding"),
        "object_already_held",
    )
    holding = Precondition(
        "holding",
        lambda context: _state(context, "holding"),
        "object_not_held",
    )

    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            "observe_target",
            _observe_target,
            lambda _context, result: bool(result.get("target_visible")),
            description="Locate and retain the overhead target calibration.",
            apply_effects=_observe_effects,
            timeout_s=5.0,
        )
    )
    registry.register(
        SkillSpec(
            "recover_pick",
            lambda context, params: _pick(context, params, recovery=True),
            lambda _context, result: bool(result.get("success")),
            description="Reacquire and pick a dropped or missed block.",
            preconditions=(target_ready,),
            apply_effects=_pick_effects,
            timeout_s=120.0,
        )
    )
    registry.register(
        SkillSpec(
            "pick",
            lambda context, params: _pick(context, params, recovery=False),
            lambda _context, result: bool(result.get("success")),
            description="Camera-gated initial pick.",
            preconditions=(target_ready, not_holding),
            apply_effects=_pick_effects,
            timeout_s=120.0,
            recovery_plan=("recover_pick",),
        )
    )
    registry.register(
        SkillSpec(
            "transport",
            _transport,
            lambda _context, result: bool(result.get("ready_for_place")),
            description="Transport while supervising wrist attachment and overhead motion.",
            preconditions=(target_ready, holding),
            apply_effects=_transport_effects,
            timeout_s=30.0,
            recovery_plan=("recover_pick", "transport"),
        )
    )
    registry.register(
        SkillSpec(
            "place_6cm",
            _place,
            lambda _context, result: bool(
                result.get("next_state") == "DONE"
                and result.get("camera_place_confirmed")
            ),
            description="Align, release at 6 cm, and verify stable target coverage.",
            preconditions=(target_ready, holding),
            apply_effects=_place_effects,
            timeout_s=60.0,
            recovery_plan=("recover_pick", "transport", "place_6cm"),
        )
    )
    registry.validate()
    return registry


def run_registered_pick_place(
    runtime: PickPlaceRuntime,
    *,
    audit_path: str | Path | None = None,
    on_transport_step=None,
    on_place_step=None,
):
    context = SkillContext(
        runtime=runtime,
        state={
            "target_acquired": False,
            "holding": False,
            "at_target": False,
            "task_done": False,
            "recovery_grasp": False,
        },
        audit_path=Path(audit_path) if audit_path else None,
        skill_stages={
            "observe_target": "observe",
            "pick": "pick",
            "recover_pick": "pick",
            "transport": "transport",
            "place_6cm": "place",
        },
        stage_budget_remaining_s={
            str(name): float(seconds)
            for name, seconds in runtime.stage_budgets_s.items()
        },
    )
    context.operation_deadline_s = (
        context.clock() + float(runtime.max_task_duration_s)
    )
    executor = SkillExecutor(build_pick_place_registry(), context)
    params = {
        "max_pick_attempts": runtime.max_pick_attempts,
        "on_transport_step": on_transport_step,
        "on_place_step": on_place_step,
    }
    executions = []
    observed = executor.execute("observe_target", params)
    executions.append(observed)
    cycles_attempted = 0
    if observed.success:
        for cycle in range(1, runtime.max_task_cycles + 1):
            cycles_attempted = cycle
            for name in ("pick", "transport", "place_6cm"):
                execution = executor.execute(name, params)
                executions.append(execution)
                if not execution.success:
                    break
            if context.state["task_done"]:
                break
            # Any failed chain is treated as an uncertain grasp. The next
            # bounded cycle must visually reacquire before moving again.
            context.state["holding"] = False
            context.state["at_target"] = False
    return {
        "success": bool(context.state["task_done"]),
        "final_state": dict(context.state),
        "cycles_attempted": cycles_attempted,
        "max_task_cycles": runtime.max_task_cycles,
        "max_task_duration_s": runtime.max_task_duration_s,
        "stage_budgets_s": {
            str(name): float(seconds)
            for name, seconds in runtime.stage_budgets_s.items()
        },
        "stage_budget_remaining_s": dict(context.stage_budget_remaining_s),
        "registered_skills": list(executor.registry.names()),
        "executions": [execution.to_dict() for execution in executions],
    }
