"""Minimal registry and executor for reusable robot skills.

The executor deliberately does not try to interrupt a robot function from a
background thread.  Robot skills must remain internally bounded; ``timeout_s``
is an additional contract check that rejects an overlong result and routes it
through the declared recovery plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np


SkillRunner = Callable[["SkillContext", Mapping[str, Any]], Any]
SkillVerifier = Callable[["SkillContext", Any], bool]
SkillEffect = Callable[["SkillContext", Any, bool], None]
ConditionCheck = Callable[["SkillContext"], bool]


@dataclass(frozen=True)
class Precondition:
    name: str
    check: ConditionCheck
    failure_reason: str


@dataclass(frozen=True)
class SkillSpec:
    name: str
    run: SkillRunner
    success_verifier: SkillVerifier
    description: str = ""
    preconditions: tuple[Precondition, ...] = ()
    apply_effects: SkillEffect | None = None
    timeout_s: float = 30.0
    recovery_plan: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Skill name must not be empty")
        if self.timeout_s <= 0:
            raise ValueError(f"Skill {self.name!r} timeout_s must be positive")


@dataclass
class SkillContext:
    runtime: Any
    state: dict[str, Any] = field(default_factory=dict)
    audit_path: Path | None = None
    clock: Callable[[], float] = time.monotonic


@dataclass
class SkillExecution:
    skill: str
    status: str
    success: bool
    elapsed_s: float
    failure_reason: str | None = None
    result: Any = None
    recovered: bool = False
    recovery: list["SkillExecution"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            raise ValueError(f"Skill already registered: {spec.name}")
        self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown skill {name!r}; available={sorted(self._skills)}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    def validate(self) -> None:
        for spec in self._skills.values():
            missing = [name for name in spec.recovery_plan if name not in self._skills]
            if missing:
                raise ValueError(
                    f"Skill {spec.name!r} has unknown recovery skills: {missing}"
                )


class SkillExecutor:
    def __init__(self, registry: SkillRegistry, context: SkillContext):
        registry.validate()
        self.registry = registry
        self.context = context

    def execute(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        allow_recovery: bool = True,
    ) -> SkillExecution:
        params = params or {}
        spec = self.registry.get(name)

        for condition in spec.preconditions:
            try:
                satisfied = bool(condition.check(self.context))
            except Exception as exc:
                execution = SkillExecution(
                    name,
                    "blocked",
                    False,
                    0.0,
                    f"precondition_error:{condition.name}:{type(exc).__name__}:{exc}",
                )
                self._audit(execution)
                return execution
            if not satisfied:
                execution = SkillExecution(
                    name, "blocked", False, 0.0, condition.failure_reason
                )
                self._audit(execution)
                return execution

        start = self.context.clock()
        result = None
        status = "failed"
        failure_reason = None
        succeeded = False
        try:
            result = spec.run(self.context, params)
            elapsed = max(0.0, self.context.clock() - start)
            if elapsed > spec.timeout_s:
                status = "timed_out"
                failure_reason = (
                    f"elapsed {elapsed:.3f}s exceeded timeout {spec.timeout_s:.3f}s"
                )
            else:
                succeeded = bool(spec.success_verifier(self.context, result))
                status = "succeeded" if succeeded else "failed"
                if not succeeded:
                    failure_reason = self._failure_reason(result)
            if spec.apply_effects is not None:
                spec.apply_effects(self.context, result, succeeded)
        except Exception as exc:
            elapsed = max(0.0, self.context.clock() - start)
            status = "failed"
            succeeded = False
            failure_reason = f"{type(exc).__name__}:{exc}"

        execution = SkillExecution(
            skill=name,
            status=status,
            success=succeeded,
            elapsed_s=elapsed,
            failure_reason=failure_reason,
            result=result,
        )

        if not succeeded and allow_recovery and spec.recovery_plan:
            recovery_results = []
            recovered = True
            for recovery_name in spec.recovery_plan:
                recovery = self.execute(
                    recovery_name, params, allow_recovery=False
                )
                recovery_results.append(recovery)
                if not recovery.success:
                    recovered = False
                    break
            execution.recovery = recovery_results
            if recovered:
                execution.status = "recovered"
                execution.success = True
                execution.recovered = True

        self._audit(execution)
        return execution

    @staticmethod
    def _failure_reason(result) -> str:
        if isinstance(result, Mapping):
            for key in ("failure_reason", "stop_reason", "reason"):
                if result.get(key):
                    return str(result[key])
        return "success_verifier_rejected_result"

    def _audit(self, execution: SkillExecution) -> None:
        if self.context.audit_path is None:
            return
        path = Path(self.context.audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution": execution.to_dict(),
            "state": _json_safe(self.context.state),
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
