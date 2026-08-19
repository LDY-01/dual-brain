"""Validate the generic executor and its MuJoCo pick/place adapters."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco

from envs.so101_pick_env import BLOCK_HALF_H, SO101PickEnv
from skills.pick_place_catalog import PickPlaceRuntime, run_registered_pick_place
from skills.registry import (
    Precondition,
    SkillContext,
    SkillExecutor,
    SkillRegistry,
    SkillSpec,
)


def run_contract_checks():
    state = {"holding": False, "recoveries": 0}

    def pick(_context, _params):
        return {"success": False, "failure_reason": "synthetic_miss"}

    def recover(context, _params):
        context.state["recoveries"] += 1
        return {"success": True}

    def effect(context, result, _succeeded):
        context.state["holding"] = bool(result.get("success"))

    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            "recover",
            recover,
            lambda _context, result: result["success"],
            apply_effects=effect,
        )
    )
    registry.register(
        SkillSpec(
            "pick",
            pick,
            lambda _context, result: result["success"],
            apply_effects=effect,
            recovery_plan=("recover",),
        )
    )
    registry.register(
        SkillSpec(
            "transport",
            lambda _context, _params: {"success": True},
            lambda _context, result: result["success"],
            preconditions=(
                Precondition(
                    "holding",
                    lambda context: context.state["holding"],
                    "object_not_held",
                ),
            ),
        )
    )
    registry.register(
        SkillSpec(
            "slow",
            lambda _context, _params: (time.sleep(0.05) or {"success": True}),
            lambda _context, result: result["success"],
            timeout_s=0.001,
        )
    )
    executor = SkillExecutor(registry, SkillContext(runtime=None, state=state))
    blocked = executor.execute("transport")
    recovered = executor.execute("pick")
    transported = executor.execute("transport")
    timed_out = executor.execute("slow")

    fake_time = [0.0]
    cooperative_steps = [0]

    def fake_clock():
        return fake_time[0]

    def cooperative(context, _params):
        for _ in range(10):
            cooperative_steps[0] += 1
            fake_time[0] += 0.4
            context.check_deadline()
        return {"success": True}

    bounded_registry = SkillRegistry()
    bounded_registry.register(
        SkillSpec(
            "bounded",
            cooperative,
            lambda _context, result: result["success"],
            timeout_s=1.0,
        )
    )
    bounded = SkillExecutor(
        bounded_registry,
        SkillContext(runtime=None, clock=fake_clock),
    ).execute("bounded")
    checks = {
        "precondition_blocked": blocked.status == "blocked",
        "recovery_completed": recovered.status == "recovered",
        "effect_applied": state["holding"] and state["recoveries"] == 1,
        "transport_after_recovery": transported.success,
        "timeout_rejected": timed_out.status == "timed_out",
        "cooperative_timeout_interrupted": (
            bounded.status == "timed_out" and cooperative_steps[0] == 3
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def inject_slip(env):
    addr = env.block_qpos_addr
    block = env.data.qpos[addr : addr + 7].copy()
    block[1] += 0.055
    block[2] = max(0.075, block[2] - 0.025)
    env.data.qpos[addr : addr + 7] = block
    joint_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "block_free"
    )
    dof = env.model.jnt_dofadr[joint_id]
    env.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def run_mujoco(seed, forced_place_drop, audit_path):
    env = SO101PickEnv(camera="wrist")
    env.reset(seed=seed)
    env.data.qpos[env.block_qpos_addr : env.block_qpos_addr + 3] = [
        0.23,
        0.0,
        BLOCK_HALF_H,
    ]
    mujoco.mj_forward(env.model, env.data)
    injected = False

    def maybe_inject(current_env, step, _signals):
        nonlocal injected
        if forced_place_drop and not injected and step == 3:
            inject_slip(current_env)
            injected = True

    try:
        result = run_registered_pick_place(
            PickPlaceRuntime(env),
            audit_path=audit_path,
            on_place_step=maybe_inject,
        )
        info = env._get_info()
        result["truth_success"] = bool(info["success"])
        result["truth_target_coverage"] = float(info["target_coverage"])
        result["fault_injected"] = injected
        if not result["success"] or not result["truth_success"]:
            raise AssertionError(
                f"registered MuJoCo flow failed: state={result['final_state']}"
            )
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--mujoco", action="store_true")
    parser.add_argument("--forced-place-drop", action="store_true")
    parser.add_argument("--seed", type=int, default=9101)
    args = parser.parse_args()

    report = {"contract_checks": run_contract_checks()}
    if args.mujoco:
        report["mujoco"] = run_mujoco(
            args.seed, args.forced_place_drop, args.audit
        )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
