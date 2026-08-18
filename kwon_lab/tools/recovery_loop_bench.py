"""Validate bounded autonomous recovery through task completion."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco

from envs.so101_pick_env import BLOCK_HALF_H, SO101PickEnv
from skills.block_reacquisition import locate_overhead_block
from skills.pick_place_state_machine import run_pick_place_until_done


POSITIONS = (
    (0.18, -0.10),
    (0.23, 0.00),
    (0.28, 0.10),
)


def inject_slip(env):
    """Test-only drop injection; runtime state logic cannot read object truth."""
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


def summarize_cycle(cycle):
    pick = cycle["pick"]
    transport = cycle["transport"]
    place = cycle["place"]
    final_pick_attempt = (
        pick["attempt_reports"][-1]
        if pick and pick["attempt_reports"]
        else None
    )
    return {
        "cycle": cycle["cycle"],
        "pick_verified": bool(pick and pick["success"]),
        "pick_attempts": pick["attempts"] if pick else 0,
        "pick_estimated_xy": (
            final_pick_attempt["estimated_table_xy"]
            if final_pick_attempt else None
        ),
        "pick_pose_class": (
            final_pick_attempt["overhead_pose_class"]
            if final_pick_attempt else None
        ),
        "clear_view_selected_pose": (
            final_pick_attempt["clear_view"]["selected_pose"]
            if final_pick_attempt else None
        ),
        "clear_view_alternate_used": bool(
            final_pick_attempt
            and final_pick_attempt["clear_view"]["alternate_pose_used"]
        ),
        "pick_center_z_m": (
            final_pick_attempt["pick_center_z_m"]
            if final_pick_attempt else None
        ),
        "pick_wrist_roll_rad": (
            final_pick_attempt["pick_wrist_roll_rad"]
            if final_pick_attempt else None
        ),
        "pick_final_gripper": (
            final_pick_attempt["final_gripper_joint"]
            if final_pick_attempt else None
        ),
        "pick_monitor_metrics": (
            final_pick_attempt["grasp_monitor_metrics"]
            if final_pick_attempt else None
        ),
        "transport_ready": bool(
            transport and transport["ready_for_place"]
        ),
        "transport_drop_detected": bool(
            transport and transport["drop_detected"]
        ),
        "place_started": place is not None,
        "place_drop_detected": bool(place and place["drop_detected"]),
        "released": bool(place and place["released"]),
        "camera_place_confirmed": bool(
            place and place["camera_place_confirmed"]
        ),
        "next_state": cycle["next_state"],
        "failure_reason": (
            place["failure_reason"]
            if place
            else transport["failure_reason"]
            if transport
            else pick["failure_reason"]
            if pick
            else "target_not_visible"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--max-pick-attempts", type=int, default=6)
    parser.add_argument("--forced-place-drop", action="store_true")
    parser.add_argument("--inject-step", type=int, default=3)
    args = parser.parse_args()

    env = SO101PickEnv(camera="wrist")
    block_xy = POSITIONS[args.position_index]
    env.reset(seed=args.seed)
    addr = env.block_qpos_addr
    env.data.qpos[addr : addr + 3] = [*block_xy, BLOCK_HALF_H]
    mujoco.mj_forward(env.model, env.data)
    injected = False
    injected_cycle = None

    def maybe_inject(current_env, step, _signals):
        nonlocal injected, injected_cycle
        if (
            args.forced_place_drop
            and not injected
            and step == args.inject_step
        ):
            inject_slip(current_env)
            injected = True
            injected_cycle = 1

    def print_cycle(cycle):
        print(json.dumps(summarize_cycle(cycle)), flush=True)

    result = run_pick_place_until_done(
        env,
        max_cycles=args.max_cycles,
        max_pick_attempts_per_cycle=args.max_pick_attempts,
        on_place_step=maybe_inject,
        on_cycle_complete=print_cycle,
    )
    info = env._get_info()
    overhead_block = locate_overhead_block(env.render_overhead())
    truth_block_qpos = env.data.qpos[
        env.block_qpos_addr : env.block_qpos_addr + 7
    ].copy()
    cycle_summaries = [
        summarize_cycle(cycle) for cycle in result["cycle_reports"]
    ]
    report = {
        "mode": (
            "forced_place_drop_recovery"
            if args.forced_place_drop
            else "normal"
        ),
        "block_xy": block_xy,
        "fault_injected": injected,
        "fault_injected_cycle": injected_cycle,
        "task_success": result["success"],
        "cycles_attempted": result["cycles_attempted"],
        "recovery_cycles": result["recovery_cycles"],
        "final_state": result["final_state"],
        "stop_reason": result["stop_reason"],
        "truth_success": bool(info["success"]),
        "truth_target_coverage": float(info["target_coverage"]),
        "truth_block_height_m": float(info["block_height"]),
        "truth_block_xyz": truth_block_qpos[:3].tolist(),
        "truth_block_quaternion": truth_block_qpos[3:7].tolist(),
        "post_task_overhead_visible": overhead_block.visible,
        "post_task_overhead_pixels": overhead_block.pixels,
        "post_task_estimated_block_xy": overhead_block.table_xy,
        "post_task_block_reachable": overhead_block.reachable,
        "cycles": cycle_summaries,
    }
    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
