"""Validate the full camera-gated PICK -> TRANSPORT -> PLACE sequence."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco

from envs.so101_pick_env import BLOCK_HALF_H, SO101PickEnv
from skills.pick_place_state_machine import pick_transport_place


POSITIONS = (
    (0.18, -0.10),
    (0.23, 0.00),
    (0.28, 0.10),
    (0.18, 0.10),
    (0.28, -0.10),
)


def inject_slip(env):
    """Test-only drop injection; state-machine code never sees object truth."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=9100)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--forced-place-drop", action="store_true")
    parser.add_argument("--inject-step", type=int, default=3)
    args = parser.parse_args()

    env = SO101PickEnv(camera="wrist")
    rows = []
    selected = (
        range(args.episodes) if args.indices is None else args.indices
    )
    for index in selected:
        block_xy = POSITIONS[index % len(POSITIONS)]
        env.reset(seed=args.start_seed + index)
        addr = env.block_qpos_addr
        env.data.qpos[addr : addr + 3] = [*block_xy, BLOCK_HALF_H]
        mujoco.mj_forward(env.model, env.data)
        injected = False

        def maybe_inject(current_env, step, _signals):
            nonlocal injected
            if (
                args.forced_place_drop
                and not injected
                and step == args.inject_step
            ):
                inject_slip(current_env)
                injected = True

        result = pick_transport_place(
            env,
            on_place_step=maybe_inject,
        )
        info = env._get_info()
        pick = result["pick"]
        transport = result["transport"]
        place = result["place"]
        row = {
            "index": index,
            "block_xy": block_xy,
            "forced_place_drop": bool(args.forced_place_drop),
            "fault_injected": injected,
            "pick_verified": bool(pick and pick["success"]),
            "pick_attempts": pick["attempts"] if pick else 0,
            "transport_ready": bool(
                transport and transport["ready_for_place"]
            ),
            "place_started": place is not None,
            "place_drop_detected": bool(place and place["drop_detected"]),
            "place_drop_detected_step": (
                place["drop_detected_step"] if place else None
            ),
            "released": bool(place and place["released"]),
            "alignment_errors_px": (
                place["alignment_errors_px"] if place else []
            ),
            "alignment_skipped_reason": (
                place["alignment_skipped_reason"] if place else None
            ),
            "camera_place_confirmed": bool(
                place and place["camera_place_confirmed"]
            ),
            "camera_target_coverage": (
                place["image_target_coverage"] if place else 0.0
            ),
            "camera_table_area_ratio": (
                place["table_area_ratio"] if place else 0.0
            ),
            "place_evidence_frames": (
                place["place_evidence_frames"] if place else 0
            ),
            "final_gripper_joint": float(env._get_obs()["agent_pos"][-1]),
            "next_state": result["next_state"],
            "truth_target_coverage": float(info["target_coverage"]),
            "truth_block_height_m": float(info["block_height"]),
            "truth_success": bool(info["success"]),
        }
        rows.append(row)
        print(row, flush=True)
    env.close()

    summary = {
        "mode": (
            "forced_place_drop" if args.forced_place_drop else "normal"
        ),
        "episodes": len(rows),
        "picks_verified": sum(row["pick_verified"] for row in rows),
        "transports_ready": sum(row["transport_ready"] for row in rows),
        "places_started": sum(row["place_started"] for row in rows),
        "camera_successes": sum(
            row["camera_place_confirmed"] for row in rows
        ),
        "truth_successes": sum(row["truth_success"] for row in rows),
        "false_successes": sum(
            row["camera_place_confirmed"] and not row["truth_success"]
            for row in rows
        ),
        "missed_successes": sum(
            not row["camera_place_confirmed"] and row["truth_success"]
            for row in rows
        ),
        "forced_faults": sum(row["fault_injected"] for row in rows),
        "forced_faults_detected": sum(
            row["fault_injected"] and row["place_drop_detected"]
            for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
