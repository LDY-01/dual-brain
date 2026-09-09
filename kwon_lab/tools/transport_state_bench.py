"""Validate camera-gated PICK -> TRANSPORT and transport-drop aborts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np

from envs.so101_pick_env import BLOCK_HALF_H, SO101PickEnv
from skills.pick_place_state_machine import pick_then_transport


POSITIONS = (
    (0.18, -0.10),
    (0.23, 0.00),
    (0.28, 0.10),
    (0.18, 0.10),
    (0.28, -0.10),
)


def inject_slip(env):
    """Test-only fault injection; production supervision cannot access this."""
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
    parser.add_argument("--forced-drop", action="store_true")
    parser.add_argument("--inject-step", type=int, default=3)
    args = parser.parse_args()

    env = SO101PickEnv(camera="wrist")
    rows = []
    for index in range(args.episodes):
        block_xy = POSITIONS[index % len(POSITIONS)]
        env.reset(seed=9100 + index)
        addr = env.block_qpos_addr
        env.data.qpos[addr : addr + 3] = [*block_xy, BLOCK_HALF_H]
        mujoco.mj_forward(env.model, env.data)
        injected = False

        def maybe_inject(current_env, step, _signals):
            nonlocal injected
            if args.forced_drop and not injected and step == args.inject_step:
                inject_slip(current_env)
                injected = True

        result = pick_then_transport(
            env,
            on_transport_step=maybe_inject,
        )
        info = env._get_info()
        target_truth = info["target_pos"][:2]
        target_estimate = result["target_xy"]
        pick_success = bool(result["pick"] and result["pick"]["success"])
        transport = result["transport"]
        row = {
            "index": index,
            "block_xy": block_xy,
            "forced_drop": bool(args.forced_drop),
            "fault_injected": injected,
            "target_visible": result["target_visible"],
            "target_localization_error_mm": (
                None
                if target_estimate is None
                else float(
                    np.linalg.norm(np.asarray(target_estimate) - target_truth)
                    * 1000
                )
            ),
            "pick_verified": pick_success,
            "pick_attempts": (
                result["pick"]["attempts"] if result["pick"] else 0
            ),
            "transport_started": transport is not None,
            "drop_detected": (
                transport["drop_detected"] if transport else False
            ),
            "drop_detected_step": (
                transport["drop_detected_step"] if transport else None
            ),
            "ready_for_place": (
                transport["ready_for_place"] if transport else False
            ),
            "next_state": result["next_state"],
            "truth_block_height_m": float(info["block_height"]),
            "truth_distance_to_target_m": float(info["dist_to_target"]),
        }
        rows.append(row)
        print(row, flush=True)
    env.close()

    started = [row for row in rows if row["transport_started"]]
    summary = {
        "mode": "forced_drop" if args.forced_drop else "normal",
        "episodes": len(rows),
        "picks_verified": sum(row["pick_verified"] for row in rows),
        "transports_started": len(started),
        "ready_for_place": sum(row["ready_for_place"] for row in rows),
        "drop_detections": sum(row["drop_detected"] for row in rows),
        "false_drop_aborts": (
            0
            if args.forced_drop
            else sum(row["drop_detected"] for row in rows)
        ),
        "forced_faults": sum(row["fault_injected"] for row in rows),
        "forced_faults_detected": sum(
            row["fault_injected"] and row["drop_detected"] for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
