"""Inject closed-gripper transport slips and validate the dual-camera monitor.

Privileged block state is used only by this test harness to create repeatable
positive examples. The monitor itself still receives only two RGB streams and
joint state.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np
import torch

from envs.so101_pick_env import SO101PickEnv
from eval import eval_act_pick as eval_act
from skills.aiming import aim_at
from skills.vision_supervision import DualCameraTaskMonitor


def inject_slip(env):
    """Move the held block just outside the fingers without opening them."""
    addr = env.block_qpos_addr
    block = env.data.qpos[addr : addr + 7].copy()
    block[1] += 0.055
    block[2] = max(0.075, block[2] - 0.025)
    env.data.qpos[addr : addr + 7] = block
    env.data.qvel[env.model.jnt_dofadr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "block_free")
    ] :][:6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", default="act_pick_v22")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    eval_act.IMAGE_KEY = "observation.images.wrist"
    policy, pre, post = eval_act.load_policy(args.step, args.device, args.checkpoint_root)
    policy.config.n_action_steps = 10
    env = SO101PickEnv(camera="wrist")
    rows = []
    for index in range(args.episodes):
        seed = args.start_seed + index
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        obs, _ = env.reset(seed=seed)
        monitor = DualCameraTaskMonitor(env.render_overhead())
        aim_at(env, "red_block")
        obs = env._get_obs()
        policy.reset()
        injected_step = detected_step = landed_step = None
        recovery_step = None
        recovery_reaimed = None
        recovery_pick_retries = 0
        attempt_start_step = 0
        success_steps = 0
        eventual_success = False
        low_steps = 0
        detached_step = stationary_step = None
        max_drop_evidence = 0
        landed_snapshot = None
        for step in range(1, args.max_steps + 1):
            with torch.inference_mode():
                action = policy.select_action(pre(eval_act.obs_to_batch(obs)))
            action = np.asarray(post(action).squeeze(0).cpu(), dtype=np.float64)
            obs, _, _, _, info = env.step(action)
            signals = monitor.update(
                obs["pixels"], env.render_overhead(), obs["agent_pos"],
                step - attempt_start_step,
            )
            if signals["wrist_detached"] and detached_step is None:
                detached_step = step
            if signals["overhead_stationary"] and stationary_step is None:
                stationary_step = step
            max_drop_evidence = max(
                max_drop_evidence, signals["drop_evidence_frames"]
            )
            if (
                injected_step is None
                and signals["dual_grasp_confirmed"]
            ):
                inject_slip(env)
                obs = env._get_obs()
                injected_step = step
                policy.reset()
                continue
            if injected_step is not None:
                low_steps = low_steps + 1 if info["block_height"] < 0.055 else 0
                if low_steps >= eval_act.DROP_CONFIRM_STEPS and landed_step is None:
                    landed_step = step
                    landed_snapshot = {
                        "wrist_detached": signals["wrist_detached"],
                        "wrist_offset_px": round(signals["wrist_offset_px"], 2),
                        "overhead_stationary": signals["overhead_stationary"],
                        "overhead_spread_px": round(
                            signals["overhead_spread_px"], 2
                        ),
                        "arm_motion": round(signals["arm_motion"], 4),
                        "gripper": round(float(obs["agent_pos"][-1]), 4),
                        "image_target_coverage": round(
                            signals["image_target_coverage"], 4
                        ),
                    }
                if signals["transport_drop"] and detected_step is None:
                    detected_step = step
                    if args.recover:
                        recovery_step = step
                        policy.reset()
                        recovery_reaimed = eval_act.vision_reaim(env, [])
                        obs = env._get_obs()
                        monitor = DualCameraTaskMonitor(env.render_overhead())
                        attempt_start_step = step
                        continue
            if (
                args.recover
                and recovery_step is not None
                and signals["pick_event"] is not None
                and recovery_pick_retries < 3
            ):
                recovery_pick_retries += 1
                policy.reset()
                reaimed = eval_act.vision_reaim(env, [])
                recovery_reaimed = bool(recovery_reaimed or reaimed)
                obs = env._get_obs()
                monitor = DualCameraTaskMonitor(env.render_overhead())
                attempt_start_step = step
                continue
            success_steps = success_steps + 1 if info["success"] else 0
            if success_steps >= eval_act.SETTLE_STEPS:
                eventual_success = True
                break
        row = {
            "seed": seed,
            "injected": injected_step is not None,
            "injected_step": injected_step,
            "truth_landed": landed_step is not None,
            "truth_landed_step": landed_step,
            "vision_transport_drop": detected_step is not None,
            "vision_detected_step": detected_step,
            "latency_steps": (
                detected_step - landed_step
                if detected_step is not None and landed_step is not None else None
            ),
            "wrist_detached_step": detached_step,
            "overhead_stationary_step": stationary_step,
            "max_drop_evidence": max_drop_evidence,
            "landed_snapshot": landed_snapshot,
            "recovery_step": recovery_step,
            "recovery_reaimed": recovery_reaimed,
            "recovery_pick_retries": recovery_pick_retries,
            "eventual_success": eventual_success,
        }
        rows.append(row)
        print(row, flush=True)
    env.close()

    positives = [row for row in rows if row["truth_landed"]]
    summary = {
        "episodes": len(rows),
        "injected": sum(row["injected"] for row in rows),
        "landed": len(positives),
        "detected": sum(row["vision_transport_drop"] for row in positives),
        "missed": sum(not row["vision_transport_drop"] for row in positives),
        "recovery_attempts": sum(row["recovery_step"] is not None for row in rows),
        "reaim_successes": sum(row["recovery_reaimed"] is True for row in rows),
        "eventual_successes": sum(row["eventual_success"] for row in rows),
        "false_before_landing": sum(
            row["vision_detected_step"] is not None
            and (
                row["truth_landed_step"] is None
                or row["vision_detected_step"] < row["truth_landed_step"]
            )
            for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
