"""Validate fused wrist/overhead task signals against MuJoCo truth."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from envs.so101_pick_env import SO101PickEnv
from eval import eval_act_pick as eval_act
from skills.aiming import aim_at
from skills.vision_supervision import DualCameraTaskMonitor


def confusion(rows, pred, truth):
    return {
        "tp": sum(bool(row[pred]) and bool(row[truth]) for row in rows),
        "tn": sum(not row[pred] and not row[truth] for row in rows),
        "fp": sum(bool(row[pred]) and not row[truth] for row in rows),
        "fn": sum(not row[pred] and bool(row[truth]) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", default="act_pick_v22")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
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
        lifted = transport_drop = failed_release = success = False
        closed_low_steps = open_low_steps = success_steps = 0
        vision_drop_step = vision_failed_release_step = vision_place_step = None
        max_offset = max_overhead_spread = 0.0
        detached_seen = stationary_seen = False
        truth_drop_step = truth_failed_release_step = detached_step = stationary_step = None
        max_drop_evidence = 0
        max_failed_release_evidence = 0
        closest_drop_candidate = None
        truth_drop_snapshot = None
        for step in range(1, 301):
            with torch.inference_mode():
                action = policy.select_action(pre(eval_act.obs_to_batch(obs)))
            action = np.asarray(post(action).squeeze(0).cpu(), dtype=np.float64)
            obs, _, _, _, info = env.step(action)
            lifted = lifted or info["block_height"] > 0.08
            low_after_lift = lifted and info["block_height"] < 0.055 and not info["success"]
            gripper = float(obs["agent_pos"][-1])
            closed_low = low_after_lift and gripper <= 0.65
            open_low = low_after_lift and gripper >= 1.25
            closed_low_steps = closed_low_steps + 1 if closed_low else 0
            open_low_steps = open_low_steps + 1 if open_low else 0
            if closed_low_steps >= eval_act.DROP_CONFIRM_STEPS and truth_drop_step is None:
                truth_drop_step = step
            if open_low_steps >= eval_act.DROP_CONFIRM_STEPS and truth_failed_release_step is None:
                truth_failed_release_step = step
            transport_drop = transport_drop or truth_drop_step is not None
            failed_release = failed_release or truth_failed_release_step is not None
            success_steps = success_steps + 1 if info["success"] else 0
            success = success or success_steps >= eval_act.SETTLE_STEPS

            signals = monitor.update(
                obs["pixels"], env.render_overhead(), obs["agent_pos"], step
            )
            max_offset = max(max_offset, signals["wrist_offset_px"])
            max_overhead_spread = max(
                max_overhead_spread, signals["overhead_spread_px"]
            )
            detached_seen = detached_seen or signals["wrist_detached"]
            stationary_seen = stationary_seen or signals["overhead_stationary"]
            if signals["wrist_detached"] and detached_step is None:
                detached_step = step
            if signals["overhead_stationary"] and stationary_step is None:
                stationary_step = step
            if signals["drop_evidence_frames"] > max_drop_evidence:
                max_drop_evidence = signals["drop_evidence_frames"]
                closest_drop_candidate = {
                    "step": step,
                    "wrist_detached": signals["wrist_detached"],
                    "overhead_stationary": signals["overhead_stationary"],
                    "gripper": round(float(obs["agent_pos"][-1]), 4),
                    "image_target_coverage": round(
                        signals["image_target_coverage"], 4
                    ),
                    "overhead_spread_px": round(signals["overhead_spread_px"], 2),
                    "arm_motion": round(signals["arm_motion"], 4),
                }
            max_failed_release_evidence = max(
                max_failed_release_evidence,
                signals["failed_release_evidence_frames"],
            )
            if step == truth_drop_step:
                truth_drop_snapshot = {
                    "wrist_detached": signals["wrist_detached"],
                    "wrist_offset_px": round(signals["wrist_offset_px"], 2),
                    "overhead_stationary": signals["overhead_stationary"],
                    "overhead_spread_px": round(signals["overhead_spread_px"], 2),
                    "arm_motion": round(signals["arm_motion"], 4),
                    "gripper": round(float(obs["agent_pos"][-1]), 4),
                    "image_target_coverage": round(
                        signals["image_target_coverage"], 4
                    ),
                }
            if signals["transport_drop"] and vision_drop_step is None:
                vision_drop_step = step
            if signals["failed_release"] and vision_failed_release_step is None:
                vision_failed_release_step = step
            if signals["possible_place"] and vision_place_step is None:
                vision_place_step = step

        row = {
            "seed": seed,
            "truth_lifted": lifted,
            "vision_grasp": monitor.pick.grasp_confirmed,
            "vision_dual_grasp": monitor.was_wrist_attached,
            "truth_drop": transport_drop,
            "truth_drop_step": truth_drop_step,
            "vision_drop": vision_drop_step is not None,
            "vision_drop_step": vision_drop_step,
            "truth_failed_release": failed_release,
            "truth_failed_release_step": truth_failed_release_step,
            "vision_failed_release": vision_failed_release_step is not None,
            "vision_failed_release_step": vision_failed_release_step,
            "truth_recovery": transport_drop or failed_release,
            "vision_recovery": (
                vision_drop_step is not None or vision_failed_release_step is not None
            ),
            "truth_success": success,
            "vision_place": vision_place_step is not None,
            "vision_place_step": vision_place_step,
            "wrist_detached_seen": detached_seen,
            "wrist_detached_step": detached_step,
            "overhead_stationary_seen": stationary_seen,
            "overhead_stationary_step": stationary_step,
            "max_drop_evidence": max_drop_evidence,
            "max_failed_release_evidence": max_failed_release_evidence,
            "closest_drop_candidate": closest_drop_candidate,
            "truth_drop_snapshot": truth_drop_snapshot,
            "max_wrist_offset_px": round(max_offset, 2),
            "max_overhead_spread_px": round(max_overhead_spread, 2),
        }
        rows.append(row)
        print(row, flush=True)
    env.close()

    summary = {
        "episodes": len(rows),
        "grasp": confusion(rows, "vision_grasp", "truth_lifted"),
        "dual_grasp": confusion(rows, "vision_dual_grasp", "truth_lifted"),
        "drop": confusion(rows, "vision_drop", "truth_drop"),
        "failed_release": confusion(
            rows, "vision_failed_release", "truth_failed_release"
        ),
        "recovery": confusion(rows, "vision_recovery", "truth_recovery"),
        "place": confusion(rows, "vision_place", "truth_success"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
