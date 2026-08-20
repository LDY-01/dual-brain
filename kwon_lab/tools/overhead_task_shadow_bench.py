"""Validate fixed-overhead RGB task signals against MuJoCo truth labels."""

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
from skills.vision_supervision import OverheadTaskMonitor, VisionPickMonitor


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eval_act.IMAGE_KEY = "observation.images.wrist"
    policy, pre, post = eval_act.load_policy(args.step, args.device, args.checkpoint_root)
    policy.config.n_action_steps = 10
    env = SO101PickEnv(camera="wrist")
    rows = []
    for index in range(args.episodes):
        seed = 5000 + index
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        obs, _ = env.reset(seed=seed)
        overhead = OverheadTaskMonitor(env.render_overhead())
        aim_at(env, "red_block")
        obs = env._get_obs()
        policy.reset()
        pick = VisionPickMonitor()
        lifted = dropped = success = False
        low_steps = success_steps = 0
        vision_drop_step = vision_place_step = None
        max_image_coverage = max_truth_coverage = 0.0
        max_coverage_area_ratio = max_coverage_gripper = 0.0
        for step in range(1, 301):
            with torch.inference_mode():
                action = policy.select_action(pre(eval_act.obs_to_batch(obs)))
            action = np.asarray(post(action).squeeze(0).cpu(), dtype=np.float64)
            obs, _, _, _, info = env.step(action)
            lifted = lifted or info["block_height"] > 0.08
            low_after_lift = lifted and info["block_height"] < 0.055 and not info["success"]
            low_steps = low_steps + 1 if low_after_lift else 0
            dropped = dropped or low_steps >= eval_act.DROP_CONFIRM_STEPS
            success_steps = success_steps + 1 if info["success"] else 0
            success = success or success_steps >= eval_act.SETTLE_STEPS

            pick.update(obs["pixels"], obs["agent_pos"], step)
            _, signals = overhead.update(
                env.render_overhead(), obs["agent_pos"], pick.grasp_confirmed
            )
            if signals["image_target_coverage"] > max_image_coverage:
                max_image_coverage = signals["image_target_coverage"]
                max_coverage_area_ratio = signals["table_area_ratio"]
                max_coverage_gripper = float(obs["agent_pos"][-1])
            max_truth_coverage = max(max_truth_coverage, info["target_coverage"])
            if signals["possible_drop"] and vision_drop_step is None:
                vision_drop_step = step
            if signals["possible_place"] and vision_place_step is None:
                vision_place_step = step

        row = {
            "seed": seed,
            "truth_lifted": lifted,
            "vision_grasp": pick.grasp_confirmed,
            "truth_drop": dropped,
            "vision_drop": vision_drop_step is not None,
            "vision_drop_step": vision_drop_step,
            "truth_success": success,
            "vision_place": vision_place_step is not None,
            "vision_place_step": vision_place_step,
            "max_truth_coverage": round(max_truth_coverage, 4),
            "max_image_coverage": round(max_image_coverage, 4),
            "max_coverage_area_ratio": round(max_coverage_area_ratio, 4),
            "max_coverage_gripper": round(max_coverage_gripper, 4),
        }
        rows.append(row)
        print(row, flush=True)
    env.close()

    summary = {
        "episodes": len(rows),
        "grasp": confusion(rows, "vision_grasp", "truth_lifted"),
        "drop": confusion(rows, "vision_drop", "truth_drop"),
        "place": confusion(rows, "vision_place", "truth_success"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
