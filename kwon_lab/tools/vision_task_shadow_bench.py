"""Compare camera-only task shadow signals against MuJoCo truth."""

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
from skills.vision_supervision import VisionTaskShadow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--step", default="act_pick_v22")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    eval_act.IMAGE_KEY = "observation.images.wrist"
    policy, pre, post = eval_act.load_policy(args.step, args.device, args.checkpoint_root)
    policy.config.n_action_steps = 10
    rows = []
    env = SO101PickEnv(camera="wrist")
    for index in range(args.episodes):
        seed = 5000 + index
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        obs, info = env.reset(seed=seed)
        aim_at(env, "red_block")
        obs = env._get_obs()
        policy.reset()
        shadow = VisionTaskShadow()
        lifted = dropped = success = False
        low_after_lift_steps = success_steps = 0
        first_truth_drop = first_vision_drop = first_vision_place = None
        for step in range(1, 301):
            with torch.inference_mode():
                action = policy.select_action(pre(eval_act.obs_to_batch(obs)))
            action = np.asarray(post(action).squeeze(0).cpu(), dtype=np.float64)
            obs, _, _, _, info = env.step(action)
            lifted = lifted or info["block_height"] > 0.08
            low_after_lift = (
                lifted and info["block_height"] < 0.055 and not info["success"]
            )
            low_after_lift_steps = low_after_lift_steps + 1 if low_after_lift else 0
            if low_after_lift_steps >= eval_act.DROP_CONFIRM_STEPS:
                dropped = True
                if first_truth_drop is None:
                    first_truth_drop = step - eval_act.DROP_CONFIRM_STEPS + 1
            _, signals = shadow.update(obs["pixels"], obs["agent_pos"], step)
            if signals["possible_drop"] and first_vision_drop is None:
                first_vision_drop = step
            if signals["possible_place"] and first_vision_place is None:
                first_vision_place = step
            success_steps = success_steps + 1 if info["success"] else 0
            success = success or success_steps >= eval_act.SETTLE_STEPS
        row = {
            "seed": seed,
            "truth_lifted": lifted,
            "vision_grasp": shadow.pick.grasp_confirmed,
            "truth_drop": dropped,
            "truth_drop_step": first_truth_drop,
            "vision_drop": first_vision_drop is not None,
            "vision_drop_step": first_vision_drop,
            "truth_success": success,
            "vision_place": first_vision_place is not None,
            "vision_place_step": first_vision_place,
        }
        rows.append(row)
        print(row)
    env.close()

    def confusion(pred, truth):
        return {
            "tp": sum(bool(r[pred]) and bool(r[truth]) for r in rows),
            "tn": sum(not r[pred] and not r[truth] for r in rows),
            "fp": sum(bool(r[pred]) and not r[truth] for r in rows),
            "fn": sum(not r[pred] and bool(r[truth]) for r in rows),
        }

    summary = {
        "episodes": len(rows),
        "grasp": confusion("vision_grasp", "truth_lifted"),
        "drop": confusion("vision_drop", "truth_drop"),
        "place": confusion("vision_place", "truth_success"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
