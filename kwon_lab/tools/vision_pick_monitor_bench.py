"""Compare the camera-only missed-pick monitor with MuJoCo truth labels."""

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
from skills.supervision import repick_reason
from skills.vision_supervision import VisionPickMonitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--step", default="act_pick_v22")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path)
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
        monitor = VisionPickMonitor()
        lifted_once = False
        vision_step = truth_step = None
        for step in range(1, 301):
            batch = pre(eval_act.obs_to_batch(obs))
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = np.asarray(post(action).squeeze(0).cpu(), dtype=np.float64)
            obs, _, _, _, info = env.step(action)
            lifted_once = lifted_once or info["block_height"] > 0.08
            _, event = monitor.update(obs["pixels"], obs["agent_pos"], step)
            if event is not None and vision_step is None:
                vision_step = step
            if repick_reason(info, step, lifted_once) is not None and truth_step is None:
                truth_step = step
        row = {
            "seed": seed,
            "vision_missed_pick": vision_step is not None,
            "vision_step": vision_step,
            "truth_missed_pick": truth_step is not None,
            "truth_step": truth_step,
            "vision_grasp_confirmed": monitor.grasp_confirmed,
            "truth_lifted": lifted_once,
        }
        rows.append(row)
        print(row)
    env.close()

    tp = sum(r["vision_missed_pick"] and r["truth_missed_pick"] for r in rows)
    tn = sum(not r["vision_missed_pick"] and not r["truth_missed_pick"] for r in rows)
    fp = sum(r["vision_missed_pick"] and not r["truth_missed_pick"] for r in rows)
    fn = sum(not r["vision_missed_pick"] and r["truth_missed_pick"] for r in rows)
    summary = {"episodes": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "rows": rows}
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
