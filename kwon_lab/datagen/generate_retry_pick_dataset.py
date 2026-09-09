"""Collect recovery-only demonstrations after camera-detected missed picks.

The intervention decision uses wrist RGB and joint state only. MuJoCo object
poses are available solely to the offline teacher that produces labels; the
resulting recovery policy receives only the same RGB + joints as the real arm.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from envs.so101_pick_env import SO101PickEnv
from eval import eval_act_pick as eval_act
from skills.aiming import aim_at
from skills.primitives import GRIPPER_OPEN, move_to, pick, set_gripper
from skills.vision_supervision import VisionPickMonitor


FPS = 25
REPO_ID = "kwonlab/so101_sim_retry_pick_v1"
TASK = "Recover from a missed pick and lift the red block."
SAFE_RETREAT_HEIGHT = 0.20
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
FEATURES = {
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
}


def teacher_repick_only(env):
    """Offline privileged teacher: recover to a verified lift, without placing."""
    info, _ = set_gripper(env, GRIPPER_OPEN, duration=0.35)
    ee_xy = np.asarray(info["gripper_pos"][:2], dtype=float)
    move_to(
        env, [ee_xy[0], ee_xy[1], SAFE_RETREAT_HEIGHT],
        gripper=GRIPPER_OPEN, duration=0.8,
    )
    found, centered = aim_at(env, "red_block", attempts=3)
    if not found:
        return False, env._get_info(), {"found": False, "centered": bool(centered)}
    info = env._get_info()
    grasped, final = pick(env, info["block_pos"])
    return bool(grasped), final, {"found": True, "centered": bool(centered)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="?", default=20)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--step", default="act_pick_v22")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-start", type=int, default=9000)
    ap.add_argument("--max-attempts", type=int, default=300)
    args = ap.parse_args()

    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite retry dataset: {root}")
    eval_act.IMAGE_KEY = "observation.images.wrist"
    policy, pre, post = eval_act.load_policy(args.step, args.device, args.checkpoint_root)
    policy.config.n_action_steps = 10
    dataset = None
    saved = attempts = teacher_failures = total_frames = 0
    episode_rows = []
    started = time.time()
    env = SO101PickEnv(camera="wrist")

    while saved < args.episodes and attempts < args.max_attempts:
        seed = args.seed_start + attempts
        attempts += 1
        obs, _ = env.reset(seed=seed)
        found, _ = aim_at(env, "red_block")
        if not found:
            print(f"seed {seed}: block not found during camera aim")
            continue
        obs = env._get_obs()
        policy.reset()
        monitor = VisionPickMonitor()
        event = None
        event_step = None
        env.recorder = None
        for step in range(1, 301):
            with torch.inference_mode():
                action = policy.select_action(pre(eval_act.obs_to_batch(obs)))
            action = np.asarray(post(action).squeeze(0).cpu(), dtype=np.float64)
            obs, _, _, _, _ = env.step(action)
            _, event = monitor.update(obs["pixels"], obs["agent_pos"], step)
            if event is not None:
                event_step = step
                break
        if event is None:
            print(f"seed {seed}: no camera-detected missed pick")
            continue

        env._last_obs = obs
        env.recorder = []
        grasped, final, diagnostics = teacher_repick_only(env)
        correction = list(env.recorder)
        env.recorder = None
        if not grasped:
            teacher_failures += 1
            print(f"seed {seed}: teacher re-pick failed")
            continue

        if dataset is None:
            dataset = LeRobotDataset.create(
                REPO_ID, fps=FPS, features=FEATURES, root=root,
                robot_type="so101_sim",
            )
        for frame in correction:
            dataset.add_frame({
                "observation.images.wrist": frame["pixels"],
                "observation.state": frame["state"].astype(np.float32),
                "action": frame["action"].astype(np.float32),
                "task": TASK,
            })
        dataset.save_episode()
        episode_rows.append({
            "episode_index": saved,
            "seed": seed,
            "vision_event": event,
            "event_step": event_step,
            "frames": len(correction),
            "final_block_height_m": float(final["block_height"]),
            "diagnostics": diagnostics,
        })
        saved += 1
        total_frames += len(correction)
        print(f"saved {saved}/{args.episodes}: seed {seed}, {len(correction)} frames")

    env.close()
    manifest = {
        "format": "vision_triggered_retry_pick_only",
        "complete": saved == args.episodes,
        "episodes": saved,
        "attempts": attempts,
        "teacher_failures": teacher_failures,
        "frames": total_frames,
        "intervention_inputs": ["observation.images.wrist", "observation.state"],
        "offline_teacher_uses_privileged_sim_state": True,
        "deployment_policy_uses_privileged_sim_state": False,
        "elapsed_seconds": time.time() - started,
        "episode_rows": episode_rows,
    }
    if dataset is not None:
        (root / "retry_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if saved != args.episodes:
        raise RuntimeError(f"Collected only {saved}/{args.episodes} retry episodes")


if __name__ == "__main__":
    main()
