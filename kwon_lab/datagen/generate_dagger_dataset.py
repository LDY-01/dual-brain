"""ACT 학생의 실패 상태에서 6cm 교사 교정만 수집하는 DAgger 데이터 생성기.

학생은 Mac v2.2 정책을 0.4초 재관측으로 실행한다. 학생이 제시간에 들지
못하거나, 운반 중 떨어뜨리거나, 든 채로 정체하면 현재 시뮬레이터 상태를
그대로 유지한 채 6cm 교사가 인계한다. 저장되는 프레임은 인계 뒤 교사의
관측·행동 쌍뿐이며 실패한 학생 행동은 저장하지 않는다.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from envs.so101_pick_env import SO101PickEnv
from skills.aiming import aim_at
from skills.primitives import GRIPPER_OPEN, move_to, pick, place, set_gripper
from skills.supervision import FALLBACK_NO_LIFT_STEP, repick_reason

from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK = "Pick up the red block and place it on the green target zone."
REPO_ID = "kwonlab/so101_sim_pick_dagger_v1"
DEFAULT_ROOT = Path("outputs/datasets/so101_sim_pick_dagger_v1")
FPS = 25
NO_LIFT_STEPS = FALLBACK_NO_LIFT_STEP
TRANSPORT_STEPS = 200     # 8.0초
MAX_STUDENT_STEPS = 300   # 12.0초
SETTLE_STEPS = 10
LIFTED_HEIGHT = 0.08
DROP_HEIGHT = 0.055
PLACE_NEAR_DISTANCE = 0.08
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
    "observation.state": {
        "dtype": "float32", "shape": (6,), "names": JOINT_NAMES,
    },
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
}


def load_policy(checkpoint: Path, device: str, n_action_steps: int):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"ACT pretrained_model not found: {checkpoint}")
    policy = ACTPolicy.from_pretrained(str(checkpoint))
    if not 1 <= n_action_steps <= policy.config.chunk_size:
        raise ValueError(
            f"n_action_steps must be 1..{policy.config.chunk_size}, got {n_action_steps}"
        )
    policy.config.n_action_steps = n_action_steps
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def obs_to_batch(obs):
    image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).unsqueeze(0)
    return {
        "observation.images.wrist": image.float() / 255.0,
        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0),
    }


def detect_intervention(step, info, lifted_once):
    """실패가 확실하거나 더 기다려도 복구 가능성이 낮은 최초 시점을 찾는다."""
    if lifted_once and info["block_height"] < DROP_HEIGHT and not info["success"]:
        return (
            "misplaced_after_release"
            if info["dist_to_target"] <= PLACE_NEAR_DISTANCE
            else "dropped_during_transport"
        )
    pick_reason = repick_reason(info, step, lifted_once, NO_LIFT_STEPS)
    if pick_reason is not None:
        return pick_reason
    if lifted_once and step >= TRANSPORT_STEPS and not info["success"]:
        return "not_placed_by_8s"
    if step >= MAX_STUDENT_STEPS:
        return "student_timeout"
    return None


def intervention_bucket(reason):
    if reason in {"transport_without_block", "not_lifted_by_deadline"}:
        return "pick"
    if reason == "dropped_during_transport":
        return "transport"
    return "place"


def state_snapshot(env, info):
    return {
        "block_pos": np.asarray(info["block_pos"], dtype=float).tolist(),
        "target_pos": np.asarray(info["target_pos"], dtype=float).tolist(),
        "gripper_pos": np.asarray(info["gripper_pos"], dtype=float).tolist(),
        "block_height_m": float(info["block_height"]),
        "distance_to_target_m": float(info["dist_to_target"]),
        "target_coverage": float(info["target_coverage"]),
        "joint_position": env.data.qpos[:6].astype(float).tolist(),
        "joint_command": env.data.ctrl[:6].astype(float).tolist(),
    }


def teacher_recover(env, info):
    """안전 이탈·재조준을 포함해 현재 학생 상태를 6cm 놓기로 복구한다."""
    diagnostics = {"stages": [], "start": state_snapshot(env, info)}

    # 블록이 충분히 높으면 아직 쥐고 있다고 보고 바로 놓기를 시도한다.
    if info["block_height"] > LIFTED_HEIGHT:
        placed, final = place(env, info["target_pos"][:2])
        diagnostics["stages"].append({
            "stage": "place_held_block", "success": bool(placed),
            "final": state_snapshot(env, final),
        })
        if placed and final["success"]:
            return True, "place_only", final, diagnostics
        info = env._get_info()

    # 학생의 꼬인 자세에서 바로 집지 않는다. 먼저 그리퍼를 열고 현재 XY에서
    # 수직 이탈한 뒤 손목 카메라로 다시 조준해 재접근한다.
    open_info, _ = set_gripper(env, GRIPPER_OPEN, duration=0.35)
    ee_xy = np.asarray(open_info["gripper_pos"][:2], dtype=float)
    retreat_info, ik_error, _ = move_to(
        env, [ee_xy[0], ee_xy[1], SAFE_RETREAT_HEIGHT],
        gripper=GRIPPER_OPEN, duration=0.8,
    )
    diagnostics["stages"].append({
        "stage": "safe_vertical_retreat", "ik_error_m": float(ik_error),
        "state": state_snapshot(env, retreat_info),
    })
    found, centered = aim_at(env, "red_block", attempts=3)
    diagnostics["stages"].append({
        "stage": "reaim", "found": bool(found), "centered": bool(centered),
        "state": state_snapshot(env, env._get_info()),
    })
    if not found:
        final = env._get_info()
        diagnostics["failure_stage"] = "reaim"
        return False, "safe_reaim_repick_place", final, diagnostics

    current = env._get_info()
    grasped, current = pick(env, current["block_pos"])
    diagnostics["stages"].append({
        "stage": "repick", "success": bool(grasped),
        "state": state_snapshot(env, current),
    })
    if not grasped:
        diagnostics["failure_stage"] = "repick"
        return False, "safe_reaim_repick_place", current, diagnostics
    placed, final = place(env, current["target_pos"][:2])
    diagnostics["stages"].append({
        "stage": "place_6cm", "success": bool(placed),
        "state": state_snapshot(env, final),
    })
    if not (placed and final["success"]):
        diagnostics["failure_stage"] = "place_6cm"
    return (
        bool(placed and final["success"]), "safe_reaim_repick_place", final,
        diagnostics,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="?", default=10)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="v2.2 pretrained_model directory")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=7000)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--pick-quota", type=int)
    ap.add_argument("--transport-quota", type=int)
    ap.add_argument("--place-quota", type=int)
    args = ap.parse_args()

    root = args.root.resolve()
    checkpoint = args.checkpoint.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {root}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    quota_values = [args.pick_quota, args.transport_quota, args.place_quota]
    quotas = None
    if any(value is not None for value in quota_values):
        if any(value is None or value < 0 for value in quota_values):
            raise ValueError("all three quotas must be non-negative integers")
        quotas = dict(zip(("pick", "transport", "place"), quota_values))
        if sum(quotas.values()) != args.episodes:
            raise ValueError(
                f"quota sum {sum(quotas.values())} must equal episodes {args.episodes}"
            )
    policy, pre, post = load_policy(checkpoint, device, args.n_action_steps)
    env = SO101PickEnv(camera="wrist")
    dataset = None

    saved = attempts = student_successes = teacher_failures = total_frames = 0
    reasons = Counter()
    buckets = Counter()
    recoveries = Counter()
    episodes = []
    attempt_diagnostics = []
    t0 = time.time()

    while saved < args.episodes and attempts < args.max_attempts:
        seed = args.seed_start + attempts
        attempts += 1
        obs, info = env.reset(seed=seed)
        found, _ = aim_at(env, "red_block")
        if not found:
            print(f"seed {seed}: 조준 실패")
            continue
        obs = env._get_obs()
        policy.reset()
        lifted_once = False
        settled = 0
        reason = None
        intervention_step = None

        # recorder=None: 학생 행동은 의도적으로 저장하지 않는다.
        env.recorder = None
        for step in range(1, MAX_STUDENT_STEPS + 1):
            batch = pre(obs_to_batch(obs))
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = post(action)
            action = np.asarray(action.squeeze(0).cpu(), dtype=np.float64)
            obs, _, _, _, info = env.step(action)
            lifted_once = lifted_once or info["block_height"] > LIFTED_HEIGHT
            settled = settled + 1 if info["success"] else 0
            if settled >= SETTLE_STEPS:
                student_successes += 1
                break
            reason = detect_intervention(step, info, lifted_once)
            if reason:
                intervention_step = step
                break

        if settled >= SETTLE_STEPS:
            attempt_diagnostics.append({
                "seed": seed, "outcome": "student_success_not_saved",
                "student_steps": step,
            })
            print(f"seed {seed}: 학생 자체 성공 - 저장 안 함")
            continue
        if reason is None:
            reason = "student_timeout"
            intervention_step = MAX_STUDENT_STEPS
        bucket = intervention_bucket(reason)
        snapshot = state_snapshot(env, info)
        if quotas is not None and buckets[bucket] >= quotas[bucket]:
            attempt_diagnostics.append({
                "seed": seed, "outcome": "quota_full_not_saved",
                "intervention_reason": reason, "bucket": bucket,
                "intervention_step": intervention_step, "state": snapshot,
            })
            print(f"seed {seed}: {bucket} 정원 완료 - 저장 안 함")
            continue

        # 마지막 학생 관측이 첫 교사 action과 정확히 짝지어진다.
        env._last_obs = obs
        env.recorder = []
        recovered, recovery_type, final, recovery_diagnostics = teacher_recover(env, info)
        correction_frames = list(env.recorder)
        env.recorder = None
        if not recovered:
            teacher_failures += 1
            attempt_diagnostics.append({
                "seed": seed, "outcome": "teacher_failure_not_saved",
                "intervention_reason": reason, "bucket": bucket,
                "intervention_step": intervention_step, "state": snapshot,
                "recovery": recovery_diagnostics,
            })
            print(
                f"seed {seed}: {reason} @ {intervention_step} -> "
                f"교사 복구 실패 ({recovery_type})"
            )
            continue

        # 첫 교정 성공 전에는 출력 폴더를 만들지 않는다. 설정 오류나 교사
        # 복구 실패 때문에 빈 데이터셋 디렉터리만 남는 일을 방지한다.
        if dataset is None:
            dataset = LeRobotDataset.create(
                REPO_ID, fps=FPS, features=FEATURES, root=root,
                robot_type="so101_sim"
            )
        for frame in correction_frames:
            dataset.add_frame({
                "observation.images.wrist": frame["pixels"],
                "observation.state": frame["state"].astype(np.float32),
                "action": frame["action"].astype(np.float32),
                "task": TASK,
            })
        dataset.save_episode()
        reasons[reason] += 1
        buckets[bucket] += 1
        recoveries[recovery_type] += 1
        total_frames += len(correction_frames)
        episodes.append({
            "episode_index": saved,
            "seed": seed,
            "intervention_reason": reason,
            "intervention_bucket": bucket,
            "intervention_step": intervention_step,
            "intervention_seconds": intervention_step / FPS,
            "recovery_type": recovery_type,
            "correction_frames": len(correction_frames),
            "final_distance_m": float(final["dist_to_target"]),
            "final_coverage": float(final["target_coverage"]),
            "final_block_height_m": float(final["block_height"]),
            "intervention_state": snapshot,
            "recovery_diagnostics": recovery_diagnostics,
        })
        attempt_diagnostics.append({
            "seed": seed, "outcome": "saved_teacher_correction",
            "episode_index": saved, "intervention_reason": reason,
            "bucket": bucket, "intervention_step": intervention_step,
        })
        saved += 1
        print(
            f"교정 {saved}/{args.episodes}: seed {seed}, {reason} "
            f"@ {intervention_step / FPS:.1f}s, {recovery_type}, "
            f"{len(correction_frames)}프레임"
        )

    env.close()
    complete = saved == args.episodes
    manifest = {
        "format": "dagger_teacher_corrections_only",
        "complete": complete,
        "student_checkpoint": str(checkpoint),
        "student_n_action_steps": args.n_action_steps,
        "student_reobservation_seconds": args.n_action_steps / FPS,
        "teacher_release_height_m": 0.06,
        "success_coverage_threshold": 0.75,
        "seed_start": args.seed_start,
        "attempts": attempts,
        "student_successes_not_saved": student_successes,
        "teacher_failures_not_saved": teacher_failures,
        "quotas": quotas,
        "saved_correction_episodes": saved,
        "saved_frames": total_frames,
        "intervention_reasons": dict(reasons),
        "intervention_buckets": dict(buckets),
        "recovery_types": dict(recoveries),
        "episodes": episodes,
        "attempt_diagnostics": attempt_diagnostics,
    }
    (root / "dagger_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    elapsed = time.time() - t0
    print(
        f"완료: 교정 {saved}개/{attempts}시도, {total_frames}프레임, "
        f"학생 자체 성공 {student_successes}개 제외, 교사 실패 {teacher_failures}개 제외, "
        f"{elapsed / 60:.1f}분\n데이터셋: {root}"
    )
    if not complete:
        raise RuntimeError(
            f"Only collected {saved}/{args.episodes} successful corrections in "
            f"{attempts} attempts; diagnostics were saved to {root / 'dagger_manifest.json'}"
        )


if __name__ == "__main__":
    main()
