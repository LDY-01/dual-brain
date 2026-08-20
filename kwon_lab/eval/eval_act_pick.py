"""S5: 학습된 ACT(System 1) 평가 — 시뮬 롤아웃으로 성공률 측정.

선생(LLM 레시피)이 만든 데이터로 학습한 정책이 스스로 얼마나 해내는지 확인한다.
System 2(LLM 직접 제어)와의 비교가 이 프로젝트의 핵심 증거물.

- 관측 키를 학습 데이터셋과 동일하게 구성 (observation.images.front / observation.state)
- 전처리 파이프라인(정규화 통계)은 체크포인트에서 로드 — 학습과 똑같이 적용
- 평가 시드는 5000번대: 학습 데이터(1000번대)에 없던 블록 위치만 사용
- 성공 판정: 블록 바닥 면적 75% 이상이 목표 구역 안 + 테이블 위 + 10스텝 연속 유지

실행:  .venv/bin/python kwon_lab/eval/eval_act_pick.py [--step 015000] [--episodes 20]
출력:  콘솔 성공률 표 + outputs/eval/act_pick_v1/<step>/ 롤아웃 영상(처음 3개+실패 3개)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from envs.so101_pick_env import SO101PickEnv
from skills.supervision import FALLBACK_NO_LIFT_STEP, LIFTED_HEIGHT, repick_reason
from skills.vision_supervision import DualCameraTaskMonitor, VisionPickMonitor

REPO_ROOT = Path(__file__).resolve().parents[2]
# --version 인자로 v1(front·구관측)/v2(wrist·에고센트릭) 선택
VERSIONS = {
    "v1": ("outputs/train/act_pick_v1/checkpoints", "outputs/eval/act_pick_v1",
           "observation.images.front", "front"),
    "v2": ("outputs/train/act_pick_v2/checkpoints", "outputs/eval/act_pick_v2",
           "observation.images.wrist", "wrist"),
    # v2.1: 조준 후 시작 (aim_at으로 블록을 화면 중앙에 놓고 정책 인계 — 학습 분포와 일치)
    "v2.1": ("outputs/train/act_pick_v21/checkpoints", "outputs/eval/act_pick_v21",
             "observation.images.wrist", "wrist"),
    "v2.2": ("outputs/train/act_pick_v22/checkpoints", "outputs/eval/act_pick_v22",
             "observation.images.wrist", "wrist"),  # v2.1 + 에피소드 200개
    "v2.3": ("outputs/train/act_pick_v23/checkpoints", "outputs/eval/act_pick_v23",
             "observation.images.wrist", "wrist"),  # 6cm 놓기 + 집기·운반 개선
    "v2.4": ("outputs/train/act_pick_v24_ft/checkpoints", "outputs/eval/act_pick_v24_ft",
             "observation.images.wrist", "wrist"),  # v2.2 초기값 + 균등분포 6cm 시연 200개
}
AIM_START = {"v2.1", "v2.2", "v2.3", "v2.4"}
MAX_STEPS = 300          # 25Hz × 12초 — 레시피 시연(~7초)보다 넉넉하게
SETTLE_STEPS = 10        # 성공 상태가 이만큼 연속 유지돼야 인정 (스쳐 지나감 방지)

DEFAULT_PICK_VERIFY_STEPS = FALLBACK_NO_LIFT_STEP
SAFE_RETREAT_HEIGHT = 0.20
DROP_HEIGHT = 0.055
DROP_CONFIRM_STEPS = 10


def load_policy(step: str, device: str, ckpt_root):
    path = Path(ckpt_root) / step / "pretrained_model"
    if not path.is_dir():
        raise FileNotFoundError(
            f"ACT checkpoint not found: {path}. "
            "Train a policy first or pass the matching checkpoint root."
        )
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(path))
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


IMAGE_KEY = "observation.images.front"  # main()에서 버전에 따라 설정


def obs_to_batch(obs) -> dict:
    """환경 관측 → 학습 데이터셋과 동일한 키·형식의 배치 (batch=1)."""
    img = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return {
        IMAGE_KEY: img.float() / 255.0,
        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0),
    }


def teacher_repick(env, frames):
    """Safely re-aim and retry only the pick stage from the current sim state."""
    from skills.aiming import aim_at
    from skills.primitives import GRIPPER_OPEN, move_to, pick, set_gripper

    info, retry_frames = set_gripper(env, GRIPPER_OPEN, duration=0.35)
    frames.extend(retry_frames)
    ee_xy = np.asarray(info["gripper_pos"][:2], dtype=float)
    info, _, retry_frames = move_to(
        env,
        [ee_xy[0], ee_xy[1], SAFE_RETREAT_HEIGHT],
        gripper=GRIPPER_OPEN,
        duration=0.8,
    )
    frames.extend(retry_frames)
    found, _ = aim_at(env, "red_block", frames=frames, attempts=3)
    if not found:
        return False, env._get_info()
    info = env._get_info()
    return pick(env, info["block_pos"], frames=frames)


def teacher_complete_task(env, frames):
    """Recover from the current state and finish with a verified 6 cm place."""
    from skills.primitives import place

    info = env._get_info()
    if info["block_height"] <= LIFTED_HEIGHT:
        grasped, info = teacher_repick(env, frames)
        if not grasped:
            return False, info, "repick_failed"
    placed, info = place(env, info["target_pos"][:2], frames=frames)
    return bool(placed and info["success"]), info, (
        "completed" if placed and info["success"] else "place_failed"
    )


def vision_reaim(env, frames):
    """Reacquire with overhead coarse localization and wrist fine aiming."""
    from skills.block_reacquisition import reacquire_block

    report = reacquire_block(env, frames=frames)
    return bool(report["ready_for_pick"])


def rollout(
    env,
    policy,
    pre,
    post,
    seed: int,
    aim=False,
    retry_pick=False,
    vision_retry_pick=False,
    dual_camera_recovery=False,
    max_pick_retries=3,
    pick_verify_steps=DEFAULT_PICK_VERIFY_STEPS,
    full_recovery=False,
    max_task_recoveries=2,
    aim_mode="current",
):
    # ACT samples a VAE latent during action prediction. Seed both the
    # environment and policy RNG so repeated evaluations of a seed are fair.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    obs, info = env.reset(seed=seed)
    overhead_calibration = env.render_overhead() if dual_camera_recovery else None
    if aim:  # 배포 파이프라인과 동일: 로컬 지각이 조준 → 정책 인계
        if dual_camera_recovery:
            vision_reaim(env, [])
        else:
            from skills.aiming import aim_at, aim_at_legacy_v22
            if aim_mode == "legacy_v22":
                aim_at_legacy_v22(env, "red_block")
            else:
                aim_at(env, "red_block")
        obs = env._get_obs()
    policy.reset()  # ACT 액션 청크 큐 초기화 (에피소드 간 누수 방지)
    frames = [obs["pixels"]]
    settled = 0
    lifted_once = info["block_height"] > LIFTED_HEIGHT
    retries = 0
    retry_successes = 0
    retry_exhausted = False
    last_retry_reason = None
    vision_monitor = VisionPickMonitor() if vision_retry_pick else None
    dual_monitor = (
        DualCameraTaskMonitor(overhead_calibration)
        if dual_camera_recovery else None
    )
    attempt_start_step = 0
    low_after_lift_steps = 0
    task_recoveries = 0
    task_recovery_reasons = []
    vision_source = (
        "wrist_rgb_overhead_rgb_and_joints"
        if dual_camera_recovery else "wrist_rgb_and_joints"
    )
    rollout_limit = (
        MAX_STEPS * (1 + max_task_recoveries)
        if dual_camera_recovery else MAX_STEPS
    )
    for t in range(rollout_limit):
        batch = pre(obs_to_batch(obs))
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = post(action)
        action = np.asarray(action.squeeze(0).cpu(), dtype=np.float64)
        obs, _, _, _, info = env.step(action)
        frames.append(obs["pixels"])
        dual_signals = None
        if dual_camera_recovery:
            dual_signals = dual_monitor.update(
                obs["pixels"], env.render_overhead(), obs["agent_pos"],
                t + 1 - attempt_start_step,
            )
            vision_event = dual_signals["pick_event"]
            lifted_once = dual_signals["dual_grasp_confirmed"]
        elif vision_retry_pick:
            _, vision_event = vision_monitor.update(
                obs["pixels"], obs["agent_pos"], t + 1 - attempt_start_step
            )
            lifted_once = vision_monitor.grasp_confirmed
        else:
            vision_event = None
            lifted_once = lifted_once or info["block_height"] > LIFTED_HEIGHT

        if lifted_once and info["block_height"] < DROP_HEIGHT:
            low_after_lift_steps += 1
        else:
            low_after_lift_steps = 0

        # Do not let a missed pick continue into transport. Discard ACT's
        # queued actions, retry the pick, and resume only after lift is verified.
        retry_reason = (
            vision_event if (vision_retry_pick or dual_camera_recovery)
            else repick_reason(info, t + 1, lifted_once, pick_verify_steps)
        )
        if (vision_retry_pick or dual_camera_recovery) and retry_reason is not None:
            last_retry_reason = retry_reason
            if retries >= max_pick_retries:
                retry_exhausted = True
                return False, t + 1, info, frames, {
                    "pick_retries": retries,
                    "teacher_repick_successes": 0,
                    "pick_retry_exhausted": True,
                    "lift_verified": False,
                    "pick_retry_reason": retry_reason,
                    "supervision_source": vision_source,
                }
            retries += 1
            policy.reset()
            if not vision_reaim(env, frames):
                if retries >= max_pick_retries:
                    retry_exhausted = True
                    return False, t + 1, env._get_info(), frames, {
                        "pick_retries": retries,
                        "teacher_repick_successes": 0,
                        "pick_retry_exhausted": True,
                        "lift_verified": False,
                        "pick_retry_reason": "vision_reaim_failed",
                        "supervision_source": vision_source,
                    }
            obs = env._get_obs()
            info = env._get_info()
            if dual_camera_recovery:
                dual_monitor = DualCameraTaskMonitor(env.render_overhead())
            else:
                vision_monitor.reset()
            attempt_start_step = t + 1
            continue

        # Only the validated high-precision transport-drop signal is active.
        # The monitor's failed-release signal remains shadow-only because its
        # current false-positive rate is not safe for control.
        if (
            dual_camera_recovery
            and dual_signals["transport_drop"]
            and task_recoveries < max_task_recoveries
        ):
            task_recoveries += 1
            task_recovery_reasons.append("vision_transport_drop")
            policy.reset()
            reaimed = vision_reaim(env, frames)
            obs = env._get_obs()
            info = env._get_info()
            if not reaimed and task_recoveries >= max_task_recoveries:
                return False, t + 1, info, frames, {
                    "pick_retries": retries,
                    "teacher_repick_successes": 0,
                    "pick_retry_exhausted": False,
                    "lift_verified": False,
                    "pick_retry_reason": "vision_reaim_failed_after_drop",
                    "task_recoveries": task_recoveries,
                    "task_recovery_reasons": task_recovery_reasons,
                    "supervision_source": vision_source,
                }
            dual_monitor = DualCameraTaskMonitor(env.render_overhead())
            attempt_start_step = t + 1
            lifted_once = False
            low_after_lift_steps = 0
            continue
        if retry_pick and retry_reason is not None:
            last_retry_reason = retry_reason
            while retries < max_pick_retries and not lifted_once:
                retries += 1
                policy.reset()
                grasped, info = teacher_repick(env, frames)
                lifted_once = bool(grasped and info["block_height"] > LIFTED_HEIGHT)
                if lifted_once:
                    retry_successes += 1
                    obs = env._get_obs()
                    break
            if not lifted_once:
                retry_exhausted = True
                if full_recovery:
                    while task_recoveries < max_task_recoveries:
                        task_recoveries += 1
                        task_recovery_reasons.append("pick_retries_exhausted")
                        recovered, info, _ = teacher_complete_task(env, frames)
                        if recovered:
                            return True, t + 1, info, frames, {
                                "pick_retries": retries,
                                "teacher_repick_successes": retry_successes,
                                "pick_retry_exhausted": True,
                                "lift_verified": True,
                                "pick_retry_reason": retry_reason,
                                "task_recoveries": task_recoveries,
                                "task_recovery_reasons": task_recovery_reasons,
                                "supervision_source": "mujoco_privileged_state",
                            }
                return False, t + 1, info, frames, {
                    "pick_retries": retries,
                    "teacher_repick_successes": retry_successes,
                    "pick_retry_exhausted": retry_exhausted,
                    "lift_verified": False,
                    "pick_retry_reason": retry_reason,
                    "task_recoveries": task_recoveries,
                    "task_recovery_reasons": task_recovery_reasons,
                    "supervision_source": "mujoco_privileged_state",
                }

        # Once a verified lift has occurred, a low block that is not correctly
        # placed means either a transport drop or a bad release. Finish the
        # task with the teacher instead of letting ACT continue empty-handed.
        if (
            full_recovery
            and lifted_once
            and low_after_lift_steps >= DROP_CONFIRM_STEPS
            and not info["success"]
            and task_recoveries < max_task_recoveries
        ):
            reason = (
                "bad_place" if info["dist_to_target"] <= 0.08
                else "dropped_during_transport"
            )
            task_recoveries += 1
            task_recovery_reasons.append(reason)
            policy.reset()
            recovered, info, _ = teacher_complete_task(env, frames)
            obs = env._get_obs()
            if recovered:
                return True, t + 1, info, frames, {
                    "pick_retries": retries,
                    "teacher_repick_successes": retry_successes,
                    "pick_retry_exhausted": False,
                    "lift_verified": True,
                    "pick_retry_reason": last_retry_reason,
                    "task_recoveries": task_recoveries,
                    "task_recovery_reasons": task_recovery_reasons,
                    "supervision_source": "mujoco_privileged_state",
                }
            low_after_lift_steps = 0

        ok = info["success"]
        settled = settled + 1 if ok else 0
        if settled >= SETTLE_STEPS:
            return True, t + 1, info, frames, {
                "pick_retries": retries,
                "teacher_repick_successes": retry_successes,
                "pick_retry_exhausted": retry_exhausted,
                "lift_verified": lifted_once,
                "pick_retry_reason": last_retry_reason,
                "task_recoveries": task_recoveries,
                "task_recovery_reasons": task_recovery_reasons,
                "supervision_source": (
                    vision_source if (vision_retry_pick or dual_camera_recovery)
                    else "mujoco_privileged_state" if retry_pick else "none"
                ),
            }
    if full_recovery and task_recoveries < max_task_recoveries:
        task_recoveries += 1
        task_recovery_reasons.append("timeout_or_unverified_place")
        policy.reset()
        recovered, info, _ = teacher_complete_task(env, frames)
        if recovered:
            return True, MAX_STEPS, info, frames, {
                "pick_retries": retries,
                "teacher_repick_successes": retry_successes,
                "pick_retry_exhausted": retry_exhausted,
                "lift_verified": True,
                "pick_retry_reason": last_retry_reason,
                "task_recoveries": task_recoveries,
                "task_recovery_reasons": task_recovery_reasons,
                "supervision_source": "mujoco_privileged_state",
            }
    return False, rollout_limit, info, frames, {
        "pick_retries": retries,
        "teacher_repick_successes": retry_successes,
        "pick_retry_exhausted": retry_exhausted,
        "lift_verified": lifted_once,
        "pick_retry_reason": last_retry_reason,
        "task_recoveries": task_recoveries,
        "task_recovery_reasons": task_recovery_reasons,
        "supervision_source": (
            vision_source if (vision_retry_pick or dual_camera_recovery)
            else "mujoco_privileged_state" if retry_pick else "none"
        ),
    }


def save_video(frames, path: Path, fps=25):
    """PyAV로 h264 mp4 인코딩 (imageio의 pyav 플러그인은 kwargs 비호환)."""
    try:
        import av
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as container:
            h, w = frames[0].shape[:2]
            stream = container.add_stream("h264", rate=fps)
            stream.width, stream.height = w, h
            stream.pix_fmt = "yuv420p"
            for fr in frames:
                for pkt in stream.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
        return True
    except Exception as e:
        print(f"  (영상 저장 실패: {type(e).__name__}: {e})")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="015000", help="체크포인트 (005000/010000/015000)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--version", default="v1", choices=list(VERSIONS))
    ap.add_argument("--checkpoint-root", type=Path)
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"])
    ap.add_argument(
        "--n-action-steps",
        type=int,
        help="한 번 관측한 뒤 실행할 action 수 (25Hz에서 10이면 0.4초)",
    )
    ap.add_argument("--run-name", help="평가 결과를 저장할 별도 폴더 이름")
    ap.add_argument(
        "--retry-pick", action="store_true",
        help="운반 전 들림을 확인하고 실패하면 선생이 재조준·재집기",
    )
    ap.add_argument(
        "--vision-retry-pick", action="store_true",
        help="MuJoCo 물체 좌표 없이 손목 RGB와 관절값으로 실패 판정·재시도",
    )
    ap.add_argument(
        "--dual-camera-recovery", action="store_true",
        help="손목+상단 RGB로 집기 실패/운반 낙하를 감지해 재조준·재집기",
    )
    ap.add_argument("--max-pick-retries", type=int, default=3)
    ap.add_argument("--pick-verify-steps", type=int, default=DEFAULT_PICK_VERIFY_STEPS)
    ap.add_argument(
        "--full-recovery", action="store_true",
        help="좌표 교사가 운반 낙하·놓기 실패까지 복구하여 작업 완료",
    )
    ap.add_argument("--max-task-recoveries", type=int, default=2)
    args = ap.parse_args()
    recovery_modes = sum(
        (args.retry_pick, args.vision_retry_pick, args.dual_camera_recovery)
    )
    if recovery_modes > 1:
        ap.error("Choose only one recovery mode")
    if args.dual_camera_recovery and args.full_recovery:
        ap.error("--dual-camera-recovery cannot be combined with --full-recovery")

    ckpt_rel, out_rel, image_key, camera = VERSIONS[args.version]
    global IMAGE_KEY
    IMAGE_KEY = image_key
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ckpt_root = args.checkpoint_root or REPO_ROOT / ckpt_rel
    print(f"{args.version} 체크포인트 {args.step} 로드 중 (관측={camera}, device={device})...")
    policy, pre, post = load_policy(args.step, device, ckpt_root)
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy.config.chunk_size:
            raise ValueError(
                f"n_action_steps must be between 1 and {policy.config.chunk_size}, "
                f"got {args.n_action_steps}"
            )
        policy.config.n_action_steps = args.n_action_steps
    action_steps = policy.config.n_action_steps
    print(
        f"재관측 간격: {action_steps} action = "
        f"{action_steps / 25:.2f}초"
    )
    env = SO101PickEnv(camera=camera)
    run_name = args.run_name or (
        f"{args.step}_nas{action_steps:03d}"
        if args.n_action_steps is not None
        else args.step
    )
    out_dir = REPO_ROOT / out_rel / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    saved_by_outcome = {True: 0, False: 0}
    t0 = time.time()
    for i in range(args.episodes):
        success, steps, info, frames, supervisor = rollout(
            env, policy, pre, post, seed=5000 + i,
            aim=args.version in AIM_START,
            retry_pick=args.retry_pick,
            vision_retry_pick=args.vision_retry_pick,
            dual_camera_recovery=args.dual_camera_recovery,
            max_pick_retries=args.max_pick_retries,
            pick_verify_steps=args.pick_verify_steps,
            full_recovery=args.full_recovery,
            max_task_recoveries=args.max_task_recoveries,
        )
        results.append({
            "seed": 5000 + i,
            "success": bool(success),
            "steps": int(steps),
            "total_control_steps": len(frames) - 1,
            "dist_m": float(info["dist_to_target"]),
            "target_coverage": float(info["target_coverage"]),
            "block_height_m": float(info["block_height"]),
            **supervisor,
        })
        if saved_by_outcome[bool(success)] < 3:
            tag = "ok" if success else "fail"
            if save_video(frames, out_dir / f"{tag}_seed{5000 + i}.mp4"):
                saved_by_outcome[bool(success)] += 1
        mark = "OK" if success else "FAIL"
        print(f"  ep {i + 1:2d}/{args.episodes} {mark}  {steps:3d}스텝 "
              f"({steps / 25:.1f}s)  최종거리 {info['dist_to_target'] * 1000:.0f}mm")

    n_ok = sum(r["success"] for r in results)
    ok_steps = [r["steps"] for r in results if r["success"]]
    print(f"\n{'=' * 52}")
    print(f"ACT ({args.step}스텝 학습) - 성공률 {n_ok}/{args.episodes} "
          f"({n_ok / args.episodes * 100:.0f}%)")
    if ok_steps:
        print(f"성공 에피소드 평균 {np.mean(ok_steps) / 25:.1f}초 "
              f"(레시피 선생 ~7초, System 2 ~60초+API콜)")
    print(f"{'=' * 52}")

    summary = {
        "version": args.version,
        "checkpoint": args.step,
        "episodes": args.episodes,
        "n_action_steps": action_steps,
        "reobservation_seconds": action_steps / 25,
        "retry_pick": bool(args.retry_pick),
        "vision_retry_pick": bool(args.vision_retry_pick),
        "dual_camera_recovery": bool(args.dual_camera_recovery),
        "max_pick_retries": int(args.max_pick_retries),
        "pick_verify_steps": int(args.pick_verify_steps),
        "full_recovery": bool(args.full_recovery),
        "max_task_recoveries": int(args.max_task_recoveries),
        "successes": n_ok,
        "success_rate": n_ok / args.episodes,
        "mean_final_distance_m": float(np.mean([r["dist_m"] for r in results])),
        "results": results,
    }
    result_path = out_dir / "results.json"
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    saved = sum(saved_by_outcome.values())
    print(f"롤아웃 영상 {saved}개 + 결과 JSON → {out_dir}")


if __name__ == "__main__":
    main()
