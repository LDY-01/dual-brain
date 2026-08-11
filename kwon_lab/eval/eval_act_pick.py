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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from envs.so101_pick_env import SO101PickEnv

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
}
AIM_START = {"v2.1", "v2.2"}
MAX_STEPS = 300          # 25Hz × 12초 — 레시피 시연(~7초)보다 넉넉하게
SETTLE_STEPS = 10        # 성공 상태가 이만큼 연속 유지돼야 인정 (스쳐 지나감 방지)


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


def rollout(env, policy, pre, post, seed: int, aim=False):
    obs, info = env.reset(seed=seed)
    if aim:  # 배포 파이프라인과 동일: 로컬 지각이 조준 → 정책 인계
        from skills.aiming import aim_at
        aim_at(env, "red_block")
        obs = env._get_obs()
    policy.reset()  # ACT 액션 청크 큐 초기화 (에피소드 간 누수 방지)
    frames = [obs["pixels"]]
    settled = 0
    for t in range(MAX_STEPS):
        batch = pre(obs_to_batch(obs))
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = post(action)
        action = np.asarray(action.squeeze(0).cpu(), dtype=np.float64)
        obs, _, _, _, info = env.step(action)
        frames.append(obs["pixels"])

        ok = info["success"]
        settled = settled + 1 if ok else 0
        if settled >= SETTLE_STEPS:
            return True, t + 1, info, frames
    return False, MAX_STEPS, info, frames


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
    ap.add_argument("--version", default="v1", choices=["v1", "v2", "v2.1", "v2.2"])
    args = ap.parse_args()

    ckpt_rel, out_rel, image_key, camera = VERSIONS[args.version]
    global IMAGE_KEY
    IMAGE_KEY = image_key
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{args.version} 체크포인트 {args.step} 로드 중 (관측={camera}, device={device})...")
    policy, pre, post = load_policy(args.step, device, REPO_ROOT / ckpt_rel)
    env = SO101PickEnv(camera=camera)
    out_dir = REPO_ROOT / out_rel / args.step

    results = []
    t0 = time.time()
    for i in range(args.episodes):
        success, steps, info, frames = rollout(env, policy, pre, post, seed=5000 + i,
                                                aim=args.version in AIM_START)
        results.append({"seed": 5000 + i, "success": success, "steps": steps,
                        "dist": info["dist_to_target"], "frames": frames})
        mark = "✅" if success else "❌"
        print(f"  ep {i + 1:2d}/{args.episodes} {mark}  {steps:3d}스텝 "
              f"({steps / 25:.1f}s)  최종거리 {info['dist_to_target'] * 1000:.0f}mm")

    n_ok = sum(r["success"] for r in results)
    ok_steps = [r["steps"] for r in results if r["success"]]
    print(f"\n{'=' * 52}")
    print(f"ACT ({args.step}스텝 학습) — 성공률 {n_ok}/{args.episodes} "
          f"({n_ok / args.episodes * 100:.0f}%)")
    if ok_steps:
        print(f"성공 에피소드 평균 {np.mean(ok_steps) / 25:.1f}초 "
              f"(레시피 선생 ~7초, System 2 ~60초+API콜)")
    print(f"{'=' * 52}")

    # 영상: 성공 처음 3개 + 실패 처음 3개 (원인 분석용)
    saved = 0
    for r in results:
        tag = "ok" if r["success"] else "fail"
        if sum(1 for x in results[:results.index(r)]
               if x["success"] == r["success"]) < 3:
            if save_video(r["frames"], out_dir / f"{tag}_seed{r['seed']}.mp4"):
                saved += 1
    print(f"롤아웃 영상 {saved}개 → {out_dir}")


if __name__ == "__main__":
    main()
