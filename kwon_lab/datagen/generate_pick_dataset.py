"""S4: 자율 데이터 생성 v2 — 손목캠 단독 관측 (에고센트릭 전환, 2026-08-10).

선생은 특권 레시피(라벨링 원리 — 학생 입력엔 특권 없음), 학생 관측은 wrist 카메라만.
새 씬(타일 바닥·조명 보정) 기준. front는 관전용이라 데이터에 미포함.

리더암 녹화의 완전한 대체물: 사람 손 대신 LLM이 설계한 pick/place 레시피가
시연자 역할을 한다. 출력은 lerobot 표준 LeRobotDataset — 이후 lerobot-train으로
리더암 데이터와 똑같이 ACT를 학습시킨다 (S5).

실행:  .venv/bin/python kwon_lab/datagen/generate_pick_dataset.py [목표성공수]
출력:  outputs/datasets/so101_sim_pick  (동명 디렉토리 있으면 삭제 후 새로 생성)
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from envs.so101_pick_env import SO101PickEnv
from skills.aiming import aim_at
from skills.primitives import pick, place

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "Pick up the red block and place it on the green target zone."
REPO_ID = "kwonlab/so101_sim_pick_v2"
ROOT = Path("outputs/datasets/so101_sim_pick_v2")
FPS = 25

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

FEATURES = {
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
}


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ds = LeRobotDataset.create(REPO_ID, fps=FPS, features=FEATURES, root=ROOT,
                               robot_type="so101_sim")
    env = SO101PickEnv(camera="wrist")  # v2: 학생의 눈은 손목캠뿐 (에고센트릭 결정)

    t0 = time.time()
    n_success = n_attempt = total_frames = 0
    while n_success < n_target:
        obs, info = env.reset(seed=1000 + n_attempt)
        n_attempt += 1
        # v2.1: 조준 후 시작 — 첫 관측에 블록이 보여야 학생이 방향을 배울 수 있다
        # (홈 시작 v2는 정보 비대칭으로 5% — 관측 없는 지식은 증류되지 않는다)
        found, centered = aim_at(env, "red_block")
        if not found:
            continue  # 핵심 요구는 가시성 — 블록이 첫 관측에 보이기만 하면 학습 가능
        env.recorder = []  # 조준된 자세부터 기록 시작

        grasped, info = pick(env, info["block_pos"])
        success = False
        if grasped:
            placed, info = place(env, info["target_pos"][:2])
            success = info["success"]

        if success:
            for fr in env.recorder:
                ds.add_frame({
                    "observation.images.wrist": fr["pixels"],
                    "observation.state": fr["state"].astype(np.float32),
                    "action": fr["action"],
                    "task": TASK,
                })
            ds.save_episode()
            n_success += 1
            total_frames += len(env.recorder)
        env.recorder = None

        if n_attempt % 10 == 0 or success and n_success % 10 == 0:
            el = time.time() - t0
            print(f"[{el:5.0f}s] 시도 {n_attempt} → 성공 {n_success}/{n_target} "
                  f"(성공률 {n_success / n_attempt * 100:.0f}%, 누적 {total_frames}프레임)")

    el = time.time() - t0
    print(f"\n완료: {n_success}에피소드 / {n_attempt}시도 ({n_success / n_attempt * 100:.0f}%) "
          f"| {total_frames}프레임 | {el / 60:.1f}분 | API 비용 $0")
    print(f"데이터셋: {ROOT}")


if __name__ == "__main__":
    main()
