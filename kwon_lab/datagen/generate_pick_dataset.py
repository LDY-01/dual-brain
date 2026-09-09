"""S4: 자율 데이터 생성 v2 — 손목캠 단독 관측 (에고센트릭 전환, 2026-08-10).

선생은 특권 레시피(라벨링 원리 — 학생 입력엔 특권 없음), 학생 관측은 wrist 카메라만.
새 씬(타일 바닥·조명 보정) 기준. front는 관전용이라 데이터에 미포함.

리더암 녹화의 완전한 대체물: 사람 손 대신 LLM이 설계한 pick/place 레시피가
시연자 역할을 한다. 출력은 lerobot 표준 LeRobotDataset — 이후 lerobot-train으로
리더암 데이터와 똑같이 ACT를 학습시킨다 (S5).

실행:  .venv/bin/python kwon_lab/datagen/generate_pick_dataset.py [목표성공수]
출력:  outputs/datasets/so101_sim_pick  (동명 디렉토리 있으면 삭제 후 새로 생성)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from envs.so101_pick_env import BLOCK_X, BLOCK_Y, SO101PickEnv
from skills.aiming import aim_at
from skills.primitives import pick, place

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "Pick up the red block and place it on the green target zone."
REPO_ID = "kwonlab/so101_sim_pick_v24"
DEFAULT_ROOT = Path("outputs/datasets/so101_sim_pick_v24")
FPS = 25
GRID_SHAPE = (5, 5, 8)  # x, y, yaw: 모두 합치면 200개 영역
MAX_ATTEMPTS_PER_CELL = 20

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

FEATURES = {
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs="?", default=50)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--sampling", choices=["random", "stratified"], default="stratified"
    )
    parser.add_argument("--seed", type=int, default=2400)
    args = parser.parse_args()
    n_target = args.episodes
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {root}. "
            "Move or remove it explicitly before retrying."
        )
    ds = LeRobotDataset.create(REPO_ID, fps=FPS, features=FEATURES, root=root,
                               robot_type="so101_sim")
    env = SO101PickEnv(camera="wrist")  # v2: 학생의 눈은 손목캠뿐 (에고센트릭 결정)
    real_render = env.render
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    rng = np.random.default_rng(args.seed)
    cells = []
    if args.sampling == "stratified":
        nx, ny, nyaw = GRID_SHAPE
        cells = [
            (ix, iy, iyaw)
            for ix in range(nx)
            for iy in range(ny)
            for iyaw in range(nyaw)
        ]
        rng.shuffle(cells)
        if n_target > len(cells):
            raise ValueError(
                f"stratified sampling supports at most {len(cells)} episodes, "
                f"got {n_target}"
            )
        cells = cells[:n_target]

    x_edges = np.linspace(BLOCK_X[0], BLOCK_X[1], GRID_SHAPE[0] + 1)
    y_edges = np.linspace(BLOCK_Y[0], BLOCK_Y[1], GRID_SHAPE[1] + 1)
    yaw_edges = np.linspace(-np.pi, np.pi, GRID_SHAPE[2] + 1)

    def sample_cell(cell):
        ix, iy, iyaw = cell
        return (
            rng.uniform(x_edges[ix], x_edges[ix + 1]),
            rng.uniform(y_edges[iy], y_edges[iy + 1]),
            rng.uniform(yaw_edges[iyaw], yaw_edges[iyaw + 1]),
        )

    t0 = time.time()
    n_success = n_attempt = total_frames = 0
    while n_success < n_target:
        cell = cells[n_success] if cells else None
        cell_attempt = 0
        success = False
        while not success:
            if cell_attempt >= MAX_ATTEMPTS_PER_CELL:
                raise RuntimeError(
                    f"teacher failed {MAX_ATTEMPTS_PER_CELL} times in cell {cell}"
                )
            options = {"block_pose": sample_cell(cell)} if cell else None
            obs, info = env.reset(seed=args.seed + n_attempt, options=options)
            n_attempt += 1
            cell_attempt += 1

            # 조준은 데이터에 기록하지 않으므로 고해상도 렌더링을 생략한다.
            env.render = lambda: blank_frame
            found, centered = aim_at(env, "red_block")
            env.render = real_render
            if not found:
                continue  # 같은 위치 영역 안에서 다시 시도한다.
            env._last_obs = env._get_obs()
            env.recorder = []

            grasped, info = pick(env, info["block_pos"])
            if grasped:
                placed, info = place(env, info["target_pos"][:2])
                success = bool(info["success"])

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
    print(f"데이터셋: {root}")


if __name__ == "__main__":
    main()
