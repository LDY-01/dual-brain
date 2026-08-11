"""그랩 파라미터 랜덤 탐색 — S4 '탐색은 최적화기' 원칙의 미니 실증.

렌더링 없이 물리만 돌려 그랩 파라미터 공간을 뒤진다. 성공 판정은 시뮬 특권
상태(블록 높이)로 공짜. 사람이 손튜닝으로 실패한 것을 물리+무작위가 푼다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np

from envs.so101_pick_env import SO101PickEnv
from skills.primitives import GRIPPER_OPEN, move_to, set_gripper


def make_fast_env():
    """탐색용: 픽셀 렌더를 건너뛰는 환경 (물리만)."""
    env = SO101PickEnv()
    env._get_obs = lambda: {"pixels": None, "agent_pos": env.data.qpos[: env.model.nu].copy()}
    return env


def attempt(env, params, seed=1):
    pitch, lateral, grasp_z, close_val = params
    obs, info = env.reset(seed=seed)
    b = info["block_pos"]
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    jaw_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

    # 접근
    move_to(env, b + [0, 0, 0.07], gripper=GRIPPER_OPEN, duration=1.0,
            point_down=True, pitch_deg=pitch)
    # 턱 방향 실측 후 하강 목표 계산
    gf = env.data.site_xpos[sid]
    mj = env.data.xpos[jaw_id]
    jaw_dir = mj - gf
    jaw_dir[2] = 0
    n = np.linalg.norm(jaw_dir)
    if n < 1e-6:
        return 0.0, info
    jaw_dir /= n
    gx, gy = b[:2] - jaw_dir[:2] * lateral
    move_to(env, [gx, gy, grasp_z], duration=0.8)
    # 닫고 잠시 안정화 후 리프트
    set_gripper(env, close_val, duration=0.6)
    set_gripper(env, close_val, duration=0.2)
    info, _, _ = move_to(env, [b[0], b[1], 0.15], duration=1.0)
    return info["block_height"], info


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    env = make_fast_env()
    results = []
    N = 120
    for i in range(N):
        params = (
            rng.uniform(40, 85),       # pitch_deg
            rng.uniform(0.0, 0.030),   # lateral offset (턱 반대방향, m)
            rng.uniform(0.006, 0.022), # grasp z (m)
            rng.uniform(-0.15, 0.4),   # gripper close value (rad)
        )
        try:
            h, _ = attempt(env, params)
        except Exception:
            h = 0.0
        results.append((h, params))
        if h > 0.04:
            print(f"[{i:3d}] ★ 성공 h={h*1000:.0f}mm  pitch={params[0]:.0f} lat={params[1]*1000:.0f}mm z={params[2]*1000:.0f}mm close={params[3]:.2f}")

    results.sort(key=lambda r: -r[0])
    print("\n상위 5개:")
    for h, p in results[:5]:
        print(f"  h={h*1000:5.0f}mm | pitch={p[0]:4.0f} lat={p[1]*1000:4.0f}mm z={p[2]*1000:4.0f}mm close={p[3]:5.2f}")
    n_succ = sum(1 for h, _ in results if h > 0.04)
    print(f"\n성공: {n_succ}/{N}")
