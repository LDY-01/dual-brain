"""SO-101 픽업 태스크 Gymnasium 환경 (MuJoCo).

관측·액션 형태를 실물 lerobot 인터페이스와 맞춰서, 이후 정책·System 2가
시뮬과 실물을 같은 코드로 다룰 수 있게 한다 (PROJECT.md 설계 원칙).

- 관측: {"pixels": (H,W,3) uint8 고정 카메라, "agent_pos": (6,) 관절각 rad}
- 액션: (6,) 목표 관절각 rad — MJCF의 position 액추에이터가 추종
- 성공: 블록 바닥 면적의 75% 이상이 목표 구역 안에 있고 테이블에 내려놓인 상태
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# Menagerie(DeepMind) 모델 사용 — 그랩 가능한 충돌 프리미티브 내장.
# 원본 SO-ARM100 모델은 손가락이 볼록 껍질로 계산돼 그랩 불가 (PROJECT.md 결정 기록 참조)
ASSETS = Path(__file__).parent.parent / "assets" / "menagerie_so101"
SCENE_XML = ASSETS / "pick_scene.xml"
BLOCK_HALF_H = 0.03  # 4x4x6cm 박스 (Menagerie 검증 스펙)

# 블록 스폰 구역 (홈 자세 그리퍼 (0.29, 0) 기준 손 닿는 범위)
BLOCK_X = (0.18, 0.28)
BLOCK_Y = (-0.10, 0.10)
SUCCESS_COVERAGE = 0.75  # 블록 바닥 면적 중 목표 구역에 포함돼야 하는 비율
BLOCK_ON_TABLE_Z = 0.05  # 블록 중심이 이보다 낮아야 실제로 내려놓은 것으로 판정


def load_mj_model(xml_path):
    """Load MJCF even when its Windows path contains non-ASCII characters."""
    path = Path(xml_path).resolve()
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except ValueError as exc:
        if "Error opening file" not in str(exc):
            raise
        assets = {
            asset.relative_to(path.parent).as_posix(): asset.read_bytes()
            for asset in path.parent.rglob("*")
            if asset.is_file()
        }
        return mujoco.MjModel.from_xml_string(
            path.read_text(encoding="utf-8"), assets=assets
        )


def camera_gravity(model, data, camera_name):
    """카메라 프레임 기준 중력(아래) 방향 — 카메라에 IMU가 달린 것과 동일.
    반환 벡터: x=이미지 오른쪽, y=이미지 위, z=카메라 뒤쪽(시선 반대) 성분."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    R = data.cam_xmat[cid].reshape(3, 3)
    return R.T @ np.array([0.0, 0.0, -1.0])


def upright_k(g):
    """중력이 이미지 아래를 향하도록 하는 np.rot90 반시계 회전 횟수(0~3).
    스마트폰의 IMU 자동 회전과 같은 원리 (가장 가까운 90° 단위)."""
    import math
    ang = math.degrees(math.atan2(g[0], -g[1]))  # 0°=이미 정립, -90°=중력이 화면 왼쪽
    return round(-ang / 90) % 4


class SO101PickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_size=(480, 640),
        control_hz=25,
        camera="front",
        overhead_render_size=(720, 1280),
    ):
        self.model = load_mj_model(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.camera = camera
        h, w = render_size
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._overhead_render_size = tuple(overhead_render_size)
        self._overhead_renderer = None

        # 물리 timestep(모델 정의) 대비 제어 주기 → 스텝당 물리 반복 횟수
        self.n_substeps = max(1, int(1.0 / control_hz / self.model.opt.timestep))

        self.block_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_free")
        ]
        self.block_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.block_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom"
        )
        self.target_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_zone")

        # 블록 바닥 면적의 목표 구역 포함률을 고정 격자로 근사한다. 격자 중심을
        # 사용해 경계 편향을 줄이고, 블록의 현재 yaw 회전도 반영한다.
        grid_n = 61
        axis = (np.arange(grid_n, dtype=float) + 0.5) / grid_n * 2.0 - 1.0
        gx, gy = np.meshgrid(axis, axis, indexing="xy")
        half_xy = self.model.geom_size[self.block_geom, :2]
        self._block_footprint = np.column_stack(
            [gx.ravel() * half_xy[0], gy.ravel() * half_xy[1]]
        )
        self._target_radius = float(self.model.site_size[self.target_site, 0])

        ctrl = self.model.actuator_ctrlrange  # (6, 2)
        self.action_space = spaces.Box(
            low=ctrl[:, 0].astype(np.float32), high=ctrl[:, 1].astype(np.float32)
        )
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(0, 255, (h, w, 3), np.uint8),
                "agent_pos": spaces.Box(-np.inf, np.inf, (self.model.nu,), np.float64),
            }
        )

    # ── 핵심 API ──────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 기본은 블록 위치·yaw 무작위화. 데이터 수집에서는 options로 특정
        # 위치 구간을 지정할 수 있으며, 일반 reset 동작은 이전과 동일하다.
        block_pose = (options or {}).get("block_pose")
        if block_pose is None:
            x = self.np_random.uniform(*BLOCK_X)
            y = self.np_random.uniform(*BLOCK_Y)
            yaw = self.np_random.uniform(-np.pi, np.pi)
        else:
            x, y, yaw = map(float, block_pose)
            if not (BLOCK_X[0] <= x <= BLOCK_X[1]):
                raise ValueError(f"block x={x} is outside {BLOCK_X}")
            if not (BLOCK_Y[0] <= y <= BLOCK_Y[1]):
                raise ValueError(f"block y={y} is outside {BLOCK_Y}")
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.data.qpos[self.block_qpos_addr : self.block_qpos_addr + 7] = [
            x, y, BLOCK_HALF_H, *quat,
        ]

        # 팔은 홈 자세(전 관절 0 = 가동범위 중앙), 모터 목표도 0
        self.data.ctrl[:] = 0.0
        # 물리 안정화 (블록 착지 등)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        self._last_obs = obs  # 기록용: 다음 action과 짝지을 관측
        return obs, self._get_info()

    def step(self, action):
        # 데이터 수집 훅: (직전 관측, 지금 적용할 행동) 쌍을 기록
        # — 모방학습의 표준 규약 "이 상태를 보고 이 행동을 했다"
        if getattr(self, "recorder", None) is not None and getattr(self, "_last_obs", None) is not None:
            self.recorder.append({
                "pixels": self._last_obs["pixels"],
                "state": self._last_obs["agent_pos"].copy(),
                "action": np.asarray(action, dtype=np.float32).copy(),
            })
        self.data.ctrl[:] = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        # 라이브 뷰어가 붙어 있으면 화면 갱신 + 실제 속도로 페이싱
        viewer = getattr(self, "live_viewer", None)
        if viewer is not None:
            import time
            viewer.sync()
            dt = self.n_substeps * self.model.opt.timestep
            elapsed = time.time() - getattr(self, "_last_step_t", 0)
            time.sleep(max(0, dt - elapsed))
            self._last_step_t = time.time()

        info = self._get_info()
        success = info["success"]
        reward = 1.0 if success else 0.0
        return self._get_obs(), reward, bool(success), False, info

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def render_overhead(self):
        """Render the fixed 52 cm overhead U20CAM approximation at 720p."""
        if self._overhead_renderer is None:
            h, w = self._overhead_render_size
            self._overhead_renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._overhead_renderer.update_scene(self.data, camera="overhead")
        return self._overhead_renderer.render()

    def close(self):
        self.renderer.close()
        if self._overhead_renderer is not None:
            self._overhead_renderer.close()
            self._overhead_renderer = None

    # ── 내부 ──────────────────────────────────────────────────

    def _get_obs(self):
        obs = {
            "pixels": self.render(),
            "agent_pos": self.data.qpos[: self.model.nu].copy(),
        }
        self._last_obs = obs
        return obs

    def _get_info(self):
        """특권 상태 — System 2 피드백과 성공 판정에 사용 (시뮬 전용)."""
        block = self.data.xpos[self.block_body]
        target = self.data.site_xpos[self.target_site]
        rotation_xy = self.data.xmat[self.block_body].reshape(3, 3)[:2, :2]
        footprint_world = block[:2] + self._block_footprint @ rotation_xy.T
        coverage = float(np.mean(
            np.linalg.norm(footprint_world - target[:2], axis=1) <= self._target_radius
        ))
        block_height = float(block[2])
        return {
            "block_pos": block.copy(),
            "target_pos": target.copy(),
            "dist_to_target": float(np.linalg.norm(block[:2] - target[:2])),
            "target_coverage": coverage,
            "block_height": block_height,
            "success": bool(
                coverage >= SUCCESS_COVERAGE and block_height < BLOCK_ON_TABLE_Z
            ),
            "gripper_pos": self.data.xpos[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
            ].copy(),
        }
