"""Deterministic boundary checks for the shared 75% placement criterion."""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.so101_pick_env import BLOCK_HALF_H, SO101PickEnv


def set_block(env, xy_offset, yaw_deg=0.0, height=BLOCK_HALF_H):
    target = env.data.site_xpos[env.target_site].copy()
    yaw = np.radians(yaw_deg)
    quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    addr = env.block_qpos_addr
    env.data.qpos[addr : addr + 7] = [
        target[0] + xy_offset[0],
        target[1] + xy_offset[1],
        height,
        *quat,
    ]
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return env._get_info()


def main():
    env = SO101PickEnv(render_size=(120, 160), camera="wrist")
    env.render = lambda: np.zeros((120, 160, 3), dtype=np.uint8)
    try:
        env.reset(seed=0)
        cases = [
            ("centered_on_table", (0.0, 0.0), 0.0, BLOCK_HALF_H, True),
            ("centered_but_held_high", (0.0, 0.0), 0.0, 0.08, False),
            ("40mm_offset_yaw0", (0.04, 0.0), 0.0, BLOCK_HALF_H, False),
            ("40mm_offset_yaw45", (0.04, 0.0), 45.0, BLOCK_HALF_H, True),
            ("well_outside", (0.10, 0.0), 0.0, BLOCK_HALF_H, False),
        ]
        for name, offset, yaw, height, expected in cases:
            info = set_block(env, offset, yaw, height)
            actual = info["success"]
            print(
                f"{name}: coverage={info['target_coverage']:.4f}, "
                f"height={info['block_height']:.3f}m, success={actual}"
            )
            assert actual is expected, (name, expected, actual, info)
        print("All placement success checks passed.")
    finally:
        env.renderer.close()


if __name__ == "__main__":
    main()
