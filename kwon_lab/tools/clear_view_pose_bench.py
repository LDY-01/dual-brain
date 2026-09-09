"""Validate autonomous clear-view selection and block-pose classification."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np

from envs.so101_pick_env import SO101PickEnv
from skills.block_reacquisition import (
    CLEAR_VIEW_POSES,
    reacquire_block,
    select_clear_view_observation,
)
from skills.vision_supervision import color_masks


class FirstViewOcclusionEnv(SO101PickEnv):
    """Test-only camera fault that hides red at one clear-view pose."""

    def __init__(self, occluded_pose_names=()):
        super().__init__(camera="wrist")
        self.occluded_pose_names = set(occluded_pose_names)

    def render_overhead(self):
        frame = super().render_overhead()
        if not self.occluded_pose_names:
            return frame
        at_occluded_pose = any(
            np.linalg.norm(self.data.qpos[:5] - dict(CLEAR_VIEW_POSES)[name])
            <= 0.12
            for name in self.occluded_pose_names
        )
        if not at_occluded_pose:
            return frame
        red, _ = color_masks(frame)
        corrupted = frame.copy()
        corrupted[red] = np.array([128, 128, 128], dtype=np.uint8)
        return corrupted


def run_case(
    name,
    quaternion,
    center_z,
    block_xy=(0.23, 0.0),
    occluded_pose=None,
):
    env = FirstViewOcclusionEnv(
        occluded_pose_names=() if occluded_pose is None else (occluded_pose,)
    )
    env.reset(seed=1808)
    addr = env.block_qpos_addr
    env.data.qpos[addr : addr + 7] = [
        *block_xy,
        center_z,
        *quaternion,
    ]
    mujoco.mj_forward(env.model, env.data)
    location, clear_view = select_clear_view_observation(env)
    result = {
        "case": name,
        "block_xy": list(block_xy),
        "expected_pose_class": "TIPPED" if center_z < 0.025 else "UPRIGHT",
        "observed_pose_class": location.pose_class,
        "pose_confidence": location.pose_confidence,
        "aspect_ratio": location.aspect_ratio,
        "pixels": location.pixels,
        "preferred_pose": clear_view["preferred_pose"],
        "selected_pose": clear_view["selected_pose"],
        "alternate_pose_used": clear_view["alternate_pose_used"],
        "clear_view_sufficient": clear_view["clear_view_sufficient"],
        "candidates": clear_view["candidates"],
        "classification_correct": location.pose_class
        == ("TIPPED" if center_z < 0.025 else "UPRIGHT"),
        "fallback_correct": (
            occluded_pose is None
            or (
                clear_view["alternate_pose_used"]
                and clear_view["selected_pose"] != occluded_pose
            )
        ),
    }
    env.close()
    return result


def run_both_views_occluded_case():
    env = FirstViewOcclusionEnv(
        occluded_pose_names=("CLEAR_VIEW_A", "CLEAR_VIEW_B")
    )
    env.reset(seed=1808)
    addr = env.block_qpos_addr
    env.data.qpos[addr : addr + 7] = [0.23, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    reacquisition = reacquire_block(env, overhead_only=True)
    clear_view = reacquisition["clear_view"]
    result = {
        "case": "both_views_occluded_safe_stop",
        "observed_pose_class": reacquisition["overhead_pose_class"],
        "alternate_pose_used": clear_view["alternate_pose_used"],
        "clear_view_sufficient": clear_view["clear_view_sufficient"],
        "failure_reason": reacquisition["failure_reason"],
        "candidates": clear_view["candidates"],
        "safe_stop_correct": (
            reacquisition["overhead_pose_class"] == "UNKNOWN"
            and clear_view["alternate_pose_used"]
            and not clear_view["clear_view_sufficient"]
            and reacquisition["failure_reason"] == "block_not_visible_overhead"
        ),
    }
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    half_sqrt = float(np.sqrt(0.5))
    positions = ((0.18, -0.10), (0.23, 0.0), (0.28, 0.10))
    cases = [
        *[
            run_case(
                f"upright_position_{index}",
                (1.0, 0.0, 0.0, 0.0),
                0.03,
                block_xy=position,
            )
            for index, position in enumerate(positions, start=1)
        ],
        *[
            run_case(
                f"tipped_position_{index}",
                (half_sqrt, half_sqrt, 0.0, 0.0),
                0.02,
                block_xy=position,
            )
            for index, position in enumerate(positions, start=1)
        ],
        run_case(
            "preferred_view_occluded",
            (1.0, 0.0, 0.0, 0.0),
            0.03,
            occluded_pose="CLEAR_VIEW_B",
        ),
    ]
    safe_stop = run_both_views_occluded_case()
    report = {
        "benchmark": "clear_view_pose_selection",
        "cases": cases,
        "all_classifications_correct": all(
            case["classification_correct"] for case in cases
        ),
        "all_fallbacks_correct": all(case["fallback_correct"] for case in cases),
        "both_views_occluded_safe_stop": safe_stop,
        "passed": all(
            case["classification_correct"]
            and case["fallback_correct"]
            and case["clear_view_sufficient"]
            for case in cases
        )
        and safe_stop["safe_stop_correct"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
