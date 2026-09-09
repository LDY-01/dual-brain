"""Validate overhead coarse localization and wrist handoff across workspace."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np

from envs.so101_pick_env import BLOCK_HALF_H, SO101PickEnv
from skills.block_reacquisition import (
    pick_until_verified,
    reacquire_and_pick,
    reacquire_block,
)


POSITIONS = (
    (0.18, -0.10), (0.23, -0.10), (0.28, -0.10),
    (0.18, 0.00), (0.23, 0.00), (0.28, 0.00),
    (0.18, 0.10), (0.23, 0.10), (0.28, 0.10),
    (0.15, 0.18),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--retry-pick", action="store_true")
    args = parser.parse_args()
    env = SO101PickEnv(camera="wrist")
    rows = []
    selected = (
        range(len(POSITIONS)) if args.indices is None else args.indices
    )
    for index in selected:
        truth_xy = POSITIONS[index]
        env.reset(seed=8000 + index)
        addr = env.block_qpos_addr
        env.data.qpos[addr : addr + 3] = [*truth_xy, BLOCK_HALF_H]
        mujoco.mj_forward(env.model, env.data)
        if args.retry_pick:
            retry = pick_until_verified(env)
            report = dict(retry["attempt_reports"][-1])
            report["retry_success"] = retry["success"]
            report["pick_attempts"] = retry["attempts"]
        else:
            report = (
                reacquire_and_pick(env) if args.pick else reacquire_block(env)
            )
        estimated = report["estimated_table_xy"]
        error_m = (
            float(np.linalg.norm(np.asarray(estimated) - np.asarray(truth_xy)))
            if estimated is not None else None
        )
        row = {
            "truth_xy": truth_xy,
            "localization_error_mm": None if error_m is None else error_m * 1000,
            **report,
        }
        if args.pick or args.retry_pick:
            final_info = env._get_info()
            row["truth_block_height_m"] = float(final_info["block_height"])
            row["final_gripper_joint"] = float(env._get_obs()["agent_pos"][-1])
            row["truth_lifted"] = bool(final_info["block_height"] > 0.08)
        rows.append(row)
        print(row, flush=True)
    env.close()
    visible = sum(row["overhead_visible"] for row in rows)
    ready = sum(row["ready_for_pick"] for row in rows)
    errors = [
        row["localization_error_mm"] for row in rows
        if row["localization_error_mm"] is not None
    ]
    summary = {
        "positions": len(rows),
        "overhead_visible": visible,
        "ready_for_pick": ready,
        "mean_localization_error_mm": float(np.mean(errors)) if errors else None,
        "max_localization_error_mm": float(np.max(errors)) if errors else None,
        "rows": rows,
    }
    if args.pick or args.retry_pick:
        summary["pick_truth_successes"] = sum(row["truth_lifted"] for row in rows)
        summary["pick_camera_confirmed"] = sum(
            row["camera_grasp_confirmed"] for row in rows
        )
        summary["pick_confusion"] = {
            "tp": sum(
                row["truth_lifted"] and row["camera_grasp_confirmed"]
                for row in rows
            ),
            "tn": sum(
                not row["truth_lifted"] and not row["camera_grasp_confirmed"]
                for row in rows
            ),
            "fp": sum(
                not row["truth_lifted"] and row["camera_grasp_confirmed"]
                for row in rows
            ),
            "fn": sum(
                row["truth_lifted"] and not row["camera_grasp_confirmed"]
                for row in rows
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
