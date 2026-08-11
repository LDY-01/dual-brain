"""Validate the complete pick/place teacher policy over fixed random seeds."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.so101_pick_env import BLOCK_HALF_H, SO101PickEnv
from skills.primitives import PLACE_RELEASE_HEIGHT, pick, place


def run_trial(seed: int) -> dict:
    env = SO101PickEnv(render_size=(120, 160), camera="wrist")
    env.render = lambda: np.zeros((120, 160, 3), dtype=np.uint8)
    try:
        _, start = env.reset(seed=seed)
        grasped, _ = pick(env, start["block_pos"])
        reported_success = False
        if grasped:
            reported_success, _ = place(env, start["target_pos"][:2])

        stable_steps = 0
        final = env._get_info()
        for _ in range(25):
            _, _, _, _, final = env.step(env.data.ctrl.copy())
            stable_steps = stable_steps + 1 if final["success"] else 0

        block_joint = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "block_free"
        )
        dof = env.model.jnt_dofadr[block_joint]
        speed = float(np.linalg.norm(env.data.qvel[dof : dof + 3]))
        offset = final["block_pos"][:2] - final["target_pos"][:2]
        return {
            "seed": seed,
            "release_height_cm": PLACE_RELEASE_HEIGHT * 100,
            "grasped": bool(grasped),
            "reported_success": bool(reported_success),
            "stable_success": bool(stable_steps >= 10),
            "coverage": final["target_coverage"],
            "center_error_cm": final["dist_to_target"] * 100,
            "x_error_cm": float(offset[0] * 100),
            "y_error_cm": float(offset[1] * 100),
            "block_bottom_cm": (final["block_height"] - BLOCK_HALF_H) * 100,
            "final_speed_m_s": speed,
        }
    finally:
        env.renderer.close()


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["stable_success"]]

    def mean(key, group=rows):
        return float(np.mean([row[key] for row in group])) if group else None

    return {
        "release_height_cm": PLACE_RELEASE_HEIGHT * 100,
        "trials": len(rows),
        "grasp_rate": sum(row["grasped"] for row in rows) / len(rows),
        "reported_success_rate": sum(row["reported_success"] for row in rows)
        / len(rows),
        "stable_success_rate": sum(row["stable_success"] for row in rows)
        / len(rows),
        "mean_coverage": mean("coverage"),
        "successful_mean_coverage": mean("coverage", successful),
        "successful_mean_center_error_cm": mean("center_error_cm", successful),
        "successful_mean_x_error_cm": mean("x_error_cm", successful),
        "successful_mean_y_error_cm": mean("y_error_cm", successful),
        "mean_final_speed_m_s": mean("final_speed_m_s"),
        "reported_stable_mismatches": sum(
            row["reported_success"] != row["stable_success"] for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=6000)
    args = parser.parse_args()

    rows = []
    for index in range(args.episodes):
        row = run_trial(args.seed_start + index)
        rows.append(row)
        mark = "OK" if row["stable_success"] else "FAIL"
        print(
            f"[{len(rows):3d}/{args.episodes}] seed={row['seed']} {mark} "
            f"grasp={row['grasped']} coverage={row['coverage']:.3f} "
            f"center={row['center_error_cm']:.2f}cm"
        )

    summary = summarize(rows)
    out_dir = Path("outputs/teacher_policy_bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"trials_{stamp}.csv"
    json_path = out_dir / f"summary_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCSV: {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
