"""Compare teacher placement release heights with fixed seeds.

Measures IK error, actual block-bottom release height, target coverage,
center error, final speed, and stable success under the shared 75% criterion.
Results are written below outputs/place_height_bench/ (gitignored).
"""

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
from skills.primitives import GRIPPER_OPEN, move_to, pick, set_gripper


def run_trial(seed: int, release_height: float) -> dict:
    # Rendering is still part of every env.step(), but the teacher benchmark does
    # not use pixels. Replace render() with a fixed frame; physics is unchanged.
    env = SO101PickEnv(render_size=(120, 160), camera="wrist")
    blank_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    env.render = lambda: blank_frame
    try:
        _, start = env.reset(seed=seed)
        grasped, pick_info = pick(env, start["block_pos"])
        result = {
            "seed": seed,
            "release_height_cm": release_height * 100,
            "grasped": bool(grasped),
            "ik_error_mm": None,
            "release_bottom_cm": None,
            "coverage": 0.0,
            "center_error_cm": None,
            "final_speed_m_s": None,
            "stable_success": False,
        }
        if not grasped:
            return result

        target = start["target_pos"][:2]
        _, ik_error, _ = move_to(
            env, [target[0], target[1], release_height], duration=1.5
        )
        before_release = env._get_info()
        result["ik_error_mm"] = ik_error * 1000
        result["release_bottom_cm"] = (
            before_release["block_height"] - BLOCK_HALF_H
        ) * 100

        set_gripper(env, GRIPPER_OPEN, duration=0.6)
        move_to(env, [target[0], target[1], 0.20], duration=0.8)

        stable_steps = 0
        final = env._get_info()
        for _ in range(25):  # release 후 1초간 안착 여부 확인
            _, _, _, _, final = env.step(env.data.ctrl.copy())
            stable_steps = stable_steps + 1 if final["success"] else 0

        block_joint = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "block_free"
        )
        dof = env.model.jnt_dofadr[block_joint]
        speed = float(np.linalg.norm(env.data.qvel[dof : dof + 3]))
        result.update(
            coverage=final["target_coverage"],
            center_error_cm=final["dist_to_target"] * 100,
            final_speed_m_s=speed,
            stable_success=bool(stable_steps >= 10),
        )
        return result
    finally:
        env.renderer.close()


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for height in sorted({row["release_height_cm"] for row in rows}, reverse=True):
        group = [row for row in rows if row["release_height_cm"] == height]
        placed = [row for row in group if row["grasped"]]

        def mean(key):
            values = [row[key] for row in placed if row[key] is not None]
            return float(np.mean(values)) if values else None

        summaries.append(
            {
                "release_height_cm": height,
                "trials": len(group),
                "grasp_rate": sum(row["grasped"] for row in group) / len(group),
                "stable_success_rate": sum(row["stable_success"] for row in group)
                / len(group),
                "mean_ik_error_mm": mean("ik_error_mm"),
                "mean_release_bottom_cm": mean("release_bottom_cm"),
                "mean_coverage": mean("coverage"),
                "mean_center_error_cm": mean("center_error_cm"),
                "mean_final_speed_m_s": mean("final_speed_m_s"),
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=6000)
    parser.add_argument("--heights-cm", type=float, nargs="+", default=[8, 6, 5, 4])
    args = parser.parse_args()

    rows = []
    total = args.episodes * len(args.heights_cm)
    for height in args.heights_cm:
        for index in range(args.episodes):
            row = run_trial(args.seed_start + index, height / 100)
            rows.append(row)
            mark = "OK" if row["stable_success"] else "FAIL"
            print(
                f"[{len(rows):3d}/{total}] h={height:g}cm seed={row['seed']} {mark} "
                f"coverage={row['coverage']:.3f} center={row['center_error_cm']}"
            )

    summary = summarize(rows)
    out_dir = Path("outputs/place_height_bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"trials_{stamp}.csv"
    json_path = out_dir / f"summary_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    for item in summary:
        print(json.dumps(item, ensure_ascii=False))
    print(f"\nCSV: {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
