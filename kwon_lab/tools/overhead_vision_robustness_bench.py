#!/usr/bin/env python3
"""Stress the overhead red-block detector before the real workspace exists."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.so101_pick_env import SO101PickEnv
from skills.block_reacquisition import (
    CLEAR_VIEW_POSES,
    SIM_OVERHEAD_PIXEL_TO_TABLE,
    SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE,
    locate_overhead_block,
)
from skills.vision_supervision import color_masks


IMAGE_SIZE = (1280, 720)
NOMINAL_LENS_HEIGHT_M = 0.52


def quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def euler_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def affine_for_parallel_plane(table_matrix, plane_height_m):
    table = np.asarray(table_matrix, dtype=float)
    center_px = np.array([IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2])
    center_xy = table @ np.array([*center_px, 1.0])
    scale = (NOMINAL_LENS_HEIGHT_M - plane_height_m) / NOMINAL_LENS_HEIGHT_M
    linear = scale * table[:, :2]
    offset = center_xy - linear @ center_px
    return np.column_stack([linear, offset])


POSE_MATRICES = {
    "UPRIGHT": SIM_OVERHEAD_PIXEL_TO_TABLE,
    "TIPPED": affine_for_parallel_plane(SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE, 0.04),
}


def apply_photometric(frame, rng, *, strength=1.0):
    array = frame.astype(np.float32) / 255.0
    brightness = rng.uniform(0.60, 1.25) ** strength
    gamma = rng.uniform(0.80, 1.25) ** strength
    warmth = rng.uniform(-0.16, 0.16) * strength
    gains = np.array([1.0 + warmth, 1.0, 1.0 - warmth], dtype=np.float32)
    array = np.clip(array * gains * brightness, 0.0, 1.0)
    array = np.power(array, gamma)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def apply_partial_occlusion(frame, rng):
    red, _ = color_masks(frame)
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        return frame, 0.0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    fraction = float(rng.uniform(0.10, 0.35))
    width = max(1, int((x1 - x0) * fraction))
    if rng.random() < 0.5:
        start, end = x0, min(x1, x0 + width)
    else:
        start, end = max(x0, x1 - width), x1
    result = frame.copy()
    result[y0:y1, start:end] = np.array([90, 90, 90], dtype=np.uint8)
    return result, fraction


def summarize(rows):
    detected = [row for row in rows if row["visible"]]
    pose_correct = [row for row in rows if row["pose_correct"]]
    pose_unknown = [row for row in rows if row["observed_pose"] == "UNKNOWN"]
    pose_wrong = [
        row for row in rows
        if row["observed_pose"] not in {row["expected_pose"], "UNKNOWN"}
        and row["actionable_pose"]
    ]
    errors = [row["xy_error_m"] for row in detected if row["xy_error_m"] is not None]
    return {
        "cases": len(rows),
        "detected": len(detected),
        "detection_rate": len(detected) / len(rows),
        "pose_correct": len(pose_correct),
        "pose_accuracy": len(pose_correct) / len(rows),
        "pose_unknown": len(pose_unknown),
        "non_actionable_pose": sum(not row["actionable_pose"] for row in rows),
        "unsafe_wrong_pose": len(pose_wrong),
        "unsafe_wrong_pose_rate": len(pose_wrong) / len(rows),
        "mean_xy_error_m": float(np.mean(errors)) if errors else None,
        "p95_xy_error_m": float(np.percentile(errors, 95)) if errors else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-per-scenario", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1908)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-frames", type=Path)
    parser.add_argument("--max-failure-frames", type=int, default=12)
    args = parser.parse_args()
    if args.cases_per_scenario < 1:
        raise SystemExit("--cases-per-scenario must be positive")

    rng = np.random.default_rng(args.seed)
    env = SO101PickEnv(camera="wrist", object_profile="real_28g")
    overhead_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    nominal_pos = env.model.cam_pos[overhead_id].copy()
    nominal_quat = env.model.cam_quat[overhead_id].copy()
    rows = []
    failures_saved = 0
    scenarios = ("nominal", "photometric", "partial_occlusion", "stale_calibration")
    try:
        for scenario in scenarios:
            for case_index in range(args.cases_per_scenario):
                seed = args.seed + len(rows)
                env.reset(seed=seed)
                pose_class = "UPRIGHT" if case_index % 2 == 0 else "TIPPED"
                xy = np.array([
                    rng.uniform(0.18, 0.28),
                    rng.uniform(-0.10, 0.10),
                ])
                yaw = rng.uniform(-math.pi, math.pi)
                if pose_class == "UPRIGHT":
                    quaternion = euler_quat(0.0, 0.0, yaw)
                    center_z = 0.03
                else:
                    quaternion = quat_multiply(
                        euler_quat(0.0, 0.0, yaw),
                        euler_quat(math.pi / 2, 0.0, 0.0),
                    )
                    center_z = 0.02
                addr = env.block_qpos_addr
                env.data.qpos[addr : addr + 7] = [*xy, center_z, *quaternion]
                # Evaluate from the same unobstructed posture that the runtime
                # clear-view selector uses before accepting a pose decision.
                clear_pose = CLEAR_VIEW_POSES[0 if xy[1] >= 0 else 1][1]
                env.data.qpos[:5] = clear_pose
                env.data.ctrl[:5] = clear_pose
                env.model.cam_pos[overhead_id] = nominal_pos
                env.model.cam_quat[overhead_id] = nominal_quat
                camera_delta = {"xyz_m": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}
                if scenario == "stale_calibration":
                    dpos = np.array([
                        rng.uniform(-0.025, 0.025),
                        rng.uniform(-0.025, 0.025),
                        rng.uniform(-0.04, 0.04),
                    ])
                    rpy = np.deg2rad([
                        rng.uniform(-3.0, 3.0),
                        rng.uniform(-3.0, 3.0),
                        rng.uniform(-5.0, 5.0),
                    ])
                    env.model.cam_pos[overhead_id] = nominal_pos + dpos
                    env.model.cam_quat[overhead_id] = quat_multiply(
                        euler_quat(*rpy), nominal_quat
                    )
                    camera_delta = {
                        "xyz_m": dpos.tolist(),
                        "rpy_deg": np.rad2deg(rpy).tolist(),
                    }
                mujoco.mj_forward(env.model, env.data)
                frame = env.render_overhead()
                occlusion_fraction = 0.0
                if scenario in {"photometric", "partial_occlusion", "stale_calibration"}:
                    frame = apply_photometric(frame, rng)
                if scenario == "partial_occlusion":
                    frame, occlusion_fraction = apply_partial_occlusion(frame, rng)
                location = locate_overhead_block(frame, POSE_MATRICES)
                xy_error = (
                    float(np.linalg.norm(np.asarray(location.table_xy) - xy))
                    if location.table_xy is not None
                    else None
                )
                pose_correct = location.pose_class == pose_class
                # The runtime visibility gate needs roughly 0.29 pose confidence
                # even with ideal area/reach/border scores. Lower-confidence
                # labels trigger the alternate clear-view posture instead of a pick.
                actionable_pose = (
                    location.pose_class != "UNKNOWN"
                    and location.pose_confidence >= 0.30
                )
                row = {
                    "scenario": scenario,
                    "case": case_index + 1,
                    "seed": seed,
                    "expected_pose": pose_class,
                    "observed_pose": location.pose_class,
                    "visible": location.visible,
                    "pose_correct": pose_correct,
                    "pose_confidence": location.pose_confidence,
                    "actionable_pose": actionable_pose,
                    "xy_error_m": xy_error,
                    "pixels": location.pixels,
                    "aspect_ratio": location.aspect_ratio,
                    "occlusion_fraction": occlusion_fraction,
                    "camera_delta": camera_delta,
                }
                rows.append(row)
                failed = not location.visible or not pose_correct or (xy_error or 0) > 0.015
                if failed and args.failure_frames and failures_saved < args.max_failure_frames:
                    args.failure_frames.mkdir(parents=True, exist_ok=True)
                    name = f"{scenario}_{case_index+1:02d}_{pose_class}.png"
                    cv2.imwrite(str(args.failure_frames / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    failures_saved += 1
    finally:
        env.close()

    summaries = {
        scenario: summarize([row for row in rows if row["scenario"] == scenario])
        for scenario in scenarios
    }
    nominal_pass = (
        summaries["nominal"]["detection_rate"] >= 0.95
        and summaries["nominal"]["pose_accuracy"] >= 0.95
        and summaries["nominal"]["p95_xy_error_m"] <= 0.015
    )
    photometric_pass = (
        summaries["photometric"]["detection_rate"] >= 0.90
        and summaries["photometric"]["pose_accuracy"] >= 0.85
        and summaries["photometric"]["p95_xy_error_m"] <= 0.015
    )
    occlusion_pass = (
        summaries["partial_occlusion"]["detection_rate"] >= 0.80
        and summaries["partial_occlusion"]["unsafe_wrong_pose_rate"] <= 0.10
    )
    stale_requires_recalibration = summaries["stale_calibration"]["p95_xy_error_m"] > 0.015
    report = {
        "benchmark": "overhead_vision_workspace_robustness",
        "seed": args.seed,
        "cases_per_scenario": args.cases_per_scenario,
        "scenarios": summaries,
        "gates": {
            "nominal_pass": nominal_pass,
            "photometric_pass": photometric_pass,
            "partial_occlusion_pass": occlusion_pass,
            "stale_calibration_correctly_requires_recalibration": stale_requires_recalibration,
        },
        "passed": bool(
            nominal_pass and photometric_pass and occlusion_pass and stale_requires_recalibration
        ),
        "interpretation": (
            "Photometric and partial-occlusion robustness are runtime requirements; "
            "camera pose changes are expected to fail metric XY accuracy until recalibrated."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
