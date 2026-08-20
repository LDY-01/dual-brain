"""Run resumable normal and forced-drop S6 registry evaluations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np

from envs.so101_pick_env import SO101PickEnv
from skills.block_reacquisition import locate_overhead_block
from skills.pick_place_catalog import PickPlaceRuntime, run_registered_pick_place
from skills.pick_place_state_machine import (
    BLOCK_FOOTPRINT_DIMENSIONS_M,
    PLACE_PROJECTED_SAFETY_MARGIN,
    PLACE_PROJECTED_SUCCESS_COVERAGE,
    PLACE_TASK_SUCCESS_COVERAGE,
)
from tools.skill_registry_bench import inject_slip


def nested_executions(execution):
    yield execution
    for child in execution.get("recovery", []):
        yield from nested_executions(child)


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - margin, center + margin]


def run_episode(mode, seed):
    env = SO101PickEnv(camera="wrist", object_profile="real_28g")
    injected = False
    started = time.perf_counter()
    try:
        env.reset(seed=seed)
        initial_block_xyz = env.data.qpos[
            env.block_qpos_addr : env.block_qpos_addr + 3
        ].copy()
        initial_yaw_quat = env.data.qpos[
            env.block_qpos_addr + 3 : env.block_qpos_addr + 7
        ].copy()

        def maybe_inject(current_env, step, _signals):
            nonlocal injected
            if mode == "forced_place_drop" and not injected and step == 3:
                inject_slip(current_env)
                injected = True

        result = run_registered_pick_place(
            PickPlaceRuntime(env),
            on_place_step=maybe_inject,
        )
        wall_time_s = time.perf_counter() - started
        info = env._get_info()
        location = locate_overhead_block(env.render_overhead())
        final_block_xyz = env.data.qpos[
            env.block_qpos_addr : env.block_qpos_addr + 3
        ].copy()
        flattened = [
            item
            for execution in result["executions"]
            for item in nested_executions(execution)
        ]
        recovery_used = any(execution.get("recovered") for execution in flattened)
        recovery_skills = [
            execution["skill"]
            for execution in flattened
            if execution["skill"] in {"recover_pick", "transport", "place_6cm"}
            and execution not in result["executions"]
        ]
        pick_attempts = sum(
            int((execution.get("result") or {}).get("attempts", 0))
            for execution in flattened
            if execution["skill"] in {"pick", "recover_pick"}
        )
        attempt_reports = [
            report
            for execution in flattened
            if execution["skill"] in {"pick", "recover_pick"}
            for report in (execution.get("result") or {}).get(
                "attempt_reports", []
            )
        ]
        boundary_guard_activations = sum(
            bool(report.get("boundary_guard_active"))
            for report in attempt_reports
        )
        boundary_guard_events = [
            {
                "estimated_table_xy": report.get("estimated_table_xy"),
                "inward_direction_xy": report.get(
                    "boundary_inward_direction_xy"
                ),
                "mode": report.get("boundary_guard_mode"),
                "edge_guards_m": report.get("boundary_edge_guards_m"),
                "approach_distance_m": report.get(
                    "pick_approach_distance_m"
                ),
                "pick_xy_offset_m": report.get("pick_xy_offset_m"),
                "pick_target_table_xy": report.get("pick_target_table_xy"),
                "pregrasp_table_xy": report.get("pregrasp_table_xy"),
                "recovery_config_cursor": report.get(
                    "recovery_config_cursor"
                ),
                "recovery_config_index": report.get(
                    "recovery_config_index"
                ),
                "failure_reason": report.get("failure_reason"),
            }
            for report in attempt_reports
            if report.get("boundary_guard_active")
        ]
        pick_attempt_diagnostics = [
            {
                "pick_attempt": report.get("pick_attempt"),
                "recovery_config_cursor": report.get(
                    "recovery_config_cursor"
                ),
                "recovery_config_index": report.get(
                    "recovery_config_index"
                ),
                "pose_class": report.get("overhead_pose_class"),
                "orientation_rad": report.get("overhead_orientation_rad"),
                "estimated_table_xy": report.get("estimated_table_xy"),
                "pick_xy_offset_m": report.get("pick_xy_offset_m"),
                "boundary_guard_active": report.get(
                    "boundary_guard_active"
                ),
                "boundary_guard_mode": report.get("boundary_guard_mode"),
                "ready_for_pick": report.get("ready_for_pick"),
                "pick_attempted": report.get("pick_attempted"),
                "camera_grasp_confirmed": report.get(
                    "camera_grasp_confirmed"
                ),
                "failure_reason": report.get("failure_reason"),
            }
            for report in attempt_reports
        ]
        place_reports = [
            execution.get("result") or {}
            for execution in flattened
            if execution["skill"] == "place_6cm"
        ]
        place_attempt_diagnostics = [
            {
                "status": execution.get("status"),
                "failure_reason": execution.get("failure_reason"),
                "elapsed_s": execution.get("elapsed_s"),
                "alignment_iterations": report.get("alignment_iterations"),
                "alignment_errors_px": report.get("alignment_errors_px"),
                "alignment_total_correction_m": report.get(
                    "alignment_total_correction_m"
                ),
                "alignment_total_limit_m": report.get(
                    "alignment_total_limit_m"
                ),
                "alignment_limited": report.get("alignment_limited"),
                "alignment_skipped_reason": report.get(
                    "alignment_skipped_reason"
                ),
                "drop_detected": report.get("drop_detected"),
                "projected_target_coverage": report.get(
                    "projected_target_coverage"
                ),
                "camera_place_confirmed": report.get(
                    "camera_place_confirmed"
                ),
            }
            for execution in flattened
            if execution["skill"] == "place_6cm"
            for report in [execution.get("result") or {}]
        ]
        final_place_report = place_reports[-1] if place_reports else {}
        failure_reasons = [
            execution["failure_reason"]
            for execution in flattened
            if execution.get("failure_reason")
        ]
        terminal = result["executions"][-1]
        deepest_failure_reason = (
            failure_reasons[-1]
            if not result["success"] and failure_reasons
            else None
        )
        return {
            "mode": mode,
            "seed": int(seed),
            "object_profile": env.object_profile,
            "friction": float(info["block_sliding_friction"]),
            "initial_block_xyz": initial_block_xyz.tolist(),
            "initial_block_quaternion": initial_yaw_quat.tolist(),
            "task_success": bool(result["success"]),
            "truth_success": bool(info["success"]),
            "camera_truth_agree": bool(result["success"]) == bool(info["success"]),
            "truth_target_coverage": float(info["target_coverage"]),
            "final_distance_m": float(info["dist_to_target"]),
            "final_block_xyz": final_block_xyz.tolist(),
            "final_overhead_visible": bool(location.visible),
            "final_block_reachable": bool(location.reachable),
            "pick_attempts": int(pick_attempts),
            "boundary_guard_activations": int(boundary_guard_activations),
            "boundary_guard_events": boundary_guard_events,
            "pick_attempt_diagnostics": pick_attempt_diagnostics,
            "place_attempt_diagnostics": place_attempt_diagnostics,
            "cycles_attempted": int(result.get("cycles_attempted", 0)),
            "recovery_used": bool(recovery_used),
            "recovery_attempted": bool(recovery_skills),
            "recovery_skills": recovery_skills,
            "fault_requested": mode == "forced_place_drop",
            "fault_injected": bool(injected),
            "fault_recovered": bool(
                injected and result["success"] and info["success"]
            ),
            "terminal_skill": terminal["skill"],
            "terminal_status": terminal["status"],
            "terminal_failure_reason": (
                None if result["success"] else deepest_failure_reason
            ),
            "top_level_failure_reason": (
                None if result["success"] else terminal.get("failure_reason")
            ),
            "observed_failure_reasons": failure_reasons,
            "final_success_coverage_method": final_place_report.get(
                "success_coverage_method"
            ),
            "final_projected_target_coverage": final_place_report.get(
                "projected_target_coverage"
            ),
            "final_image_target_coverage": final_place_report.get(
                "image_target_coverage"
            ),
            "final_camera_block_pose_class": final_place_report.get(
                "final_block_pose_class"
            ),
            "final_camera_block_pose_confidence": final_place_report.get(
                "final_block_pose_confidence"
            ),
            "final_camera_block_table_xy": final_place_report.get(
                "final_block_table_xy"
            ),
            "stage_budgets_s": result.get("stage_budgets_s"),
            "stage_budget_remaining_s": result.get(
                "stage_budget_remaining_s"
            ),
            "skill_elapsed_s": float(
                sum(float(execution["elapsed_s"]) for execution in flattened)
            ),
            "wall_time_s": float(wall_time_s),
        }
    finally:
        env.close()


def load_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_mode(rows):
    total = len(rows)
    successes = sum(row.get("task_success", False) for row in rows)
    truth_successes = sum(row.get("truth_success", False) for row in rows)
    wall_times = [float(row["wall_time_s"]) for row in rows if "wall_time_s" in row]
    attempts = [int(row["pick_attempts"]) for row in rows if "pick_attempts" in row]
    cycles = [int(row["cycles_attempted"]) for row in rows if "cycles_attempted" in row]
    failures = Counter(
        row.get("terminal_failure_reason") or "unknown"
        for row in rows
        if not row.get("task_success", False)
    )
    return {
        "episodes": total,
        "task_successes": successes,
        "task_success_rate": successes / total if total else 0.0,
        "wilson_95_interval": wilson_interval(successes, total),
        "truth_successes": truth_successes,
        "truth_success_rate": truth_successes / total if total else 0.0,
        "truth_wilson_95_interval": wilson_interval(truth_successes, total),
        "camera_truth_disagreements": sum(
            not row.get("camera_truth_agree", False) for row in rows
        ),
        "recovery_used": sum(row.get("recovery_used", False) for row in rows),
        "recovery_attempted": sum(
            row.get("recovery_attempted", False) for row in rows
        ),
        "faults_injected": sum(row.get("fault_injected", False) for row in rows),
        "faults_recovered": sum(row.get("fault_recovered", False) for row in rows),
        "boundary_guard_activations": sum(
            int(row.get("boundary_guard_activations", 0)) for row in rows
        ),
        "stage_budget_exhaustions": sum(
            any("stage budget" in str(reason) for reason in row.get(
                "observed_failure_reasons", []
            ))
            for row in rows
        ),
        "final_block_unreachable": sum(
            not row.get("final_block_reachable", False) for row in rows
        ),
        "mean_pick_attempts": float(np.mean(attempts)) if attempts else 0.0,
        "max_pick_attempts": max(attempts, default=0),
        "mean_task_cycles": float(np.mean(cycles)) if cycles else 0.0,
        "max_task_cycles": max(cycles, default=0),
        "mean_wall_time_s": float(np.mean(wall_times)) if wall_times else 0.0,
        "median_wall_time_s": float(np.median(wall_times)) if wall_times else 0.0,
        "p95_wall_time_s": float(np.percentile(wall_times, 95)) if wall_times else 0.0,
        "max_wall_time_s": max(wall_times, default=0.0),
        "false_positive_seeds": [
            int(row["seed"])
            for row in rows
            if row.get("task_success", False)
            and not row.get("truth_success", False)
        ],
        "false_negative_seeds": [
            int(row["seed"])
            for row in rows
            if not row.get("task_success", False)
            and row.get("truth_success", False)
        ],
        "failure_reasons": dict(sorted(failures.items())),
    }


def write_summary(path, rows, requested):
    by_mode = {
        mode: [row for row in rows if row.get("mode") == mode]
        for mode in ("normal", "forced_place_drop")
    }
    summary = {
        "protocol": {
            "runtime_privileged_object_state": False,
            "truth_used_after_episode_for_scoring_only": True,
            "object_profile": "real_28g",
            "success_coverage_method": (
                "physical_footprint_projected_polygon_intersection"
            ),
            "task_success_coverage_required": PLACE_TASK_SUCCESS_COVERAGE,
            "camera_success_coverage_required": (
                PLACE_PROJECTED_SUCCESS_COVERAGE
            ),
            "projection_safety_margin": PLACE_PROJECTED_SAFETY_MARGIN,
            "block_footprint_dimensions_m": BLOCK_FOOTPRINT_DIMENSIONS_M,
            "normal_seed_range": requested["normal_seed_range"],
            "forced_drop_seed_range": requested["forced_drop_seed_range"],
            "normal_seeds": requested["normal_seeds"],
            "forced_drop_seeds": requested["forced_drop_seeds"],
        },
        "normal": summarize_mode(by_mode["normal"]),
        "forced_place_drop": summarize_mode(by_mode["forced_place_drop"]),
    }
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normal-start-seed", type=int, default=12000)
    parser.add_argument("--normal-episodes", type=int, default=50)
    parser.add_argument("--fault-start-seed", type=int, default=13000)
    parser.add_argument("--fault-episodes", type=int, default=10)
    parser.add_argument(
        "--normal-seeds",
        help="Comma-separated explicit normal seeds; overrides the range.",
    )
    parser.add_argument(
        "--fault-seeds",
        help="Comma-separated explicit forced-drop seeds; overrides the range.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    if rows_path.exists() and not args.resume:
        raise SystemExit(f"Output already exists: {rows_path}; pass --resume")
    rows = load_rows(rows_path) if args.resume else []
    completed = {(row.get("mode"), int(row.get("seed", -1))) for row in rows}
    def parse_seeds(value):
        if not value:
            return None
        return [int(item.strip()) for item in value.split(",") if item.strip()]

    normal_seeds = parse_seeds(args.normal_seeds) or list(
        range(
            args.normal_start_seed,
            args.normal_start_seed + args.normal_episodes,
        )
    )
    fault_seeds = parse_seeds(args.fault_seeds) or list(
        range(
            args.fault_start_seed,
            args.fault_start_seed + args.fault_episodes,
        )
    )
    requested = {
        "normal_seed_range": (
            [min(normal_seeds), max(normal_seeds)] if normal_seeds else []
        ),
        "forced_drop_seed_range": (
            [min(fault_seeds), max(fault_seeds)] if fault_seeds else []
        ),
        "normal_seeds": normal_seeds,
        "forced_drop_seeds": fault_seeds,
    }
    schedule = [("normal", seed) for seed in normal_seeds] + [
        ("forced_place_drop", seed) for seed in fault_seeds
    ]

    total = len(schedule)
    for index, (mode, seed) in enumerate(schedule, start=1):
        if (mode, seed) in completed:
            continue
        try:
            row = run_episode(mode, seed)
        except Exception as exc:
            row = {
                "mode": mode,
                "seed": seed,
                "task_success": False,
                "truth_success": False,
                "camera_truth_agree": True,
                "terminal_failure_reason": f"exception:{type(exc).__name__}:{exc}",
                "exception": True,
            }
        append_row(rows_path, row)
        rows.append(row)
        summary = write_summary(summary_path, rows, requested)
        mode_summary = summary[mode]
        print(
            json.dumps(
                {
                    "progress": f"{index}/{total}",
                    "mode": mode,
                    "seed": seed,
                    "success": row["task_success"],
                    "mode_successes": mode_summary["task_successes"],
                    "mode_episodes": mode_summary["episodes"],
                    "failure": row.get("terminal_failure_reason"),
                    "wall_time_s": row.get("wall_time_s"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = write_summary(summary_path, rows, requested)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
