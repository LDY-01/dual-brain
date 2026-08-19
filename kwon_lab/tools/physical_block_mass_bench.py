#!/usr/bin/env python3
"""Paired v2.2 evaluation: legacy 96 g block vs measured rounded 28.6 g block."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "kwon_lab"))

from envs.so101_pick_env import SO101PickEnv
from eval import eval_act_pick as eval_act


DEFAULT_BASELINE = (
    REPO_ROOT
    / "outputs/eval/act_pick_v22/act_pick_v22_nas010/results.json"
)
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "artifacts/checkpoints"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_mcnemar_p(regressions: int, improvements: int) -> float:
    discordant = regressions + improvements
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(regressions, improvements) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def friction_summary(rows: list[dict], low: float, high: float) -> list[dict]:
    if math.isclose(low, high):
        successes = sum(row["new_success"] for row in rows)
        return [
            {
                "range": [low, high],
                "episodes": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows) if rows else None,
            }
        ]
    boundaries = np.linspace(low, high, 4)
    summaries = []
    for index in range(3):
        left, right = float(boundaries[index]), float(boundaries[index + 1])
        selected = [
            row for row in rows
            if left <= row["friction"] <= right
            and (index == 2 or row["friction"] < right)
        ]
        successes = sum(row["new_success"] for row in selected)
        summaries.append(
            {
                "range": [left, right],
                "episodes": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected) if selected else None,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--step", default="act_pick_v22")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=5000)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--friction-min", type=float, default=0.25)
    parser.add_argument("--friction-max", type=float, default=0.8)
    parser.add_argument(
        "--object-profile",
        choices=("legacy_96g", "real_28g", "sharp_28g", "rounded_96g"),
        default="real_28g",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("version") != "v2.2" or baseline.get("n_action_steps") != args.n_action_steps:
        raise SystemExit("Baseline must be the v2.2 n_action_steps=10 evaluation")
    baseline_by_seed = {int(row["seed"]): row for row in baseline["results"]}
    seeds = list(range(args.start_seed, args.start_seed + args.episodes))
    missing = [seed for seed in seeds if seed not in baseline_by_seed]
    if missing:
        raise SystemExit(f"Baseline is missing paired seeds: {missing}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    eval_act.IMAGE_KEY = "observation.images.wrist"
    policy, pre, post = eval_act.load_policy(args.step, args.device, args.checkpoint_root)
    policy.config.n_action_steps = args.n_action_steps
    env = SO101PickEnv(
        camera="wrist",
        object_profile=args.object_profile,
        block_sliding_friction_range=(args.friction_min, args.friction_max),
    )

    rows = []
    started = time.time()
    try:
        for index, seed in enumerate(seeds, start=1):
            success, steps, info, _, supervisor = eval_act.rollout(
                env,
                policy,
                pre,
                post,
                seed=seed,
                aim=True,
                aim_mode="legacy_v22",
            )
            old = baseline_by_seed[seed]
            row = {
                "seed": seed,
                "legacy_success": bool(old["success"]),
                "new_success": bool(success),
                "friction": float(info["block_sliding_friction"]),
                "steps": int(steps),
                "final_distance_m": float(info["dist_to_target"]),
                "target_coverage": float(info["target_coverage"]),
                "block_height_m": float(info["block_height"]),
                "lift_verified": bool(supervisor["lift_verified"]),
            }
            rows.append(row)
            transition = f"{int(row['legacy_success'])}->{int(row['new_success'])}"
            print(
                f"[{index:02d}/{args.episodes}] seed={seed} old->new={transition} "
                f"mu={row['friction']:.3f} dist={row['final_distance_m']*1000:.0f}mm",
                flush=True,
            )
    finally:
        env.close()

    legacy_successes = sum(row["legacy_success"] for row in rows)
    new_successes = sum(row["new_success"] for row in rows)
    regressions = sum(row["legacy_success"] and not row["new_success"] for row in rows)
    improvements = sum(not row["legacy_success"] and row["new_success"] for row in rows)
    report = {
        "benchmark": "act_v22_legacy_vs_physics_profile",
        "checkpoint": args.step,
        "n_action_steps": args.n_action_steps,
        "reobservation_seconds": args.n_action_steps / 25,
        "aim_mode": "legacy_v22_frozen_2026_08_12",
        "paired_seed_range": [seeds[0], seeds[-1]],
        "episodes": args.episodes,
        "legacy_profile": {
            "geometry": "sharp 40x40x60 mm box",
            "mass_kg": 0.096,
            "sliding_friction": 1.0,
            "successes": legacy_successes,
            "success_rate": legacy_successes / args.episodes,
        },
        "measured_profile": {
            "object_profile": args.object_profile,
            "geometry": (
                "sharp 40x40x60 mm box"
                if args.object_profile in {"legacy_96g", "sharp_28g"}
                else "supplied STEP, 40x40x60 mm, all edges R1 mm"
            ),
            "mass_kg": (
                0.096
                if args.object_profile in {"legacy_96g", "rounded_96g"}
                else 0.0286
            ),
            "sliding_friction_uniform_range": [args.friction_min, args.friction_max],
            "successes": new_successes,
            "success_rate": new_successes / args.episodes,
            "wilson_95_interval": wilson_interval(new_successes, args.episodes),
        },
        "paired_changes": {
            "regressions_old_success_to_new_failure": regressions,
            "improvements_old_failure_to_new_success": improvements,
            "unchanged": args.episodes - regressions - improvements,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(regressions, improvements),
        },
        "success_by_friction_tercile": friction_summary(
            rows, args.friction_min, args.friction_max
        ),
        "mean_final_distance_m": float(np.mean([row["final_distance_m"] for row in rows])),
        "lift_verified_count": sum(row["lift_verified"] for row in rows),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
