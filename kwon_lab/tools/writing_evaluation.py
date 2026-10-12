"""Evaluate a bounded matrix of virtual mounting/paper conditions."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kwon_lab.writing.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/writing/condition-evaluation"))
    args = parser.parse_args()
    result = run_evaluation(args.output, lambda profile, case, r: print(
        f"{profile}/{case}: {r['status']} {r['failure'] or ''}", flush=True))
    print(f"Compared {result['total_cases']} cases. Rejections are results, not physical failures.")
    print(args.output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
