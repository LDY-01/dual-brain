"""Compare fixed A4-landscape sequence under bounded virtual paper placements."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kwon_lab.writing.evaluation import landscape_extent_cases, landscape_profiles, landscape_sequence_cases, run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/writing/landscape-evaluation"))
    parser.add_argument("--extent", action="store_true", help="Probe corners, center and later rows in the selected virtual profile")
    args = parser.parse_args()
    profiles = landscape_profiles()
    if args.extent:
        profiles = {"full_a4_origin_160_-70mm": profiles["full_a4_origin_160_-70mm"]}
    result = run_evaluation(args.output, profiles=profiles, cases=landscape_extent_cases() if args.extent else landscape_sequence_cases(),
                            progress=lambda profile, case, r: print(
                                f"{profile}/{case}: {r['status']} {r['failure'] or ''}", flush=True))
    print(f"Passed {result['passed_cases']}/{result['total_cases']}; virtual tests only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
