"""Export the M0 Korean readability review bundle without connected hardware."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from writing.review import export_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or (Path(__file__).resolve().parents[2]
                             / "outputs" / "writing" / f"review_{timestamp}")
    manifest = export_review(output)
    print("Milestone: M0; mode: geometry preview only; no robot commands sent")
    print("Human readability review: pending; physical trials: 0")
    print(f"Manifest: {manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
