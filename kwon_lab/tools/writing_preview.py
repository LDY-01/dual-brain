"""Create a writing preview without importing robot, camera, or LLM modules."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from writing import PaperSettings, plan_text
from writing.preview import export_preview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="One Korean sentence")
    parser.add_argument("--width-mm", type=float, default=PaperSettings().width_mm)
    parser.add_argument("--height-mm", type=float, default=PaperSettings().height_mm)
    parser.add_argument("--character-mm", type=float, default=18)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or Path(__file__).resolve().parents[2] / "outputs" / "writing" / timestamp
    try:
        plan = plan_text(args.text, PaperSettings(width_mm=args.width_mm,
                                                height_mm=args.height_mm,
                                                character_mm=args.character_mm))
        paths = export_preview(plan, output)
    except ValueError as error:
        parser.error(str(error))
    print(f"Text: {plan.text}")
    print(f"Lines: {plan.lines}; strokes: {len(plan.strokes)}")
    print("Mode: geometry preview only; no robot commands sent")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
