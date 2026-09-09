#!/usr/bin/env python3
"""Convert the measured red-block STEP model into a centered MuJoCo STL."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EXPECTED_MM = (40.0, 40.0, 60.0)
EXPECTED_FILLET_MM = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-source-to", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=0.02)
    args = parser.parse_args()

    try:
        import cadquery as cq
    except ImportError as exc:
        raise SystemExit(
            "CadQuery is required only for STEP conversion. Install it with: "
            "uv pip install --python .venv/Scripts/python.exe cadquery"
        ) from exc

    imported = cq.importers.importStep(str(args.source.resolve()))
    solids = imported.solids().vals()
    if len(solids) != 1:
        raise SystemExit(f"Expected exactly one solid, found {len(solids)}")
    shape = solids[0]
    bounds = shape.BoundingBox()
    dimensions = (bounds.xlen, bounds.ylen, bounds.zlen)
    errors = [abs(actual - expected) for actual, expected in zip(dimensions, EXPECTED_MM)]
    if max(errors) > 0.02:
        raise SystemExit(
            f"STEP dimensions are {dimensions} mm, expected {EXPECTED_MM} mm"
        )

    circular_radii = []
    geometry_counts: dict[str, int] = {}
    for edge in shape.Edges():
        kind = edge.geomType()
        geometry_counts[kind] = geometry_counts.get(kind, 0) + 1
        if kind == "CIRCLE":
            circular_radii.append(float(edge.radius()))
    fillet_radii = sorted(
        {round(radius, 6) for radius in circular_radii if radius <= 2.0}
    )
    if not fillet_radii or min(abs(radius - EXPECTED_FILLET_MM) for radius in fillet_radii) > 0.01:
        raise SystemExit(f"Expected a 1 mm edge fillet, found radii {fillet_radii}")

    center = (
        (bounds.xmin + bounds.xmax) / 2,
        (bounds.ymin + bounds.ymax) / 2,
        (bounds.zmin + bounds.zmax) / 2,
    )
    centered = shape.translate(cq.Vector(*(-value for value in center)))
    centered_bounds = centered.BoundingBox()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(
        centered,
        str(args.output),
        tolerance=args.tolerance_mm,
        angularTolerance=0.05,
    )
    if args.copy_source_to:
        args.copy_source_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source, args.copy_source_to)

    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "units": "millimeter",
        "dimensions_mm": list(dimensions),
        "source_bbox_center_mm": list(center),
        "centered_bbox_mm": {
            "min": [centered_bounds.xmin, centered_bounds.ymin, centered_bounds.zmin],
            "max": [centered_bounds.xmax, centered_bounds.ymax, centered_bounds.zmax],
        },
        "volume_mm3": float(shape.Volume()),
        "edge_geometry_counts": geometry_counts,
        "small_circular_radii_mm": fillet_radii,
        "tessellation_tolerance_mm": args.tolerance_mm,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
