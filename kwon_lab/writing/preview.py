"""Export paper geometry for review; these files never authorize robot motion."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from .planner import WritingPlan


def to_svg(plan: WritingPlan, *, animate: bool = False) -> str:
    paper = plan.paper
    paths = []
    total_points = sum(len(stroke.points_mm) for stroke in plan.strokes)
    progress = 0
    for index, stroke in enumerate(plan.strokes):
        points = stroke.points_mm
        d = "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in points)
        animation = ""
        attrs = ""
        if animate:
            start = 12 * progress / total_points
            duration = max(.01, 12 * len(points) / total_points)
            attrs = ' pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"'
            animation = (f'<animate attributeName="stroke-dashoffset" from="1" to="0" '
                         f'begin="{start:.4f}s" dur="{duration:.4f}s" fill="freeze"/>')
        paths.append(f'<path id="stroke-{index}" d="{d}"{attrs}>{animation}</path>')
        progress += len(points)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" '
        f'height="{960 * paper.height_mm / paper.width_mm:.0f}" '
        f'viewBox="0 0 {paper.width_mm} {paper.height_mm}" role="img">'
        f'<title>{escape(plan.text)}</title>'
        '<desc>Paper-relative monoline preview. Not a calibrated robot trajectory.</desc>'
        '<rect width="100%" height="100%" fill="white"/>'
        '<g fill="none" stroke="#dce2e6" stroke-width="0.15">'
        f'<rect x="{paper.margin_mm}" y="{paper.margin_mm}" '
        f'width="{paper.width_mm - 2 * paper.margin_mm}" '
        f'height="{paper.height_mm - 2 * paper.margin_mm}"/></g>'
        '<g fill="none" stroke="#172326" stroke-width="0.65" '
        'stroke-linecap="round" stroke-linejoin="round">'
        + "".join(paths) + '</g></svg>'
    )


def export_preview(plan: WritingPlan, destination: Path) -> dict[str, Path]:
    from PIL import Image, ImageDraw

    destination.mkdir(parents=True, exist_ok=True)
    names = {name: destination / name for name in
             ("plan.json", "preview.svg", "animation.svg", "preview.png")}
    names["plan.json"].write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    names["preview.svg"].write_text(to_svg(plan), encoding="utf-8")
    names["animation.svg"].write_text(to_svg(plan, animate=True), encoding="utf-8")
    scale = 6
    image = Image.new("RGB", (round(plan.paper.width_mm * scale),
                             round(plan.paper.height_mm * scale)), "white")
    draw = ImageDraw.Draw(image)
    margin = plan.paper.margin_mm * scale
    draw.rectangle((margin, margin, image.width - margin, image.height - margin),
                   outline="#dce2e6", width=1)
    for stroke in plan.strokes:
        points = [(round(x * scale), round(y * scale)) for x, y in stroke.points_mm]
        draw.line(points, fill="#172326", width=4, joint="curve")
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#172326")
    image.save(names["preview.png"])
    return names
