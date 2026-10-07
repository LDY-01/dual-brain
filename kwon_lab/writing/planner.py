"""Build bounded, portable writing geometry without touching hardware."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, hypot, isfinite
import unicodedata

from .strokes import glyph


@dataclass(frozen=True)
class PaperSettings:
    width_mm: float = 210
    height_mm: float = 297
    margin_mm: float = 8
    character_mm: float = 18
    character_gap_mm: float = 3
    line_gap_mm: float = 6
    pen_lift_mm: float = 8
    sample_spacing_mm: float = 1

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        for name in ("width_mm", "height_mm", "character_mm", "pen_lift_mm", "sample_spacing_mm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sample_spacing_mm < .1:
            raise ValueError("sample_spacing_mm must be at least 0.1")
        if self.width_mm > 1000 or self.height_mm > 1000:
            raise ValueError("Preview paper dimensions must not exceed 1000 mm")
        if min(self.width_mm, self.height_mm) - 2 * self.margin_mm < self.character_mm:
            raise ValueError("A character does not fit inside the paper margins")


@dataclass(frozen=True)
class PenStroke:
    character: str
    character_index: int
    points_mm: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class WritingPlan:
    text: str
    paper: PaperSettings
    strokes: tuple[PenStroke, ...]
    lines: int

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "kind": "paper_relative_writing_plan",
            "units": "mm",
            "frame": {"origin": "paper_top_left", "x": "right", "y": "down",
                      "z": "away_from_paper", "contact_z_mm": 0},
            "physical_calibration_required": True,
            "motion_authorized": False,
            "text": self.text,
            "paper": asdict(self.paper),
            "lines": self.lines,
            "strokes": [asdict(stroke) for stroke in self.strokes],
            "commands": self.commands(),
        }

    def commands(self) -> list[dict]:
        commands = []
        for index, stroke in enumerate(self.strokes):
            x, y = stroke.points_mm[0]
            commands.append({"op": "travel", "stroke": index,
                             "point_mm": [x, y, self.paper.pen_lift_mm]})
            commands.append({"op": "pen_down", "stroke": index,
                             "point_mm": [x, y, 0]})
            commands.append({"op": "stroke", "stroke": index,
                             "points_mm": [[px, py, 0] for px, py in stroke.points_mm]})
            x, y = stroke.points_mm[-1]
            commands.append({"op": "pen_up", "stroke": index,
                             "point_mm": [x, y, self.paper.pen_lift_mm]})
        return commands


def _resample(points: list[tuple[float, float]], spacing: float) -> tuple[tuple[float, float], ...]:
    sampled = [points[0]]
    for start, end in zip(points, points[1:]):
        steps = max(1, ceil(hypot(end[0] - start[0], end[1] - start[1]) / spacing))
        for index in range(1, steps + 1):
            t = index / steps
            sampled.append((start[0] + t * (end[0] - start[0]),
                            start[1] + t * (end[1] - start[1])))
    return tuple(sampled)


def plan_text(text: str, paper: PaperSettings | None = None) -> WritingPlan:
    paper = paper or PaperSettings()
    paper.validate()
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if any(ord(char) < 32 for char in text):
        raise ValueError("Use a single input line; line wrapping is planned automatically")
    text = unicodedata.normalize("NFC", text).strip()
    if not text or len(text) > 120:
        raise ValueError("text must contain 1 to 120 characters")
    # Resolve every glyph before planning, so an unsupported suffix cannot yield a partial job.
    shapes = [glyph(char) for char in text]
    usable_width = paper.width_mm - 2 * paper.margin_mm
    usable_height = paper.height_mm - 2 * paper.margin_mm
    pitch_x = paper.character_mm + paper.character_gap_mm
    pitch_y = paper.character_mm + paper.line_gap_mm
    columns = int((usable_width + paper.character_gap_mm) // pitch_x)
    rows = int((usable_height + paper.line_gap_mm) // pitch_y)
    strokes = []
    column = row = 0
    for index, (char, shape) in enumerate(zip(text, shapes)):
        if column == columns:
            row += 1
            column = 0
            if char == " ":
                continue
        if row >= rows:
            raise ValueError("Text exceeds the paper area; shorten it or increase the preview area")
        x0 = paper.margin_mm + column * pitch_x
        y0 = paper.margin_mm + row * pitch_y
        for stroke in shape:
            points = [(x0 + x * paper.character_mm, y0 + y * paper.character_mm)
                      for x, y in stroke]
            sampled = _resample(points, paper.sample_spacing_mm)
            for x, y in sampled:
                if not (paper.margin_mm <= x <= paper.width_mm - paper.margin_mm
                        and paper.margin_mm <= y <= paper.height_mm - paper.margin_mm):
                    raise ValueError("A stroke crosses the paper margin")
            strokes.append(PenStroke(char, index, sampled))
        column += 1
    return WritingPlan(text, paper, tuple(strokes), row + 1)
