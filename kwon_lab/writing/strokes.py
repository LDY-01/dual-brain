"""A simple monoline Hangul alphabet; geometry is not a calligraphy model."""

from __future__ import annotations

from math import cos, pi, sin
import unicodedata

Point = tuple[float, float]
Stroke = list[Point]
Box = tuple[float, float, float, float]


def _circle(cx=0.5, cy=0.5, rx=0.35, ry=0.35) -> Stroke:
    return [(cx + rx * cos(t * 2 * pi / 40), cy + ry * sin(t * 2 * pi / 40))
            for t in range(41)]


CONSONANTS: dict[str, list[Stroke]] = {
    "ㄱ": [[(.1, .12), (.9, .12), (.9, .9)]],
    "ㄴ": [[(.1, .1), (.1, .88), (.9, .88)]],
    "ㄷ": [[(.1, .12), (.9, .12)], [(.1, .12), (.1, .88), (.9, .88)]],
    "ㄹ": [[(.1, .1), (.9, .1), (.9, .5), (.1, .5), (.1, .9), (.9, .9)]],
    "ㅁ": [[(.1, .1), (.1, .9)], [(.1, .1), (.9, .1), (.9, .9), (.1, .9)]],
    "ㅂ": [[(.1, .1), (.1, .9)], [(.9, .1), (.9, .9)],
           [(.1, .5), (.9, .5)], [(.1, .9), (.9, .9)]],
    "ㅅ": [[(.5, .12), (.1, .9)], [(.5, .12), (.9, .9)]],
    "ㅇ": [_circle()],
    "ㅈ": [[(.1, .12), (.9, .12)], [(.5, .12), (.1, .9)], [(.5, .12), (.9, .9)]],
    "ㅊ": [[(.35, .08), (.65, .08)], [(.1, .3), (.9, .3)],
           [(.5, .3), (.1, .9)], [(.5, .3), (.9, .9)]],
    "ㅋ": [[(.1, .1), (.9, .1), (.9, .9)], [(.1, .5), (.9, .5)]],
    "ㅌ": [[(.1, .1), (.9, .1)], [(.1, .5), (.9, .5)],
           [(.1, .1), (.1, .9), (.9, .9)]],
    "ㅍ": [[(.1, .12), (.9, .12)], [(.3, .12), (.3, .88)],
           [(.7, .12), (.7, .88)], [(.1, .88), (.9, .88)]],
    "ㅎ": [[(.35, .08), (.65, .08)], [(.1, .26), (.9, .26)],
           _circle(.5, .64, .32, .25)],
}

VOWELS: dict[str, list[Stroke]] = {
    "ㅏ": [[(.35, .05), (.35, .95)], [(.35, .5), (.9, .5)]],
    "ㅑ": [[(.35, .05), (.35, .95)], [(.35, .35), (.9, .35)], [(.35, .65), (.9, .65)]],
    "ㅓ": [[(.65, .05), (.65, .95)], [(.1, .5), (.65, .5)]],
    "ㅕ": [[(.65, .05), (.65, .95)], [(.1, .35), (.65, .35)], [(.1, .65), (.65, .65)]],
    "ㅗ": [[(.5, .05), (.5, .65)], [(.05, .65), (.95, .65)]],
    "ㅛ": [[(.35, .05), (.35, .65)], [(.65, .05), (.65, .65)], [(.05, .65), (.95, .65)]],
    "ㅜ": [[(.05, .35), (.95, .35)], [(.5, .35), (.5, .95)]],
    "ㅠ": [[(.05, .35), (.95, .35)], [(.35, .35), (.35, .95)], [(.65, .35), (.65, .95)]],
    "ㅡ": [[(.05, .5), (.95, .5)]],
    "ㅣ": [[(.5, .05), (.5, .95)]],
}

DOUBLE = {"ㄲ": "ㄱㄱ", "ㄸ": "ㄷㄷ", "ㅃ": "ㅂㅂ", "ㅆ": "ㅅㅅ", "ㅉ": "ㅈㅈ"}
CLUSTERS = {"ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ", "ㄺ": "ㄹㄱ",
            "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ",
            "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ"}
INITIALS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
MEDIALS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
FINALS = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
MIXED = {"ㅘ": ("ㅗ", "ㅏ"), "ㅙ": ("ㅗ", "ㅐ"), "ㅚ": ("ㅗ", "ㅣ"),
         "ㅝ": ("ㅜ", "ㅓ"), "ㅞ": ("ㅜ", "ㅔ"), "ㅟ": ("ㅜ", "ㅣ"),
         "ㅢ": ("ㅡ", "ㅣ")}
PAIRED = {"ㅐ": "ㅏ", "ㅒ": "ㅑ", "ㅔ": "ㅓ", "ㅖ": "ㅕ"}
HORIZONTAL = set("ㅗㅛㅜㅠㅡ")


def _fit(strokes: list[Stroke], box: Box) -> list[Stroke]:
    x0, y0, x1, y1 = box
    return [[(x0 + x * (x1 - x0), y0 + y * (y1 - y0)) for x, y in stroke]
            for stroke in strokes]


def consonant(char: str) -> list[Stroke]:
    pair = DOUBLE.get(char) or CLUSTERS.get(char)
    if pair:
        return (_fit(consonant(pair[0]), (0, 0, .46, 1))
                + _fit(consonant(pair[1]), (.54, 0, 1, 1)))
    return CONSONANTS[char]


def vowel(char: str) -> list[Stroke]:
    if char in PAIRED:
        return (_fit(VOWELS[PAIRED[char]], (0, 0, .65, 1))
                + [[(.85, .05), (.85, .95)]])
    return VOWELS[char]


def glyph(char: str) -> list[Stroke]:
    """Return normalized paper coordinates: rightward x, downward y."""
    if "가" <= char <= "힣":
        parts = unicodedata.normalize("NFD", char)
        initial = INITIALS[ord(parts[0]) - 0x1100]
        medial = MEDIALS[ord(parts[1]) - 0x1161]
        final = FINALS[ord(parts[2]) - 0x11A8] if len(parts) == 3 else None
        if medial in MIXED:
            horizontal, vertical = MIXED[medial]
            strokes = (_fit(consonant(initial), (.08, .05, .5, .55))
                       + _fit(vowel(horizontal), (.08, .62, .82, .92))
                       + _fit(vowel(vertical), (.62, .05, .94, .92)))
        elif medial in HORIZONTAL:
            strokes = (_fit(consonant(initial), (.16, .06, .84, .58))
                       + _fit(vowel(medial), (.08, .65, .92, .94)))
        else:
            strokes = (_fit(consonant(initial), (.06, .08, .48, .92))
                       + _fit(vowel(medial), (.55, .08, .94, .92)))
        if final:
            strokes = (_fit(strokes, (0, 0, 1, .67))
                       + _fit(consonant(final), (.15, .74, .85, .96)))
        return strokes
    if char in CONSONANTS or char in DOUBLE or char in CLUSTERS:
        return _fit(consonant(char), (.08, .08, .92, .92))
    if char in VOWELS or char in PAIRED:
        return _fit(vowel(char), (.08, .08, .92, .92))
    if char in MIXED:
        horizontal, vertical = MIXED[char]
        return (_fit(vowel(horizontal), (.08, .65, .75, .9))
                + _fit(vowel(vertical), (.7, .08, .95, .9)))
    if char == " ":
        return []
    if char == ".":
        return [_circle(.5, .88, .025, .025)]
    if char == ",":
        return [[(.55, .8), (.45, .95)]]
    if char == "!":
        return [[(.5, .1), (.5, .68)], _circle(.5, .88, .025, .025)]
    if char == "?":
        return [[(.15, .3), (.2, .15), (.5, .08), (.8, .2), (.85, .35),
                 (.5, .55), (.5, .68)], _circle(.5, .88, .025, .025)]
    raise ValueError(f"Unsupported character U+{ord(char):04X}: {char!r}")
