"""Roman-numeral analysis and repeated-loop detection."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .models import (
    PITCH_CLASSES,
    ChordSegment,
    KeyEstimate,
    Loop,
    ProgressionSummary,
    parse_chord_label,
)

#: Degree (semitones above the tonic) -> Roman numeral, spelled against the
#: major scale.  Chromatic degrees are named with flats, which is how they are
#: normally written in pop/rock analysis (bIII, bVI, bVII).
MAJOR_DEGREES = {
    0: "I",
    1: "bII",
    2: "II",
    3: "bIII",
    4: "III",
    5: "IV",
    6: "#IV",
    7: "V",
    8: "bVI",
    9: "VI",
    10: "bVII",
    11: "VII",
}

#: Same, spelled against the natural minor scale, so i-iv-v and bVI-bVII-i come
#: out with the numerals a musician would actually write.
MINOR_DEGREES = {
    0: "I",
    1: "bII",
    2: "II",
    3: "III",
    4: "#III",
    5: "IV",
    6: "#IV",
    7: "V",
    8: "VI",
    9: "#VI",
    10: "VII",
    11: "#VII",
}

#: Qualities written with a lowercase numeral.
_LOWERCASE_QUALITIES = {"min", "dim", "min7"}

#: Suffix appended after the numeral.
_QUALITY_SUFFIX = {
    "maj": "",
    "min": "",
    "dim": "°",
    "aug": "+",
    "maj6": "6",
    "maj7": "maj7",
    "min7": "7",
    "7": "7",
}


def roman_numeral(label: str, tonic: str, mode: str) -> str:
    """Roman numeral for one chord label relative to ``tonic``/``mode``.

    Chords outside the diatonic set are labelled literally (``bVII``, ``bIII``)
    rather than being coerced onto a nearby scale degree — a borrowed chord is
    information, not noise.
    """
    parsed = parse_chord_label(label)
    if parsed is None:
        return label  # the no-chord state passes straight through

    root, quality = parsed
    try:
        tonic_pc = PITCH_CLASSES.index(tonic)
    except ValueError:
        return label

    degree = (root - tonic_pc) % 12
    table = MINOR_DEGREES if mode == "minor" else MAJOR_DEGREES
    numeral = table[degree]

    if quality in _LOWERCASE_QUALITIES:
        # Keep any accidental prefix uppercase-agnostic: bIII -> biii.
        numeral = "".join(ch.lower() if ch.isalpha() else ch for ch in numeral)

    return numeral + _QUALITY_SUFFIX.get(quality, quality)


def _collapse_repeats(segments: list[ChordSegment]) -> list[tuple[str, float, float]]:
    """Collapse consecutive same-label segments into ``(label, start, end)`` groups."""
    groups: list[tuple[str, float, float]] = []
    for segment in segments:
        if groups and groups[-1][0] == segment.label:
            label, start, _ = groups[-1]
            groups[-1] = (label, start, segment.end)
        else:
            groups.append((segment.label, segment.start, segment.end))
    return groups


def find_main_loop(
    labels: list[str], min_length: int = 2, max_length: int = 8
) -> tuple[list[str], int, int, int] | None:
    """Find the most prominent immediately-repeating cycle in ``labels``.

    Returns ``(pattern, repeats, start_index, end_index)`` where the indices are
    the half-open span the loop covers, or ``None`` when nothing repeats.

    Scoring is by covered length (``len(pattern) * repeats``) so a four-chord
    cycle heard eight times beats a two-chord cycle heard three times.  Ties go
    to the shorter pattern, which keeps ``I V I V`` from being reported as a
    four-chord loop.
    """
    n = len(labels)
    if n < min_length * 2:
        return None

    best: tuple[list[str], int, int, int] | None = None
    best_score = 0

    for length in range(min_length, min(max_length, n // 2) + 1):
        start = 0
        while start + length * 2 <= n:
            pattern = labels[start : start + length]
            repeats = 1
            while (
                start + length * (repeats + 1) <= n
                and labels[start + length * repeats : start + length * (repeats + 1)] == pattern
            ):
                repeats += 1

            if repeats >= 2:
                score = length * repeats
                if score > best_score:
                    best_score = score
                    best = (pattern, repeats, start, start + length * repeats)
                start += length * repeats
            else:
                start += 1

    return best


def summarise_progression(
    chords: list[ChordSegment], key: KeyEstimate, include_no_chord: bool = False
) -> ProgressionSummary:
    """Turn a chord timeline into Roman numerals plus the dominant loop."""
    usable = [c for c in chords if include_no_chord or not c.is_no_chord]
    if not usable:
        return ProgressionSummary(roman=[], labels=[], main_loop=None)

    labels = [c.label for c in usable]
    roman = [roman_numeral(label, key.tonic, key.mode) for label in labels]

    groups = _collapse_repeats(usable)
    loop_hit = find_main_loop([label for label, _, _ in groups])

    main_loop: Loop | None = None
    if loop_hit is not None:
        pattern, repeats, start_index, end_index = loop_hit
        main_loop = Loop(
            labels=list(pattern),
            roman=[roman_numeral(label, key.tonic, key.mode) for label in pattern],
            repeats=repeats,
            start=groups[start_index][1],
            end=groups[end_index - 1][2],
        )

    return ProgressionSummary(roman=roman, labels=labels, main_loop=main_loop)


def chord_histogram(chords: list[ChordSegment]) -> list[tuple[str, float]]:
    """Total sounding time per chord label, most prominent first."""
    totals: Counter[str] = Counter()
    for chord in chords:
        if chord.is_no_chord:
            continue
        totals[chord.label] += chord.duration
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def harmonic_rhythm(chords: list[ChordSegment]) -> float | None:
    """Median chord duration in seconds — a rough harmonic-rhythm readout."""
    durations = [c.duration for c in chords if not c.is_no_chord]
    if not durations:
        return None
    return float(np.median(durations))
