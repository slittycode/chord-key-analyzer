"""Roman-numeral analysis and loop detection tests."""

from __future__ import annotations

import pytest

import fixtures as fx
from chord_key_analyzer.models import ChordSegment, KeyEstimate
from chord_key_analyzer.progression import (
    chord_histogram,
    find_main_loop,
    harmonic_rhythm,
    roman_numeral,
    summarise_progression,
)


def key(tonic="C", mode="major"):
    return KeyEstimate(tonic=tonic, mode=mode, confidence=0.9)


def segments(labels, duration=2.0):
    return [
        ChordSegment(i * duration, (i + 1) * duration, label, 0.8)
        for i, label in enumerate(labels)
    ]


# --- roman numerals ---------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("C:maj", "I"),
        ("D:min", "ii"),
        ("E:min", "iii"),
        ("F:maj", "IV"),
        ("G:maj", "V"),
        ("A:min", "vi"),
        ("B:dim", "vii°"),
        ("G:7", "V7"),
        ("C:maj7", "Imaj7"),
        ("D:min7", "ii7"),
        ("C:aug", "I+"),
        ("A#:maj", "bVII"),
        ("D#:maj", "bIII"),
        ("G#:maj", "bVI"),
        ("C#:maj", "bII"),
    ],
)
def test_major_key_numerals(label, expected):
    assert roman_numeral(label, "C", "major") == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("A:min", "i"),
        ("B:dim", "ii°"),
        ("C:maj", "III"),
        ("D:min", "iv"),
        ("E:min", "v"),
        ("E:maj", "V"),
        ("E:7", "V7"),
        ("F:maj", "VI"),
        ("G:maj", "VII"),
        ("A:min7", "i7"),
    ],
)
def test_minor_key_numerals(label, expected):
    assert roman_numeral(label, "A", "minor") == expected


def test_borrowed_chords_are_labelled_literally():
    """bVII in a major key is information, not something to snap to a scale degree."""
    assert roman_numeral("A#:maj", "C", "major") == "bVII"
    assert roman_numeral("D#:maj", "C", "major") == "bIII"


def test_no_chord_passes_through():
    assert roman_numeral("N", "C", "major") == "N"


def test_unknown_tonic_returns_the_label():
    assert roman_numeral("C:maj", "H", "major") == "C:maj"


# --- loop detection ---------------------------------------------------------


def test_finds_a_repeating_four_chord_loop():
    labels = ["C", "G", "Am", "F"] * 4
    found = find_main_loop(labels)
    assert found is not None
    pattern, repeats, start, end = found
    assert pattern == ["C", "G", "Am", "F"]
    assert repeats == 4
    assert (start, end) == (0, 16)


def test_prefers_the_shorter_pattern_on_a_tie():
    """I V I V is a two-chord loop heard twice, not a four-chord loop heard once."""
    found = find_main_loop(["I", "V", "I", "V"])
    assert found is not None
    assert found[0] == ["I", "V"]
    assert found[1] == 2


def test_prefers_wider_coverage_over_a_shorter_cycle():
    labels = ["A", "B", "C", "D"] * 4
    found = find_main_loop(labels)
    assert found is not None
    assert found[0] == ["A", "B", "C", "D"]


def test_no_loop_in_a_through_composed_sequence():
    assert find_main_loop(["A", "B", "C", "D", "E", "F", "G", "H"]) is None


def test_no_loop_in_a_too_short_sequence():
    assert find_main_loop(["A"]) is None


# --- summary ----------------------------------------------------------------


def test_summary_maps_a_pop_loop_to_roman_numerals():
    summary = summarise_progression(segments(["C:maj", "G:maj", "A:min", "F:maj"] * 3), key())
    assert summary.roman[:4] == ["I", "V", "vi", "IV"]
    assert summary.main_loop is not None
    assert summary.main_loop.roman == ["I", "V", "vi", "IV"]
    assert summary.main_loop.repeats == 3


def test_summary_loop_timings_span_the_repeats():
    chords = segments(["C:maj", "G:maj", "A:min", "F:maj"] * 3, duration=2.0)
    summary = summarise_progression(chords, key())
    assert summary.main_loop.start == pytest.approx(0.0)
    assert summary.main_loop.end == pytest.approx(24.0)


def test_summary_excludes_no_chord_by_default():
    chords = segments(["N", "C:maj", "G:maj", "N"])
    summary = summarise_progression(chords, key())
    assert "N" not in summary.labels
    assert summary.labels == ["C:maj", "G:maj"]


def test_summary_of_empty_input():
    summary = summarise_progression([], key())
    assert summary.roman == []
    assert summary.main_loop is None


def test_summary_in_a_minor_key():
    summary = summarise_progression(
        segments(["A:min", "F:maj", "C:maj", "G:maj"] * 2), key("A", "minor")
    )
    assert summary.roman[:4] == ["i", "VI", "III", "VII"]


def test_histogram_ranks_by_total_time():
    chords = [
        ChordSegment(0, 4, "C:maj"),
        ChordSegment(4, 5, "G:maj"),
        ChordSegment(5, 11, "F:maj"),
        ChordSegment(11, 12, "N"),
    ]
    assert [label for label, _ in chord_histogram(chords)] == ["F:maj", "C:maj", "G:maj"]


def test_harmonic_rhythm_is_the_median_chord_length():
    chords = [
        ChordSegment(0, 2, "C:maj"),
        ChordSegment(2, 4, "G:maj"),
        ChordSegment(4, 10, "F:maj"),
    ]
    assert harmonic_rhythm(chords) == pytest.approx(2.0)


def test_harmonic_rhythm_of_silence_is_none():
    assert harmonic_rhythm([ChordSegment(0, 5, "N")]) is None


# --- end to end -------------------------------------------------------------


def test_pipeline_reports_the_pop_loop(pop_result):
    assert pop_result.key.name == "C major"
    loop = pop_result.progression.main_loop
    assert loop is not None
    assert loop.roman == ["I", "V", "vi", "IV"]
    assert loop.labels == ["C:maj", "G:maj", "A:min", "F:maj"]
    assert loop.repeats == 3


def test_pipeline_reports_a_jazz_turnaround():
    from chord_key_analyzer.pipeline import analyze_audio

    audio = fx.render_progression(fx.JAZZ_TURNAROUND_C, chord_duration=2.0, repeats=4)
    result = analyze_audio(audio, sr=fx.SR, source="jazz")
    assert result.progression.main_loop is not None
    assert result.progression.main_loop.roman == ["ii7", "V7", "Imaj7"]
