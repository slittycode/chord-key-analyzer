"""Output layer tests: JSON schema, .lab export, terminal rendering."""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from chord_key_analyzer.models import (
    SCHEMA_VERSION,
    AnalysisResult,
    ChordSegment,
    KeyCandidate,
    KeyEstimate,
    Loop,
    Modulation,
    ProgressionSummary,
    chord_label,
    parse_chord_label,
)
from chord_key_analyzer.output import format_time, render, to_json, to_lab, write_json, write_lab


def sample_result():
    return AnalysisResult(
        file="song.wav",
        duration=8.0,
        key=KeyEstimate(
            tonic="C",
            mode="major",
            confidence=0.82,
            alternatives=[KeyCandidate("A", "minor", 0.6)],
            modulations=[Modulation(4.0, 8.0, "G", "major", 0.5)],
        ),
        chords=[
            ChordSegment(0.0, 2.0, "C:maj", 0.9),
            ChordSegment(2.0, 4.0, "G:maj", 0.8, bass="B"),
            ChordSegment(4.0, 6.0, "N", 0.3),
            ChordSegment(6.0, 8.0, "F:maj", 0.7),
        ],
        progression=ProgressionSummary(
            roman=["I", "V", "IV"],
            labels=["C:maj", "G:maj", "F:maj"],
            main_loop=Loop(["C:maj", "G:maj"], ["I", "V"], 2, 0.0, 4.0),
        ),
        tempo=120.0,
        meta={"engine": "template", "version": "0.1.0"},
    )


# --- JSON -------------------------------------------------------------------


def test_json_round_trips():
    payload = json.loads(to_json(sample_result()))
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["file"] == "song.wav"
    assert payload["duration"] == 8.0
    assert payload["tempo"] == 120.0


def test_json_top_level_shape_is_stable():
    """The schema is a published contract; adding a key is fine, losing one is not."""
    payload = json.loads(to_json(sample_result()))
    assert set(payload) == {
        "schema",
        "file",
        "duration",
        "tempo",
        "key",
        "chords",
        "progression",
        "meta",
    }
    assert set(payload["key"]) == {
        "tonic",
        "mode",
        "confidence",
        "alternatives",
        "modulations",
    }
    assert set(payload["chords"][0]) == {"start", "end", "label", "confidence", "bass"}
    assert set(payload["progression"]) == {"roman", "labels", "main_loop"}
    assert set(payload["progression"]["main_loop"]) == {
        "labels",
        "roman",
        "repeats",
        "start",
        "end",
    }


def test_json_includes_modulations_and_alternatives():
    payload = json.loads(to_json(sample_result()))
    assert payload["key"]["alternatives"] == [{"tonic": "A", "mode": "minor", "score": 0.6}]
    assert payload["key"]["modulations"][0]["tonic"] == "G"


def test_json_written_to_a_file(tmp_path):
    destination = tmp_path / "out.json"
    write_json(sample_result(), str(destination))
    assert json.loads(destination.read_text())["file"] == "song.wav"


def test_json_written_to_stdout(capsys):
    write_json(sample_result(), "-")
    assert json.loads(capsys.readouterr().out)["file"] == "song.wav"


def test_pipeline_result_serialises(pop_result):
    payload = json.loads(to_json(pop_result))
    assert payload["key"]["tonic"] == "C"
    assert payload["meta"]["engine"] == "template"
    assert len(payload["chords"]) == 12


# --- .lab -------------------------------------------------------------------


def test_lab_format_is_mirex_style():
    lines = to_lab(sample_result()).strip().split("\n")
    assert lines[0] == "0.000\t2.000\tC:maj"
    assert len(lines) == 4


def test_lab_writes_a_bass_as_a_degree_slash():
    """`G:maj/B` is not a label mir_eval parses; `G:maj/3` is."""
    assert "\tG:maj/3\n" in to_lab(sample_result())


def test_json_carries_the_bass_as_a_note_name():
    """The JSON field is for consumers, who want the note, not the degree."""
    payload = json.loads(to_json(sample_result()))
    assert [c["bass"] for c in payload["chords"]] == [None, "B", None, None]
    assert payload["chords"][1]["label"] == "G:maj", "the label itself stays plain"


def test_render_shows_a_slash_chord_by_note_name():
    console = Console(record=True, width=100)
    render(sample_result(), console=console)
    assert "G:maj/B" in console.export_text()


def test_lab_keeps_the_no_chord_state():
    """MIREX annotations mark silence explicitly rather than leaving gaps."""
    assert "\tN" in to_lab(sample_result())


def test_lab_written_to_a_file(tmp_path):
    destination = tmp_path / "out.lab"
    write_lab(sample_result(), str(destination))
    assert destination.read_text().startswith("0.000")


def test_lab_written_to_stdout(capsys):
    write_lab(sample_result(), "-")
    assert capsys.readouterr().out.startswith("0.000")


# --- terminal ---------------------------------------------------------------


def test_render_mentions_key_chords_and_loop():
    console = Console(record=True, width=100)
    render(sample_result(), console=console)
    text = console.export_text()
    assert "C major" in text
    assert "C:maj" in text
    assert "I" in text


def test_render_handles_an_empty_chord_track():
    result = AnalysisResult(
        file="quiet.wav",
        duration=1.0,
        key=KeyEstimate("C", "major", 0.5),
        chords=[],
        progression=ProgressionSummary([], [], None),
    )
    console = Console(record=True, width=100)
    render(result, console=console)
    assert "No chords detected" in console.export_text()


def test_render_truncates_a_long_timeline():
    chords = [ChordSegment(i, i + 1, "C:maj", 0.5) for i in range(120)]
    result = AnalysisResult(
        file="long.wav",
        duration=120.0,
        key=KeyEstimate("C", "major", 0.9),
        chords=chords,
        progression=ProgressionSummary(["I"] * 120, ["C:maj"] * 120, None),
    )
    console = Console(record=True, width=100)
    render(result, console=console)
    assert "more segments" in console.export_text()


def test_render_a_real_result(pop_result):
    console = Console(record=True, width=120)
    render(pop_result, console=console)
    assert "C major" in console.export_text()


@pytest.mark.parametrize(
    ("seconds", "expected"), [(0.0, "0:00.0"), (5.25, "0:05.2"), (65.0, "1:05.0"), (-1.0, "0:00.0")]
)
def test_time_formatting(seconds, expected):
    assert format_time(seconds) == expected


# --- models -----------------------------------------------------------------


def test_chord_label_round_trip():
    for root in range(12):
        for quality in ("maj", "min", "min7"):
            label = chord_label(root, quality)
            assert parse_chord_label(label) == (root, quality)


def test_no_chord_does_not_parse():
    assert parse_chord_label("N") is None


def test_malformed_label_does_not_parse():
    assert parse_chord_label("H:maj") is None
    assert parse_chord_label("garbage") is None


def test_segment_duration_and_no_chord_flag():
    assert ChordSegment(1.0, 3.5, "C:maj").duration == pytest.approx(2.5)
    assert ChordSegment(0, 1, "N").is_no_chord
    assert not ChordSegment(0, 1, "C:maj").is_no_chord
