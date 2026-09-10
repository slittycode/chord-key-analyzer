"""Accuracy-harness tests.

The reference annotations here are written by hand from what the fixture
*renders*, never from what the analyzer *reports* — a ground truth derived from
the thing under test would score 1.0 by construction and measure nothing.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mir_eval", reason="requires the [eval] extra")

import fixtures as fx  # noqa: E402
from chord_key_analyzer.evaluate import (  # noqa: E402
    CHORD_METRICS,
    TrackEvaluation,
    discover_pairs,
    evaluate_track,
    load_key_reference,
    summarise,
)

#: POP_LOOP_C rendered at 2 s per chord, three times: C G Am F, C G Am F, C G Am F.
POP_LAB_CHORDS = ["C:maj", "G:maj", "A:min", "F:maj"]


def write_pop_lab(path, chord_duration=2.0, repeats=3):
    """Hand-written ground truth for a POP_LOOP_C rendering."""
    lines = []
    time = 0.0
    for _ in range(repeats):
        for label in POP_LAB_CHORDS:
            lines.append(f"{time:.6f}\t{time + chord_duration:.6f}\t{label}")
            time += chord_duration
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def pop_dataset(tmp_path):
    """A one-track dataset: audio, hand-written chords, and a key."""
    audio = fx.render_progression(fx.POP_LOOP_C, chord_duration=2.0, repeats=3)
    fx.write_wav(tmp_path / "pop.wav", audio)
    write_pop_lab(tmp_path / "pop.lab")
    (tmp_path / "pop.key").write_text("C major\n", encoding="utf-8")
    return tmp_path


def test_every_vocabulary_label_parses_in_mir_eval():
    """Proof the exported labels are the dialect mir_eval reads."""
    import mir_eval

    from chord_key_analyzer.chords import CHORD_QUALITIES, chord_label

    for root in range(12):
        for quality in CHORD_QUALITIES:
            mir_eval.chord.encode(chord_label(root, quality))
    mir_eval.chord.encode("N")


def test_every_slash_label_parses_in_mir_eval():
    """Same proof, extended over every chord tone a slash can name."""
    import mir_eval

    from chord_key_analyzer.models import CHORD_QUALITIES, ChordSegment, chord_label

    for root in range(12):
        for quality, intervals in CHORD_QUALITIES.items():
            for interval in intervals:
                bass = fx.PITCH_CLASSES[(root + interval) % 12]
                segment = ChordSegment(0.0, 1.0, chord_label(root, quality), 1.0, bass=bass)
                mir_eval.chord.encode(segment.mirex_label)


def test_a_slash_reference_still_scores_the_plain_estimate(tmp_path):
    """Bass is not part of what root/majmin measure, so a reference written with
    inversions must not penalise an analyzer that reports the chord alone."""
    audio = fx.render_progression(fx.POP_LOOP_C, chord_duration=2.0, repeats=3)
    fx.write_wav(tmp_path / "pop.wav", audio)

    lines, time = [], 0.0
    for _ in range(3):
        for label in ["C:maj/3", "G:maj/5", "A:min/b3", "F:maj/3"]:
            lines.append(f"{time:.6f}\t{time + 2.0:.6f}\t{label}")
            time += 2.0
    (tmp_path / "pop.lab").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pairs, _ = discover_pairs(tmp_path)
    evaluation = evaluate_track(pairs[0])
    assert evaluation.ok, evaluation.error
    assert evaluation.chord_scores["root"] >= 0.75, evaluation.chord_scores
    assert evaluation.chord_scores["majmin"] >= 0.75, evaluation.chord_scores


def test_evaluate_track_scores_the_pop_fixture(pop_dataset):
    """End-to-end: a known rendering against hand-written truth must score high."""
    pairs, orphans = discover_pairs(pop_dataset)
    assert orphans == []
    assert len(pairs) == 1

    evaluation = evaluate_track(pairs[0])
    assert evaluation.ok, evaluation.error
    assert evaluation.chord_scores["majmin"] >= 0.75, evaluation.chord_scores
    assert evaluation.chord_scores["root"] >= 0.75, evaluation.chord_scores
    assert evaluation.key_score == 1.0
    assert evaluation.reference_key == "C major"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("C major\n", "C major"),
        ("C\n", "C major"),
        ("E:minor\n", "E minor"),
        ("f# minor\n", "F# minor"),
        ("Bb\n", "Bb major"),
        ("A:min\n", "A minor"),
        ("nonsense\n", None),
        ("G:mixolydian\n", None),
        ("\n", None),
    ],
)
def test_load_key_reference_formats(tmp_path, text, expected):
    path = tmp_path / "song.key"
    path.write_text(text, encoding="utf-8")
    assert load_key_reference(path) == expected


def test_load_key_reference_reads_isophonics_segments(tmp_path):
    """Longest tonality wins; Silence segments carry no key."""
    path = tmp_path / "song.key.lab"
    path.write_text(
        "0.000000\t1.500000\tSilence\n1.500000\t10.000000\tKey\tG\n10.000000\t200.000000\tKey\tE\n",
        encoding="utf-8",
    )
    assert load_key_reference(path) == "E major"


def test_discovery_is_recursive_and_matches_by_stem(tmp_path):
    nested = tmp_path / "album" / "disc1"
    nested.mkdir(parents=True)
    fx.write_wav(nested / "track.wav", fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
    write_pop_lab(nested / "track.lab", chord_duration=0.5, repeats=1)

    pairs, orphans = discover_pairs(tmp_path)
    assert orphans == []
    assert len(pairs) == 1
    assert pairs[0].name == str(nested.relative_to(tmp_path) / "track")
    assert pairs[0].audio.name == "track.wav"


def test_key_lab_is_not_mistaken_for_a_chord_annotation(tmp_path):
    fx.write_wav(tmp_path / "song.wav", fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
    write_pop_lab(tmp_path / "song.lab", chord_duration=0.5, repeats=1)
    (tmp_path / "song.key.lab").write_text("0.0\t10.0\tKey\tC\n", encoding="utf-8")

    pairs, orphans = discover_pairs(tmp_path)
    assert orphans == []
    assert len(pairs) == 1
    assert pairs[0].chords.name == "song.lab"
    assert pairs[0].key is not None and pairs[0].key.name == "song.key.lab"


def test_discovery_matches_case_insensitive_extensions(tmp_path):
    """`Track.LAB` beside `Track.WAV` is a pair, not two orphans."""
    fx.write_wav(tmp_path / "song.WAV", fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
    write_pop_lab(tmp_path / "song.LAB", chord_duration=0.5, repeats=1)
    (tmp_path / "song.KEY").write_text("C major\n", encoding="utf-8")

    pairs, orphans = discover_pairs(tmp_path)
    assert orphans == []
    assert len(pairs) == 1
    assert pairs[0].audio.name == "song.WAV"
    assert pairs[0].key is not None and pairs[0].key.name == "song.KEY"


def test_discovery_matches_a_mixed_case_audio_extension(tmp_path):
    fx.write_wav(tmp_path / "song.Wav", fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
    write_pop_lab(tmp_path / "song.lab", chord_duration=0.5, repeats=1)

    pairs, orphans = discover_pairs(tmp_path)
    assert orphans == []
    assert [p.audio.name for p in pairs] == ["song.Wav"]


def test_a_lab_without_audio_is_skipped_not_fatal(tmp_path):
    fx.write_wav(tmp_path / "good.wav", fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
    write_pop_lab(tmp_path / "good.lab", chord_duration=0.5, repeats=1)
    write_pop_lab(tmp_path / "orphan.lab", chord_duration=0.5, repeats=1)

    pairs, orphans = discover_pairs(tmp_path)
    assert [p.name for p in pairs] == ["good"]
    assert [o.name for o in orphans] == ["orphan.lab"]


def test_summary_weights_chord_metrics_by_duration():
    """10 s at 1.0 and 30 s at 0.5 is 0.625, not the 0.75 a plain mean gives."""
    tracks = [
        TrackEvaluation(
            name="short", duration=10.0, chord_scores=dict.fromkeys(CHORD_METRICS, 1.0)
        ),
        TrackEvaluation(name="long", duration=30.0, chord_scores=dict.fromkeys(CHORD_METRICS, 0.5)),
    ]
    summary = summarise(tracks)
    assert summary["majmin"] == pytest.approx(0.625)
    assert summary["scored"] == 2
    assert summary["failed"] == 0


def test_summary_counts_failures_and_averages_keys():
    tracks = [
        TrackEvaluation(name="ok", duration=10.0, chord_scores={"majmin": 0.8}, key_score=1.0),
        TrackEvaluation(name="bad", error="boom"),
    ]
    summary = summarise(tracks)
    assert (summary["scored"], summary["failed"]) == (1, 1)
    assert summary["key"] == pytest.approx(1.0)
    assert summary["key_tracks"] == 1


def test_undecodable_audio_is_recorded_not_raised(tmp_path):
    """One bad file must not take the corpus run down with it."""
    (tmp_path / "junk.wav").write_bytes(b"definitely not audio")
    write_pop_lab(tmp_path / "junk.lab", chord_duration=0.5, repeats=1)

    pairs, _ = discover_pairs(tmp_path)
    evaluation = evaluate_track(pairs[0])
    assert not evaluation.ok
    assert evaluation.error


def test_an_unparseable_reference_label_is_recorded_not_raised(tmp_path):
    """A bad line in a hand-made .lab must cost that track, not the corpus.

    `load_labeled_intervals` reads a label file without validating it, so the
    first thing to parse `C:notachord` is the scorer — and it raises from
    `Exception`, not `ValueError`, which is what made this escape the handler.
    """
    fx.write_wav(tmp_path / "good.wav", fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
    write_pop_lab(tmp_path / "good.lab", chord_duration=0.5, repeats=1)
    fx.write_wav(tmp_path / "bad.wav", fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
    (tmp_path / "bad.lab").write_text("0.000000\t2.000000\tC:notachord\n", encoding="utf-8")

    pairs, _ = discover_pairs(tmp_path)
    tracks = [evaluate_track(pair) for pair in pairs]
    by_name = {track.name: track for track in tracks}

    assert not by_name["bad"].ok
    assert "chord scoring failed" in by_name["bad"].error
    assert by_name["good"].ok, by_name["good"].error

    summary = summarise(tracks)
    assert (summary["scored"], summary["failed"]) == (1, 1)


def test_key_scoring_failure_is_recorded_not_raised(pop_dataset, monkeypatch):
    """mir_eval owns the last word on a key string; its verdict must not be fatal."""
    import mir_eval

    def boom(reference, estimate):
        raise ValueError("Key H is invalid")

    monkeypatch.setattr(mir_eval.key, "weighted_score", boom)

    pairs, _ = discover_pairs(pop_dataset)
    evaluation = evaluate_track(pairs[0])
    assert not evaluation.ok
    assert "key scoring failed" in evaluation.error


def test_empty_chord_result_still_scores(tmp_path, monkeypatch):
    """A track the analyzer finds no chords in scores 0, it does not crash."""
    audio = fx.render_progression(fx.POP_LOOP_C, 1.0, 1)
    fx.write_wav(tmp_path / "s.wav", audio)
    write_pop_lab(tmp_path / "s.lab", chord_duration=1.0, repeats=1)

    # evaluate_track imports analyze_source lazily, so patch it at its source.
    class EmptyResult:
        duration = 4.0
        chords: list = []

        class key:
            name = "C major"

    monkeypatch.setattr("chord_key_analyzer.pipeline.analyze_source", lambda *a, **k: EmptyResult())
    pairs, _ = discover_pairs(tmp_path)
    evaluation = evaluate_track(pairs[0])
    assert evaluation.ok, evaluation.error
    assert evaluation.chord_scores["majmin"] == pytest.approx(0.0)


def test_cli_eval_end_to_end(pop_dataset, tmp_path):
    from click.testing import CliRunner

    from chord_key_analyzer.cli import main

    destination = tmp_path / "eval.json"
    result = CliRunner().invoke(
        main, ["eval", str(pop_dataset), "--json", str(destination), "--quiet"]
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(destination.read_text())
    assert "summary" in payload and "tracks" in payload
    assert payload["summary"]["scored"] == 1
    assert payload["summary"]["majmin"] >= 0.75, payload["summary"]
    assert payload["tracks"][0]["name"] == "pop"


def test_cli_eval_writes_csv(pop_dataset, tmp_path):
    from click.testing import CliRunner

    from chord_key_analyzer.cli import main

    destination = tmp_path / "eval.csv"
    result = CliRunner().invoke(
        main, ["eval", str(pop_dataset), "--csv", str(destination), "--quiet"]
    )
    assert result.exit_code == 0, result.output

    lines = destination.read_text().strip().splitlines()
    assert lines[0].startswith("name,duration")
    assert lines[-1].startswith("__summary__")


def test_cli_eval_renders_a_table(pop_dataset):
    from click.testing import CliRunner

    from chord_key_analyzer.cli import main

    result = CliRunner().invoke(main, ["eval", str(pop_dataset)])
    assert result.exit_code == 0, result.output
    assert "MIREX" in result.output


def test_cli_eval_on_an_empty_dataset_explains_the_layout(tmp_path):
    from click.testing import CliRunner

    from chord_key_analyzer.cli import main

    result = CliRunner().invoke(main, ["eval", str(tmp_path)])
    assert result.exit_code == 2
    assert ".lab" in result.output
    assert "no evaluable tracks" in result.output.lower()
