"""Chord recognition tests."""

from __future__ import annotations

import numpy as np
import pytest

import fixtures as fx
from chord_key_analyzer.chords import (
    TemplateHMMEngine,
    build_chord_templates,
    get_engine,
    snap_to_beats,
    viterbi_decode,
)
from chord_key_analyzer.features import extract_features
from chord_key_analyzer.models import CHORD_QUALITIES, NO_CHORD, ChordSegment

#: Boundary tolerance we hold the decoder to, in seconds.
BOUNDARY_TOLERANCE = 0.25


def expected_labels(progression, repeats):
    """Labels after adjacent duplicates merge, which is what the decoder emits."""
    labels = []
    for root, quality in progression * repeats:
        label = f"{root}:{quality}"
        if not labels or labels[-1] != label:
            labels.append(label)
    return labels


def decode(progression, chord_duration=2.0, repeats=3, **kwargs):
    audio = fx.render_progression(progression, chord_duration=chord_duration, repeats=repeats)
    features = extract_features(audio, fx.SR)
    return TemplateHMMEngine(**kwargs).analyze(features)


@pytest.mark.parametrize(
    ("name", "progression", "repeats"),
    [
        ("pop", fx.POP_LOOP_C, 3),
        ("minor", fx.MINOR_LOOP_A, 3),
        ("blues", fx.BLUES_TURNAROUND_C, 3),
        ("jazz", fx.JAZZ_TURNAROUND_C, 4),
        ("sevenths", [("C", "maj7"), ("A", "min7"), ("D", "min7"), ("G", "7")], 3),
        ("dim/aug", [("C", "maj"), ("B", "dim"), ("C", "aug"), ("F", "maj")], 3),
    ],
)
def test_recognises_progressions(name, progression, repeats):
    segments = decode(progression, repeats=repeats)
    assert [s.label for s in segments] == expected_labels(progression, repeats)


def test_seventh_chords_are_not_flattened_to_triads():
    """Regression: equal-weighted templates made min7/maj7 lose to their triads."""
    segments = decode(fx.JAZZ_TURNAROUND_C, repeats=3)
    labels = {s.label for s in segments}
    assert "D:min7" in labels
    assert "C:maj7" in labels
    assert "G:7" in labels


def test_chord_boundaries_land_within_tolerance():
    segments = decode(fx.POP_LOOP_C, chord_duration=2.0, repeats=3)
    expected = [2.0 * i for i in range(1, 12)]
    boundaries = [s.start for s in segments[1:]]
    assert len(boundaries) == len(expected)
    for got, want in zip(boundaries, expected, strict=True):
        assert abs(got - want) <= BOUNDARY_TOLERANCE, f"{got:.3f} vs {want:.3f}"


def test_segments_are_contiguous_and_ordered():
    segments = decode(fx.POP_LOOP_C, repeats=2)
    for previous, following in zip(segments, segments[1:], strict=False):
        assert previous.end == pytest.approx(following.start, abs=1e-6)
        assert previous.start < previous.end


def test_adjacent_duplicate_labels_are_merged():
    segments = decode(fx.BLUES_TURNAROUND_C, repeats=2)
    for previous, following in zip(segments, segments[1:], strict=False):
        assert previous.label != following.label


def test_silence_becomes_no_chord():
    audio = np.concatenate(
        [
            fx.render_silence(1.5),
            fx.render_progression(fx.POP_LOOP_C, chord_duration=2.0, repeats=1),
            fx.render_silence(1.5),
        ]
    ).astype(np.float32)
    segments = TemplateHMMEngine().analyze(extract_features(audio, fx.SR))

    assert segments[0].label == NO_CHORD
    assert segments[-1].label == NO_CHORD
    assert segments[0].end == pytest.approx(1.5, abs=0.3)
    assert [s.label for s in segments if s.label != NO_CHORD] == [
        "C:maj",
        "G:maj",
        "A:min",
        "F:maj",
    ]


def test_triads_only_never_emits_sevenths():
    segments = decode(fx.JAZZ_TURNAROUND_C, repeats=3, triads_only=True)
    for segment in segments:
        if segment.label != NO_CHORD:
            assert segment.label.split(":")[1] in {"maj", "min", "dim", "aug"}


def test_confidence_is_bounded():
    for segment in decode(fx.POP_LOOP_C, repeats=2):
        assert 0.0 <= segment.confidence <= 1.0


# --- templates --------------------------------------------------------------


def test_template_matrix_is_row_normalised():
    templates, labels = build_chord_templates()
    assert templates.shape == (len(labels), 12)
    assert np.allclose(np.linalg.norm(templates, axis=1), 1.0)
    assert labels[-1] == NO_CHORD


def test_enharmonic_augmented_duplicates_are_collapsed():
    """C:aug, E:aug and G#:aug are one pitch-class set; chroma cannot separate
    them, so only the canonical spelling should be a state."""
    _, labels = build_chord_templates()
    aug = [label for label in labels if label.endswith(":aug")]
    assert len(aug) == 4
    assert aug == ["C:aug", "C#:aug", "D:aug", "D#:aug"]


def test_state_count_matches_vocabulary():
    _, labels = build_chord_templates()
    # 12 roots x 7 qualities, minus 8 duplicate augmented spellings, plus N.
    assert len(labels) == 12 * len(CHORD_QUALITIES) - 8 + 1


def test_triads_only_template_set_is_smaller():
    _, full = build_chord_templates()
    _, triads = build_chord_templates(("maj", "min", "dim", "aug"))
    assert len(triads) < len(full)


# --- viterbi ----------------------------------------------------------------


def test_viterbi_follows_a_clear_signal():
    emissions = np.array([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    assert list(viterbi_decode(emissions, self_prob=0.6)) == [0, 0, 1, 1]


def test_viterbi_suppresses_a_single_frame_flicker():
    emissions = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 3.1], [3.0, 0.0], [3.0, 0.0]])
    assert list(viterbi_decode(emissions, self_prob=0.995)) == [0, 0, 0, 0, 0]


def test_viterbi_handles_degenerate_inputs():
    assert viterbi_decode(np.zeros((0, 5))).size == 0
    assert list(viterbi_decode(np.zeros((3, 1)))) == [0, 0, 0]


def test_viterbi_path_length_matches_input():
    rng = np.random.default_rng(0)
    emissions = rng.normal(size=(40, 12))
    assert viterbi_decode(emissions).shape == (40,)


# --- beat snapping ----------------------------------------------------------


def test_snapping_moves_only_nearby_boundaries():
    segments = [
        ChordSegment(0.0, 1.9, "C:maj"),
        ChordSegment(1.9, 4.0, "G:maj"),
    ]
    beats = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    snapped = snap_to_beats(segments, beats, tolerance=0.3)
    assert snapped[0].end == pytest.approx(2.0)
    assert snapped[1].start == pytest.approx(2.0)


def test_snapping_leaves_distant_boundaries_alone():
    segments = [ChordSegment(0.0, 1.5, "C:maj"), ChordSegment(1.5, 4.0, "G:maj")]
    beats = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    snapped = snap_to_beats(segments, beats, tolerance=0.2)
    assert snapped[0].end == pytest.approx(1.5)


def test_snapping_keeps_boundaries_increasing():
    segments = [
        ChordSegment(0.0, 1.0, "C:maj"),
        ChordSegment(1.0, 1.1, "G:maj"),
        ChordSegment(1.1, 3.0, "F:maj"),
    ]
    snapped = snap_to_beats(segments, np.array([0.0, 1.05, 2.0]), tolerance=0.5)
    for previous, following in zip(snapped, snapped[1:], strict=False):
        assert previous.end <= following.start
        assert following.start < following.end


def test_disabling_beat_snap_still_decodes_correctly():
    segments = decode(fx.POP_LOOP_C, repeats=2, beat_snap=False)
    assert [s.label for s in segments] == expected_labels(fx.POP_LOOP_C, 2)


# --- engine registry --------------------------------------------------------


def test_get_engine_returns_the_template_engine():
    assert get_engine("template").name == "template"


def test_unimplemented_engines_fail_loudly():
    """--engine crema must not silently fall back to the template engine."""
    with pytest.raises(ValueError, match="not implemented"):
        get_engine("crema")


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine("nonsense")
