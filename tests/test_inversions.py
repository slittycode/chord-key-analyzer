"""Bass salience and inversion (slash-chord) detection.

Two kinds of test here, deliberately.  The renderings exercise the whole path —
audio in, slash label out — and the hand-built :class:`Features` pin each of the
three thresholds on its own, without asking a DSP front-end to land exactly on a
boundary for the assertion to mean anything.
"""

from __future__ import annotations

import numpy as np
import pytest

import fixtures as fx
from chord_key_analyzer.chords import TemplateHMMEngine, _coalesce, detect_inversions
from chord_key_analyzer.features import Features, extract_features
from chord_key_analyzer.models import PITCH_CLASSES, ChordSegment


def analyse(chords, chord_duration=1.5, repeats=3):
    audio = fx.render_progression(chords, chord_duration=chord_duration, repeats=repeats)
    features = extract_features(audio, fx.SR)
    return detect_inversions(TemplateHMMEngine().analyze(features), features)


def longest_pitched(segments):
    return max((s for s in segments if not s.is_no_chord), key=lambda s: s.duration)


def bass_above(root: str, interval: int) -> str:
    return PITCH_CLASSES[(PITCH_CLASSES.index(root) + interval) % 12]


# --- rendered audio ---------------------------------------------------------


@pytest.mark.parametrize(
    ("root", "quality", "interval", "degree"),
    [
        *[(r, "maj", 4, "3") for r in ("C", "F", "G", "D")],
        *[(r, "min", 3, "b3") for r in ("A", "E", "D", "B")],
        *[(r, "maj", 7, "5") for r in ("C", "F", "G", "D")],
        *[(r, "min", 7, "5") for r in ("A", "E", "D", "B")],
        *[(r, "maj7", 11, "7") for r in ("F", "G", "A", "B")],
    ],
)
def test_inversions_are_detected_and_written_as_degrees(root, quality, interval, degree):
    segment = longest_pitched(analyse([(root, quality, interval)]))
    expected = bass_above(root, interval)

    assert segment.label == f"{root}:{quality}", "the plain label must stay plain"
    assert segment.bass == expected
    assert segment.mirex_label == f"{root}:{quality}/{degree}"
    assert segment.display_label == f"{root}:{quality}/{expected}"


@pytest.mark.parametrize(
    ("root", "quality", "interval"),
    [
        ("C", "min7", 3),
        ("D", "min7", 3),
        ("C", "min7", 10),
        ("A", "min7", 10),
        ("E", "maj7", 4),
    ],
)
def test_a_wrong_bass_is_never_reported(root, quality, interval):
    """The designed failure is no slash, not a slash naming the wrong note.

    These are the awkward renderings: with a b3 or a b7 in the bass the chroma
    often reads the chord as a different root altogether — C:min7 over Eb is
    Eb major with an added sixth, and no pitch-class histogram can referee that.
    Once the root has moved, the sounding bass is frequently that chord's own
    root, and root position is exactly what gets reported.  Silence is the right
    answer; some third pitch class would not be.
    """
    segment = longest_pitched(analyse([(root, quality, interval)]))
    assert segment.bass in (bass_above(root, interval), None)


def test_root_position_chords_carry_no_bass():
    for segment in analyse(fx.POP_LOOP_C, chord_duration=2.0, repeats=2):
        assert segment.bass is None
        assert segment.mirex_label == segment.label


def test_a_render_with_no_bass_note_carries_no_bass():
    """Nothing sounding down there is nothing to be salient about."""
    audio = np.concatenate(
        [fx.render_chord(root, quality, 2.0, with_bass=False) for root, quality in fx.POP_LOOP_C]
        * 2
    ).astype(np.float32)
    features = extract_features(audio, fx.SR)

    for segment in detect_inversions(TemplateHMMEngine().analyze(features), features):
        assert segment.bass is None


def test_bass_chroma_is_frame_aligned_with_the_main_chroma(pop_features):
    assert pop_features.bass_chroma.shape == pop_features.chroma.shape


def test_bass_chroma_is_not_column_normalised(pop_features):
    """Normalising it would erase the very difference the salience test reads."""
    norms = np.linalg.norm(pop_features.bass_chroma, axis=0)
    assert not np.allclose(norms, 1.0)


# --- thresholds, pinned deterministically -----------------------------------


def features_with_bass(per_frame: list[int | None]) -> Features:
    """A :class:`Features` whose bass chroma says exactly what the test wants.

    ``None`` in a frame means an even spread across all twelve classes — the
    murky low end, whose salience is 1/12 and which must never yield a slash.
    """
    n_frames = len(per_frame)
    bass = np.full((12, n_frames), 1.0)
    for index, pitch_class in enumerate(per_frame):
        if pitch_class is not None:
            bass[:, index] = 0.05
            bass[pitch_class, index] = 1.0

    hop, sr = 2048, 22050
    return Features(
        chroma=np.zeros((12, n_frames)),
        bass_chroma=bass,
        times=np.arange(n_frames) * hop / sr,
        rms=np.ones(n_frames),
        silent=np.zeros(n_frames, dtype=bool),
        sr=sr,
        hop_length=hop,
        tuning=0.0,
        tempo=None,
        beat_times=np.array([], dtype=float),
        beats_reliable=False,
    )


def detect_one(label: str, per_frame: list[int | None]) -> ChordSegment:
    features = features_with_bass(per_frame)
    span = ChordSegment(0.0, float(features.times[-1]) + 1.0, label, 0.9)
    return detect_inversions([span], features)[0]


def test_a_steady_chord_tone_bass_is_reported():
    segment = detect_one("C:maj", [4] * 40)
    assert segment.bass == "E"
    assert segment.mirex_label == "C:maj/3"


def test_a_bass_on_the_root_is_root_position_not_a_slash():
    segment = detect_one("C:maj", [0] * 40)
    assert segment.bass is None
    assert segment.mirex_label == "C:maj"


def test_a_non_chord_tone_bass_is_not_reported():
    """D is no part of C:maj, so a low D is a passing bass, not an inversion."""
    assert detect_one("C:maj", [2] * 40).bass is None


def test_a_moving_bass_is_not_reported():
    """An even split between two notes clears neither the agreement share."""
    assert detect_one("C:maj", [4] * 20 + [7] * 20).bass is None


def test_a_murky_low_end_is_not_reported():
    """Too few frames carry a dominant bass note for any of them to speak."""
    assert detect_one("C:maj", [4] * 15 + [None] * 25).bass is None


def test_the_no_chord_state_never_gets_a_bass():
    assert detect_one("N", [4] * 40).bass is None


def test_segments_without_a_bass_are_returned_unchanged():
    features = features_with_bass([None] * 20)
    original = ChordSegment(0.0, 5.0, "C:maj", 0.9)
    assert detect_inversions([original], features)[0] is original


# --- ordering ---------------------------------------------------------------


def test_coalescing_drops_a_bass_which_is_why_detection_runs_last():
    """Pins the reason the pipeline calls detect_inversions after the engine.

    Every merge and snap helper rebuilds segments field by field from their
    neighbours, and none of them knows about the bass.  Detecting inversions
    before they run would quietly lose the result.
    """
    merged = _coalesce(
        [
            ChordSegment(0.0, 1.0, "C:maj", 0.9, bass="E"),
            ChordSegment(1.0, 2.0, "C:maj", 0.9, bass="E"),
        ]
    )
    assert len(merged) == 1
    assert merged[0].bass is None


# --- export -----------------------------------------------------------------


def test_slash_labels_round_trip_through_mir_eval():
    """A `.lab` we write has to be one mir_eval can read back."""
    mir_eval = pytest.importorskip("mir_eval", reason="requires the [eval] extra")

    from chord_key_analyzer.models import AnalysisResult, KeyEstimate, ProgressionSummary
    from chord_key_analyzer.output import to_lab

    result = AnalysisResult(
        file="inverted.wav",
        duration=6.0,
        key=KeyEstimate("C", "major", 0.9),
        chords=[
            ChordSegment(0.0, 2.0, "C:maj", 0.9, bass="E"),
            ChordSegment(2.0, 4.0, "F:maj7", 0.8, bass="E"),
            ChordSegment(4.0, 6.0, "A:min", 0.8),
        ],
        progression=ProgressionSummary([], [], None),
    )

    lines = [line.split("\t")[2] for line in to_lab(result).strip().splitlines()]
    assert lines == ["C:maj/3", "F:maj7/7", "A:min"]
    for label in lines:
        mir_eval.chord.encode(label)
