"""Bass-driven re-spelling of chords whose notes have two equally good names.

`C:maj6` and `A:min7` are the same four pitch classes.  The decoder holds one
spelling as a state and cannot possibly choose between them from chroma, because
there is nothing to choose between — the sets are identical.  The bass is the
tie-breaker.
"""

from __future__ import annotations

import numpy as np
import pytest

import fixtures as fx
from chord_key_analyzer.chords import TemplateHMMEngine, detect_inversions, respell_with_bass
from chord_key_analyzer.features import extract_features
from chord_key_analyzer.models import (
    CHORD_QUALITIES,
    RESPELLED_QUALITIES,
    ChordSegment,
    KeyEstimate,
)
from chord_key_analyzer.progression import roman_numeral


def respell_one(label: str, bass: str | None) -> ChordSegment:
    return respell_with_bass([ChordSegment(0.0, 2.0, label, 0.9, bass=bass)])[0]


# --- the rule ----------------------------------------------------------------


def test_a_min7_over_its_third_is_a_major_sixth():
    """A:min7 over C is C-E-G-A, which is C6 with the root in the bass."""
    segment = respell_one("A:min7", "C")
    assert segment.label == "C:maj6"
    assert segment.bass is None, "the re-spelling is root position by construction"
    assert segment.mirex_label == "C:maj6"


@pytest.mark.parametrize("root", ["C", "D#", "F#", "A", "B"])
def test_the_respellings_hold_at_every_root(root):
    from chord_key_analyzer.models import PITCH_CLASSES, chord_label

    pitch_class = PITCH_CLASSES.index(root)
    third = PITCH_CLASSES[(pitch_class + 3) % 12]

    assert respell_one(chord_label(pitch_class, "min7"), third).label == chord_label(
        pitch_class + 3, "maj6"
    )


# --- everything else is left alone ------------------------------------------


def test_a_root_position_min7_is_not_respelled():
    assert respell_one("A:min7", None).label == "A:min7"


def test_a_min7_over_its_fifth_stays_a_slash_chord():
    """Only the b3 bass makes the sixth reading right; a fifth in the bass is
    still an inversion of the seventh."""
    segment = respell_one("A:min7", "E")
    assert segment.label == "A:min7"
    assert segment.bass == "E"
    assert segment.mirex_label == "A:min7/5"


def test_a_maj_triad_over_its_third_stays_a_slash_chord():
    segment = respell_one("C:maj", "E")
    assert (segment.label, segment.mirex_label) == ("C:maj", "C:maj/3")


def test_the_no_chord_state_passes_through():
    assert respell_one("N", None).label == "N"


def test_segments_needing_no_respelling_are_returned_unchanged():
    original = ChordSegment(0.0, 2.0, "C:maj", 0.9)
    assert respell_with_bass([original])[0] is original


# --- downstream -------------------------------------------------------------


def test_respelled_qualities_get_roman_numerals():
    """A sixth chord is major, so the numeral stays uppercase."""
    key = KeyEstimate("C", "major", 0.9)
    assert roman_numeral("C:maj6", key.tonic, key.mode) == "I6"
    assert roman_numeral("F:maj6", key.tonic, key.mode) == "IV6"
    assert roman_numeral("A:maj6", key.tonic, key.mode) == "VI6"


def test_respelled_qualities_are_never_decoder_states():
    """They have no template of their own — that is the whole point of them."""
    from chord_key_analyzer.chords import build_chord_templates

    _, labels = build_chord_templates()
    for quality in RESPELLED_QUALITIES:
        assert quality not in CHORD_QUALITIES
        assert not any(label.endswith(f":{quality}") for label in labels)


def test_respelled_pitch_class_sets_really_do_collide():
    """The premise of the whole module, asserted rather than assumed."""
    def notes(root: int, intervals: tuple[int, ...]) -> frozenset[int]:
        return frozenset((root + i) % 12 for i in intervals)

    for root in range(12):
        assert notes(root, RESPELLED_QUALITIES["maj6"]) == notes(
            root + 9, CHORD_QUALITIES["min7"]
        )


# --- end to end -------------------------------------------------------------


@pytest.mark.parametrize(("root", "expected"), [("A", "C:maj6"), ("C", "D#:maj6"), ("E", "G:maj6")])
def test_a_rendered_min7_over_its_third_becomes_a_major_sixth(root, expected):
    """The whole path: audio in, re-spelled label out."""
    audio = np.concatenate(
        [fx.render_chord(root, "min7", 1.5, bass_interval=3, bass_amplitude=0.5)] * 3
    ).astype(np.float32)
    features = extract_features(audio, fx.SR)
    segments = respell_with_bass(
        detect_inversions(TemplateHMMEngine().analyze(features), features)
    )

    longest = max((s for s in segments if not s.is_no_chord), key=lambda s: s.duration)
    assert longest.label == expected
    assert longest.bass is None


def test_a_dominant_bass_re_roots_the_decode_before_respelling_can_apply():
    """The honest limit of the re-spelling, pinned rather than left implicit.

    Raise the bass above the voicing and the chroma stops reading the chord as a
    minor seventh at all — it reads the major triad on the bass note, which is a
    perfectly defensible name for the same notes and needs no re-spelling.  So
    the rule fires on the middle ground: a bass clear enough to identify, but not
    so dominant that the decoder has already re-rooted for its own reasons.
    """
    audio = np.concatenate(
        [fx.render_chord("A", "min7", 1.5, bass_interval=3, bass_amplitude=1.0)] * 3
    ).astype(np.float32)
    features = extract_features(audio, fx.SR)
    segments = respell_with_bass(
        detect_inversions(TemplateHMMEngine().analyze(features), features)
    )

    longest = max((s for s in segments if not s.is_no_chord), key=lambda s: s.duration)
    assert longest.label == "C:maj"
    assert longest.bass is None
