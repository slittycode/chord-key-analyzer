"""Key detection tests."""

from __future__ import annotations

import numpy as np
import pytest

import fixtures as fx
from chord_key_analyzer.chords import TemplateHMMEngine
from chord_key_analyzer.features import extract_features
from chord_key_analyzer.key import (
    chord_evidence_scores,
    detect_key,
    detect_modulations,
    estimate_key_from_chroma,
    score_chroma_vector,
)
from chord_key_analyzer.models import PITCH_CLASSES

MAJOR_DEGREES = [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")]  # I V vi IV
MINOR_DEGREES = [(0, "min"), (8, "maj"), (3, "maj"), (10, "maj")]  # i VI III VII


def build_progression(tonic: str, degrees: list[tuple[int, str]]) -> list[tuple[str, str]]:
    root = fx.note_name_to_pc(tonic)
    return [(PITCH_CLASSES[(root + step) % 12], quality) for step, quality in degrees]


def analyse_key(progression, chord_duration=1.5, repeats=3):
    audio = fx.render_progression(progression, chord_duration=chord_duration, repeats=repeats)
    features = extract_features(audio, fx.SR)
    chords = TemplateHMMEngine().analyze(features)
    return detect_key(features, scan_modulations=False, chords=chords)


@pytest.mark.parametrize("tonic", PITCH_CLASSES)
def test_detects_every_major_key(tonic):
    key = analyse_key(build_progression(tonic, MAJOR_DEGREES))
    assert (key.tonic, key.mode) == (tonic, "major")
    assert key.confidence > 0.4


@pytest.mark.parametrize("tonic", PITCH_CLASSES)
def test_detects_every_minor_key(tonic):
    """The relative-major trap: i–VI–III–VII shares every pitch class with the
    relative major, so only the chord evidence can resolve it."""
    key = analyse_key(build_progression(tonic, MINOR_DEGREES))
    assert (key.tonic, key.mode) == (tonic, "minor")


def test_minor_key_is_not_reported_as_its_relative_major():
    key = analyse_key(build_progression("A", MINOR_DEGREES))
    assert key.name == "A minor"
    assert key.name != "C major"


def test_alternatives_are_ranked_and_exclude_the_winner():
    key = analyse_key(build_progression("C", MAJOR_DEGREES))
    assert len(key.alternatives) == 3
    assert all(alt.name != key.name for alt in key.alternatives)
    scores = [alt.score for alt in key.alternatives]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    ("tonic", "mode"),
    [("C", "major"), ("G", "major"), ("F#", "major"), ("A", "minor"), ("D", "minor")],
)
def test_melodic_only_input_finds_the_right_tonic(tonic, mode):
    """A bare scale has no harmony to establish mode, so we only hold the tonic
    to account — and expect the confidence to admit the uncertainty."""
    audio = fx.render_scale(tonic, mode, note_duration=0.45)
    features = extract_features(audio, fx.SR)
    chords = TemplateHMMEngine().analyze(features)
    key = detect_key(features, scan_modulations=False, chords=chords)
    assert key.tonic == tonic


def test_ambiguous_material_reports_low_confidence():
    """Whole-tone planing has no tonic; the estimate must not claim otherwise."""
    audio = fx.render_progression(
        [("C", "aug"), ("D", "aug"), ("C", "aug"), ("D", "aug")], chord_duration=1.5, repeats=3
    )
    features = extract_features(audio, fx.SR)
    key = detect_key(features, scan_modulations=False, chords=TemplateHMMEngine().analyze(features))
    assert key.confidence < 0.5


def test_confidence_is_higher_for_clear_than_ambiguous_material():
    clear = analyse_key(build_progression("C", MAJOR_DEGREES))
    audio = fx.render_progression(
        [("C", "aug"), ("D", "aug")], chord_duration=1.5, repeats=4
    )
    features = extract_features(audio, fx.SR)
    murky = detect_key(features, scan_modulations=False)
    assert clear.confidence > murky.confidence


def test_profile_scores_peak_on_the_matching_key():
    """A pure C-major scale profile should correlate best with C major."""
    chroma = np.zeros(12)
    for pitch_class in (0, 2, 4, 5, 7, 9, 11):
        chroma[pitch_class] = 1.0
    chroma[0] += 0.8  # tonic emphasis
    best, _, _ = estimate_key_from_chroma(chroma)
    assert best.name == "C major"


def test_score_vector_covers_all_24_keys():
    scores = score_chroma_vector(np.ones(12))
    assert scores.shape == (24,)
    assert np.all(np.isfinite(scores))


def test_chord_evidence_prefers_the_tonic_of_the_progression():
    from chord_key_analyzer.key import _TEMPLATE_NAMES
    from chord_key_analyzer.models import ChordSegment

    chords = [
        ChordSegment(0, 2, "A:min"),
        ChordSegment(2, 4, "F:maj"),
        ChordSegment(4, 6, "C:maj"),
        ChordSegment(6, 8, "G:maj"),
        ChordSegment(8, 10, "A:min"),
    ]
    scores = chord_evidence_scores(chords)
    assert _TEMPLATE_NAMES[int(np.argmax(scores))] == ("A", "minor")


def test_chord_evidence_is_empty_without_chords():
    assert np.all(chord_evidence_scores([]) == 0)


#: The hand-written diatonic tables the derivation replaced, frozen verbatim.
#: Reproducing these exactly is the whole licence for deriving them instead.
LEGACY_MAJOR_DEGREES = [
    (0, ("maj", "maj7")),
    (2, ("min", "min7")),
    (4, ("min", "min7")),
    (5, ("maj", "maj7")),
    (7, ("maj", "7")),
    (9, ("min", "min7")),
    (11, ("dim",)),
]
LEGACY_MINOR_DEGREES = [
    (0, ("min", "min7")),
    (2, ("dim",)),
    (3, ("maj", "maj7")),
    (5, ("min", "min7")),
    (7, ("min", "min7", "maj", "7")),
    (8, ("maj", "maj7")),
    (10, ("maj", "7")),
    (11, ("dim",)),
]
LEGACY_QUALITIES = frozenset({"maj", "min", "dim", "aug", "maj7", "min7", "7"})


@pytest.mark.parametrize(
    ("mode", "legacy"), [("major", LEGACY_MAJOR_DEGREES), ("minor", LEGACY_MINOR_DEGREES)]
)
@pytest.mark.parametrize("tonic", [0, 5, 7, 11])
def test_derived_diatonic_set_reproduces_the_hand_written_table(mode, legacy, tonic):
    """Restricted to the original vocabulary, the derivation must agree exactly.

    The minor case is the one that matters: merging natural and harmonic minor
    into a single eight-note collection derives three chords neither scale
    contains — iv°, bVI° and bvi — because it lets a natural-minor sixth degree
    sit under a harmonic-minor leading tone.  Stacking thirds on each scale
    separately does not.
    """
    from chord_key_analyzer.key import _diatonic_chords

    expected = {((tonic + degree) % 12, quality) for degree, qualities in legacy
                for quality in qualities}
    derived = {
        (root, quality)
        for root, quality in _diatonic_chords(tonic, mode)
        if quality in LEGACY_QUALITIES
    }
    assert derived == expected


def test_the_derivation_extends_to_the_new_qualities():
    """C6, F6 and G6 are all plainly in C major, and now score as in-key.

    The added sixth is not a tertian stack, so it comes from the second half of
    the derivation: every note of it belongs to the scale.
    """
    from chord_key_analyzer.key import _diatonic_chords

    in_c_major = _diatonic_chords(0, "major")
    assert {(0, "maj6"), (5, "maj6"), (7, "maj6")} <= in_c_major
    # D6 would need an F#, so it is not in the key.
    assert (2, "maj6") not in in_c_major


def test_augmented_triads_are_never_diatonic():
    """III+ is a tertian triad of the harmonic minor, but an augmented chord is
    a chromatic colour wherever it sits — counting it in-key would let whole-tone
    planing score as tonal."""
    from chord_key_analyzer.key import _diatonic_chords

    for mode in ("major", "minor"):
        assert not any(quality == "aug" for _, quality in _diatonic_chords(0, mode))


def test_modulation_is_detected_and_localised():
    first = fx.render_progression(build_progression("C", MAJOR_DEGREES), 2.0, 4)
    second = fx.render_progression(build_progression("E", MAJOR_DEGREES), 2.0, 4)
    audio = np.concatenate([first, second]).astype(np.float32)
    boundary = len(first) / fx.SR

    features = extract_features(audio, fx.SR)
    key = detect_key(features, scan_modulations=True, chords=TemplateHMMEngine().analyze(features))

    assert key.name == "C major"
    assert key.modulations, "expected the move to E major to be reported"
    e_major = [m for m in key.modulations if m.name == "E major"]
    assert e_major, f"expected E major, got {[m.name for m in key.modulations]}"
    # Window geometry (20 s window, 5 s hop) bounds how precisely the change
    # can be placed, so allow one window of slack around the true boundary.
    assert abs(e_major[0].start - boundary) <= 20.0
    assert e_major[0].end > boundary


def test_modulation_scan_damps_evidence_by_window_support(monkeypatch):
    """Each window's chord evidence must be scaled by that window's own variety.

    Deterministic counterpart to the behavioural test below: it asserts the
    damping is consulted per window rather than relying on a score margin.
    """
    from chord_key_analyzer import key as key_module

    audio = fx.render_progression(build_progression("C", MAJOR_DEGREES), 2.0, 6)
    features = extract_features(audio, fx.SR)
    chords = TemplateHMMEngine().analyze(features)

    calls: list[int] = []
    original = key_module._evidence_support

    def recording_support(segments):
        calls.append(len(segments))
        return original(segments)

    monkeypatch.setattr(key_module, "_evidence_support", recording_support)
    key_module.detect_modulations(features, ("C", "major"), chords=chords)

    assert calls, "the modulation scan never consulted the evidence-support damping"
    assert all(count > 0 for count in calls)


def test_thin_evidence_pulls_the_scan_less_than_undamped():
    """Damping must measurably weaken what one sustained chord can do.

    Asserted on the mixed score rather than on the winning key: with this
    fixture a single chord damps to support 1/3, and 1/3 of the evidence margin
    (1.51) still exceeds the chroma margin between C major and its neighbour
    (0.94), so the window's winner does not flip.  Damping narrows that gap by
    two thirds, which is the effect the fix is actually claiming.
    """
    from chord_key_analyzer import key as key_module
    from chord_key_analyzer.models import ChordSegment

    sustained = [ChordSegment(0.0, 45.0, "G:maj")]

    support = key_module._evidence_support(sustained)
    assert support == pytest.approx(1 / 3), "one distinct chord is one third of full support"

    evidence = key_module._standardise(
        key_module.chord_evidence_scores(sustained, use_edges=False)
    )
    c_index = key_module._TEMPLATE_NAMES.index(("C", "major"))
    g_index = key_module._TEMPLATE_NAMES.index(("G", "major"))
    margin = evidence[g_index] - evidence[c_index]

    undamped = key_module.KEY_CHORD_WEIGHT * margin
    damped = key_module.KEY_CHORD_WEIGHT * support * margin
    assert damped < undamped / 2


def test_estimate_key_consults_the_evidence_damping(monkeypatch):
    """The damping lives inside the key seam now, so every caller inherits it.

    Both the global key and the modulation scan used to apply it themselves,
    which is exactly the arrangement that let the two drift apart.
    """
    from chord_key_analyzer import key as key_module
    from chord_key_analyzer.models import ChordSegment

    calls: list[int] = []
    original = key_module._evidence_support

    def recording_support(segments):
        calls.append(len(segments))
        return original(segments)

    monkeypatch.setattr(key_module, "_evidence_support", recording_support)
    key_module.estimate_key_from_chroma(
        np.ones(12), chords=[ChordSegment(0, 2, "C:maj"), ChordSegment(2, 4, "G:maj")]
    )

    assert calls == [2], "estimate_key_from_chroma did not damp the chord evidence"


def test_thin_evidence_pulls_the_global_key_less_than_undamped():
    """Counterpart to the modulation-path test above, for the global key seam.

    Asserted on the score gap rather than on the winning key, and for the same
    reason: one sustained G against a pure C-major profile still carries the
    estimate even damped.  What the damping controls is by how much — one
    distinct chord is a third of full support, so a third of the pull.
    """
    from chord_key_analyzer import key as key_module
    from chord_key_analyzer.models import ChordSegment

    chroma = np.zeros(12)
    for pitch_class in (0, 2, 4, 5, 7, 9, 11):
        chroma[pitch_class] = 1.0
    chroma[0] += 0.8  # tonic emphasis, as in the profile test above
    sustained = [ChordSegment(0.0, 45.0, "G:maj")]

    best, _, alternatives = estimate_key_from_chroma(chroma, chords=sustained)
    scores = {best.name: best.score, **{alt.name: alt.score for alt in alternatives}}
    gap = scores["G major"] - scores["C major"]

    g_index = key_module._TEMPLATE_NAMES.index(("G", "major"))
    c_index = key_module._TEMPLATE_NAMES.index(("C", "major"))
    profile = key_module._standardise(key_module.score_chroma_vector(chroma))
    evidence = key_module._standardise(key_module.chord_evidence_scores(sustained))
    profile_margin = profile[g_index] - profile[c_index]
    evidence_margin = evidence[g_index] - evidence[c_index]

    damped = profile_margin + key_module.KEY_CHORD_WEIGHT * (1 / 3) * evidence_margin
    undamped = profile_margin + key_module.KEY_CHORD_WEIGHT * evidence_margin
    assert gap == pytest.approx(damped)
    assert gap < undamped / 2


def test_no_chord_evidence_leaves_the_scan_untouched():
    """All-N chords carry zero support, so they must not perturb the scan."""
    from chord_key_analyzer.models import ChordSegment

    audio = fx.render_progression(build_progression("C", MAJOR_DEGREES), 2.0, 6)
    features = extract_features(audio, fx.SR)
    duration = len(audio) / fx.SR
    silent = [ChordSegment(0.0, duration, "N")]

    with_n = detect_modulations(features, ("C", "major"), chords=silent)
    without = detect_modulations(features, ("C", "major"), chords=None)
    assert [(m.name, m.start) for m in with_n] == [(m.name, m.start) for m in without]


def test_single_key_track_reports_no_modulations():
    audio = fx.render_progression(build_progression("C", MAJOR_DEGREES), 2.0, 8)
    features = extract_features(audio, fx.SR)
    key = detect_key(features, scan_modulations=True, chords=TemplateHMMEngine().analyze(features))
    assert key.modulations == []


def test_short_track_skips_the_modulation_scan():
    audio = fx.render_progression(fx.POP_LOOP_C, chord_duration=0.5, repeats=1)
    features = extract_features(audio, fx.SR)
    key = detect_key(features, scan_modulations=True)
    assert key.modulations == []
