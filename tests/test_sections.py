"""Structural section detection.

The rendered tracks here are long — a section is a 30-second-plus object, and
there is no way to test one on a short clip.  Each fixture is built so its true
structure is known by construction, and asserted against with a tolerance that
matches what a 1 s block grid can actually resolve.
"""

from __future__ import annotations

import numpy as np
import pytest

import fixtures as fx
from chord_key_analyzer.chords import TemplateHMMEngine
from chord_key_analyzer.features import extract_features
from chord_key_analyzer.key import detect_key
from chord_key_analyzer.models import Section
from chord_key_analyzer.sections import (
    SECTION_MIN_TRACK,
    _boundaries,
    _checkerboard_kernel,
    _letter,
    detect_sections,
    novelty_curve,
)

#: How far a detected boundary may sit from the true one.  Blocks are 1 s and the
#: checkerboard kernel is deliberately broad, so a few seconds of slack is the
#: honest resolution of the method rather than a concession to flakiness.
BOUNDARY_TOLERANCE = 4.0

E_MAJOR_LOOP = [("E", "maj"), ("B", "maj"), ("C#", "min"), ("A", "maj")]


def analyse(audio):
    features = extract_features(audio, fx.SR)
    chords = TemplateHMMEngine().analyze(features)
    key = detect_key(features, scan_modulations=False, chords=chords)
    return detect_sections(features, chords, key), key


def base_letter(section: Section) -> str:
    return section.label.rstrip("'")


@pytest.fixture(scope="module")
def aba_sections():
    """32 s of I–V–vi–IV in C, 32 s of the same shape in E, then C again."""
    a = fx.render_progression(fx.POP_LOOP_C, chord_duration=2.0, repeats=4)
    b = fx.render_progression(E_MAJOR_LOOP, chord_duration=2.0, repeats=4)
    sections, _ = analyse(np.concatenate([a, b, a]).astype(np.float32))
    return sections


# --- structure --------------------------------------------------------------


def test_an_aba_track_yields_three_sections(aba_sections):
    assert len(aba_sections) == 3


def test_boundaries_land_where_the_music_changes(aba_sections):
    for section, expected in zip(aba_sections, (0.0, 32.0, 64.0), strict=True):
        assert abs(section.start - expected) <= BOUNDARY_TOLERANCE, section


def test_the_repeated_section_gets_the_repeated_letter(aba_sections):
    first, middle, last = aba_sections
    assert base_letter(first) == base_letter(last)
    assert base_letter(first) != base_letter(middle)


def test_sections_tile_the_whole_track(aba_sections):
    assert aba_sections[0].start == pytest.approx(0.0)
    for previous, following in zip(aba_sections, aba_sections[1:], strict=False):
        assert previous.end == pytest.approx(following.start)
    assert aba_sections[-1].end == pytest.approx(96.0, abs=1.0)


def test_each_section_gets_its_own_key_hint(aba_sections):
    assert aba_sections[1].tonic == "E"
    assert aba_sections[1].key_name == "E major"
    assert all(0.0 <= s.key_confidence <= 1.0 for s in aba_sections)


def test_section_numerals_stay_relative_to_the_global_key(aba_sections):
    """Two sections in different keys must stay comparable, so the numerals are
    written against the track's key rather than each section's own."""
    first, middle, _ = aba_sections
    assert first.progression.roman[:4] == ["I", "V", "vi", "IV"]
    # The E-major section is not I-V-vi-IV *in C*, and must not be relabelled so.
    assert middle.progression.roman[:4] != ["I", "V", "vi", "IV"]


def test_a_homogeneous_track_is_one_section():
    audio = fx.render_progression(fx.POP_LOOP_C, chord_duration=2.0, repeats=8)
    sections, _ = analyse(audio)
    assert [s.label for s in sections] == ["A"]


def test_a_silent_track_is_one_section():
    """Silence is long enough to be sectioned and has nothing to section on.

    It is the reachable input whose novelty curve is flat to the bit, which is
    the case the peak threshold cannot judge for itself — see
    :func:`test_a_flat_novelty_curve_holds_no_boundaries`.
    """
    sections, _ = analyse(fx.render_silence(40.0))
    assert [s.label for s in sections] == ["A"]


def test_a_short_track_gets_no_sections():
    audio = fx.render_progression(fx.POP_LOOP_C, chord_duration=2.0, repeats=3)
    sections, _ = analyse(audio)
    assert sections == []
    assert len(audio) / fx.SR < SECTION_MIN_TRACK


def test_the_existing_pop_fixture_is_below_the_guard(pop_result):
    """Everything already in the suite analyses a 24 s clip, so nothing there
    should have started reporting sections."""
    assert pop_result.duration < SECTION_MIN_TRACK
    assert pop_result.sections == []


# --- the pieces -------------------------------------------------------------


def test_the_checkerboard_kernel_is_signed_in_quadrants():
    kernel = _checkerboard_kernel(4)
    assert kernel.shape == (8, 8)
    assert kernel[:4, :4].sum() > 0 and kernel[4:, 4:].sum() > 0
    assert kernel[:4, 4:].sum() < 0 and kernel[4:, :4].sum() < 0
    assert kernel.sum() == pytest.approx(0.0, abs=1e-9)


def test_novelty_peaks_at_a_change_and_not_within_a_block():
    """Two unlike halves: the curve must spike at the join, not inside either."""
    left = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, 40))
    right = np.tile(np.array([[0.0], [1.0], [0.0]]), (1, 40))
    novelty = novelty_curve(np.hstack([left, right]))

    assert novelty.argmax() == pytest.approx(40, abs=2)
    assert novelty[10] < novelty.max() / 2
    assert novelty[70] < novelty.max() / 2


def test_novelty_is_flat_on_uniform_input():
    uniform = np.tile(np.array([[1.0], [0.5], [0.25]]), (1, 60))
    assert novelty_curve(uniform).max() == pytest.approx(0.0, abs=1e-9)


def test_a_flat_novelty_curve_holds_no_boundaries():
    """The seam between the two halves of the flat-curve case.

    ``novelty_curve`` reporting a flat curve is only half an answer; the other
    half is ``_boundaries`` declining to read peaks into it.  Left to itself,
    ``peak_pick`` on an all-zero array with ``delta=0`` calls every position a
    peak and returns one every ``wait`` blocks — maximum structure from the
    input with none.
    """
    assert _boundaries(np.zeros(60), min_blocks=8) == [0]
    assert _boundaries(np.full(60, 0.7), min_blocks=8) == [0]


def test_novelty_handles_degenerate_input():
    assert novelty_curve(np.zeros((13, 0))).size == 0
    assert novelty_curve(np.zeros((13, 1))).size == 1


@pytest.mark.parametrize(
    ("index", "expected"), [(0, "A"), (1, "B"), (25, "Z"), (26, "AA"), (27, "AB")]
)
def test_section_letters_do_not_run_out(index, expected):
    assert _letter(index) == expected
