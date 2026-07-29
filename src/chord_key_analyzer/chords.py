"""Chord recognition.

:class:`ChordEngine` is the pluggable seam described in ``docs/PLAN.md``: any
object with an ``analyze(features) -> list[ChordSegment]`` method can back the
``--engine`` flag.  v1 ships one implementation, :class:`TemplateHMMEngine`,
which scores harmonically-weighted chord templates against chroma frames and
smooths the result with a Viterbi decode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from .features import Features
from .models import (
    CHORD_QUALITIES,
    NO_CHORD,
    PITCH_CLASSES,
    TRIAD_QUALITIES,
    ChordSegment,
    chord_label,
    parse_chord_label,
)

#: Harmonic series weights applied when building templates.  Partial *k* of a
#: chord tone lands on the pitch class ``round(12*log2(k))`` semitones above it
#: and contributes ``HARMONIC_DECAY ** (k-1)`` energy.
N_HARMONICS = 6
HARMONIC_DECAY = 0.6

#: Sharpness of the emission distribution.  Higher makes the decoder trust a
#: single frame's best-matching template more; lower leans on the transition
#: prior.  Cosine scores between chroma and templates bunch up in the 0.5-0.95
#: range, so the multiplier has to be sizeable before neighbouring chords
#: separate at all; 8-20 is the useful band.
EMISSION_SHARPNESS = 16.0

#: Probability of staying on the same chord from one frame to the next.  At the
#: default ~93 ms hop this corresponds to a mean chord length of roughly 1.9 s,
#: which suppresses frame-level flicker without smearing fast changes.
SELF_TRANSITION_PROB = 0.95

#: Segments shorter than this are absorbed into their neighbours — below about
#: a beat, a "chord" is far more likely to be a decoding artefact.
MIN_SEGMENT_DURATION = 0.30


class ChordEngine(Protocol):
    """Any chord recogniser the CLI can drive."""

    name: str

    def analyze(self, features: Features) -> list[ChordSegment]:
        """Return non-overlapping chord segments covering the track."""
        ...


#: Base weight per chord tone, keyed by position in the chord.  Chord tones are
#: not equally present in a real recording: the root is reinforced by the bass
#: and by every other tone's partials, while a seventh is typically the quietest
#: voice.  Weighting them equally makes a four-note template spread its norm
#: over more bins than a triad's, so root-heavy audio matches ``min`` better
#: than ``min7`` even when the seventh is plainly sounding — sevenths collapse
#: to triads.  These weights restore the balance.
TONE_WEIGHTS = (1.0, 0.75, 0.8, 0.5)  # root, third, fifth, seventh


def _harmonic_template(intervals: tuple[int, ...]) -> np.ndarray:
    """A 12-vector of expected pitch-class energy for one chord.

    Two effects are modelled: chord tones carry different weight depending on
    their role (see :data:`TONE_WEIGHTS`), and each sounding tone also energises
    its own harmonic series — a sounding C puts real energy on G and, higher up,
    on E.  Plain binary templates ignore both and separate qualities poorly.
    """
    template = np.zeros(12, dtype=float)
    for position, interval in enumerate(intervals):
        base = TONE_WEIGHTS[position] if position < len(TONE_WEIGHTS) else TONE_WEIGHTS[-1]
        for partial in range(1, N_HARMONICS + 1):
            pitch_class = int(round(interval + 12 * np.log2(partial))) % 12
            template[pitch_class] += base * HARMONIC_DECAY ** (partial - 1)
    return template


def build_chord_templates(
    qualities: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the (n_states, 12) template matrix and its parallel label list.

    The final row is the no-chord state, modelled as a flat pitch-class
    distribution: silence and unpitched noise match it better than any chord.
    """
    if qualities is None:
        qualities = tuple(CHORD_QUALITIES)

    rows: list[np.ndarray] = []
    labels: list[str] = []
    seen: set[bytes] = set()
    for root in range(12):
        for quality in qualities:
            intervals = CHORD_QUALITIES[quality]
            pitch_classes = frozenset((root + i) % 12 for i in intervals)
            fingerprint = bytes(sorted(pitch_classes))
            if fingerprint in seen:
                # Augmented triads repeat every four semitones, so C:aug, E:aug
                # and G#:aug are one and the same pitch-class set.  Chroma has no
                # bass information to tell them apart, and keeping all three only
                # splits probability mass between identical templates.  Keep the
                # lowest-root spelling as the canonical one.
                continue
            seen.add(fingerprint)
            rows.append(np.roll(_harmonic_template(intervals), root))
            labels.append(chord_label(root, quality))

    rows.append(np.full(12, 1.0 / np.sqrt(12)))
    labels.append(NO_CHORD)

    templates = np.stack(rows, axis=0)
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9)
    return templates, labels


def viterbi_decode(
    log_emissions: np.ndarray, self_prob: float = SELF_TRANSITION_PROB
) -> np.ndarray:
    """Decode the most likely state path under a uniform-switch transition model.

    With a uniform "switch to anything else" probability the usual O(T·S²)
    recursion collapses to O(T·S): the best predecessor for state *j* is either
    *j* itself (self-transition) or the globally best previous state.  Since
    ``log_self > log_switch``, taking the global max is safe even when it is
    *j* — the self term already dominates in that case.
    """
    n_frames, n_states = log_emissions.shape
    if n_frames == 0:
        return np.zeros(0, dtype=int)
    if n_states == 1:
        return np.zeros(n_frames, dtype=int)

    log_self = np.log(self_prob)
    log_switch = np.log((1.0 - self_prob) / (n_states - 1))

    delta = log_emissions[0].copy()
    backpointers = np.zeros((n_frames, n_states), dtype=np.int32)

    for frame in range(1, n_frames):
        best_prev_state = int(np.argmax(delta))
        best_prev_value = delta[best_prev_state]

        stay = delta + log_self
        switch = best_prev_value + log_switch

        take_stay = stay >= switch
        backpointers[frame] = np.where(take_stay, np.arange(n_states), best_prev_state)
        delta = np.where(take_stay, stay, switch) + log_emissions[frame]

    path = np.zeros(n_frames, dtype=int)
    path[-1] = int(np.argmax(delta))
    for frame in range(n_frames - 1, 0, -1):
        path[frame - 1] = backpointers[frame, path[frame]]
    return path


def _segments_from_path(
    path: np.ndarray,
    labels: list[str],
    times: np.ndarray,
    frame_duration: float,
    confidences: np.ndarray,
) -> list[ChordSegment]:
    """Merge runs of identical states into segments."""
    if path.size == 0:
        return []

    segments: list[ChordSegment] = []
    start_index = 0
    for index in range(1, path.size + 1):
        if index < path.size and path[index] == path[start_index]:
            continue
        end_index = index - 1
        start_time = float(times[start_index])
        end_time = float(times[end_index]) + frame_duration
        segments.append(
            ChordSegment(
                start=start_time,
                end=end_time,
                label=labels[int(path[start_index])],
                confidence=float(np.mean(confidences[start_index:index])),
            )
        )
        start_index = index
    return segments


def _merge_short_segments(
    segments: list[ChordSegment], min_duration: float
) -> list[ChordSegment]:
    """Drop sub-``min_duration`` segments, extending whichever neighbour is longer."""
    if len(segments) <= 1:
        return segments

    working = list(segments)
    changed = True
    while changed and len(working) > 1:
        changed = False
        for index, segment in enumerate(working):
            if segment.duration >= min_duration:
                continue

            previous = working[index - 1] if index > 0 else None
            following = working[index + 1] if index + 1 < len(working) else None
            if previous is None and following is None:
                continue

            take_previous = following is None or (
                previous is not None and previous.duration >= following.duration
            )
            if take_previous and previous is not None:
                working[index - 1] = ChordSegment(
                    start=previous.start,
                    end=segment.end,
                    label=previous.label,
                    confidence=previous.confidence,
                )
            else:
                assert following is not None
                working[index + 1] = ChordSegment(
                    start=segment.start,
                    end=following.end,
                    label=following.label,
                    confidence=following.confidence,
                )
            working.pop(index)
            changed = True
            break

    return _coalesce(working)


def _coalesce(segments: list[ChordSegment]) -> list[ChordSegment]:
    """Join adjacent segments that carry the same label."""
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        last = merged[-1]
        if segment.label == last.label:
            merged[-1] = ChordSegment(
                start=last.start,
                end=segment.end,
                label=last.label,
                confidence=(last.confidence * last.duration + segment.confidence * segment.duration)
                / max(last.duration + segment.duration, 1e-9),
            )
        else:
            merged.append(segment)
    return merged


#: Beat snapping is only applied when the decoded boundaries already agree with
#: the beat grid this closely (as a fraction of one beat).  A grid can be
#: perfectly steady and still be wrong — offset by half a beat, or locked to a
#: subdivision — and snapping to a wrong-but-steady grid moves every boundary
#: *away* from where the audio actually put it.  Requiring prior agreement means
#: snapping can only tidy up a grid that is already right.
BEAT_AGREEMENT_RATIO = 0.20


def snap_to_beats(
    segments: list[ChordSegment], beat_times: np.ndarray, tolerance: float
) -> list[ChordSegment]:
    """Nudge segment boundaries onto nearby beats.

    Harmony changes on beats far more often than between them, so when the beat
    grid is trustworthy this removes the residual boundary jitter.  Only
    boundaries already within ``tolerance`` of a beat move, so a wrong grid
    cannot drag a boundary far from where the audio put it.
    """
    if beat_times.size < 2 or len(segments) < 2:
        return segments

    boundaries = [segments[0].start] + [s.end for s in segments]
    snapped = list(boundaries)
    for index in range(1, len(boundaries) - 1):
        nearest = beat_times[int(np.argmin(np.abs(beat_times - boundaries[index])))]
        if abs(nearest - boundaries[index]) <= tolerance:
            snapped[index] = float(nearest)

    # Keep boundaries strictly increasing after snapping.
    for index in range(1, len(snapped)):
        if snapped[index] <= snapped[index - 1]:
            snapped[index] = boundaries[index]

    result = [
        ChordSegment(
            start=snapped[index],
            end=snapped[index + 1],
            label=segment.label,
            confidence=segment.confidence,
        )
        for index, segment in enumerate(segments)
        if snapped[index + 1] > snapped[index]
    ]
    return _coalesce(result)


@dataclass
class TemplateHMMEngine:
    """Chroma-template chord recogniser with Viterbi smoothing.

    Frame-level rather than beat-synchronous by design: the decode should not
    inherit beat-tracker failures (exactly the rubato case where beats are least
    reliable), and frame resolution keeps boundaries well inside the +/-0.25 s
    tolerance the tests hold us to.  Beats are still used, but only as an
    optional post-hoc snap when the grid looks steady.
    """

    triads_only: bool = False
    self_transition: float = SELF_TRANSITION_PROB
    sharpness: float = EMISSION_SHARPNESS
    min_segment_duration: float = MIN_SEGMENT_DURATION
    beat_snap: bool = True
    name: str = "template"

    def _qualities(self) -> tuple[str, ...]:
        return TRIAD_QUALITIES if self.triads_only else tuple(CHORD_QUALITIES)

    def analyze(self, features: Features) -> list[ChordSegment]:
        templates, labels = build_chord_templates(self._qualities())
        chroma = features.chroma  # (12, n_frames), columns already L2-normalised
        if chroma.shape[1] == 0:
            return []

        # Cosine similarity between each frame and each template, in [-1, 1] but
        # non-negative in practice since both operands are non-negative.
        similarity = templates @ chroma  # (n_states, n_frames)
        log_emissions = (self.sharpness * similarity).T  # (n_frames, n_states)

        # Silence is not a chord: force those frames onto the no-chord state.
        no_chord_index = labels.index(NO_CHORD)
        if features.silent.any():
            log_emissions[features.silent, :] = -1e3
            log_emissions[features.silent, no_chord_index] = 0.0

        path = viterbi_decode(log_emissions, self_prob=self.self_transition)

        # Per-frame confidence: softmax probability of the decoded state.
        shifted = log_emissions - log_emissions.max(axis=1, keepdims=True)
        posterior = np.exp(shifted)
        posterior /= posterior.sum(axis=1, keepdims=True)
        frame_confidence = posterior[np.arange(path.size), path]

        segments = _segments_from_path(
            path, labels, features.times, features.frame_duration, frame_confidence
        )
        segments = _coalesce(segments)
        segments = _merge_short_segments(segments, self.min_segment_duration)

        if self.beat_snap and features.beats_reliable and features.beat_times.size > 2:
            segments = self._maybe_snap(segments, features.beat_times)

        return segments

    @staticmethod
    def _maybe_snap(segments: list[ChordSegment], beat_times: np.ndarray) -> list[ChordSegment]:
        """Snap to the beat grid, but only if the boundaries already agree with it."""
        if len(segments) < 3:
            return segments

        beat_period = float(np.median(np.diff(beat_times)))
        if beat_period <= 0:
            return segments

        interior = np.array([s.end for s in segments[:-1]], dtype=float)
        deviations = np.abs(interior[:, None] - beat_times[None, :]).min(axis=1)
        if float(np.median(deviations)) > beat_period * BEAT_AGREEMENT_RATIO:
            return segments

        return snap_to_beats(segments, beat_times, tolerance=beat_period / 2.0)


#: A frame's bass counts as voiced when its strongest pitch class holds at least
#: this share of the frame's total bass energy.  A frame with nothing in
#: particular happening down there spreads evenly and gives every class about
#: 1/12 (0.083), so this asks for a genuinely dominant note rather than whichever
#: bin won a coin toss in a murky low end.
BASS_SALIENCE = 0.30

#: And the voiced frames of a segment must agree on that pitch class this often.
#: Below it the bass is moving — a walking line, a fill, a passing tone — and no
#: single note describes the segment.
BASS_AGREEMENT = 0.6


def detect_inversions(
    segments: list[ChordSegment], features: Features
) -> list[ChordSegment]:
    """Attach the sounding bass note to each segment, where there plainly is one.

    Run this **after** every merge and snap helper.  Those rebuild
    ``ChordSegment``s field by field from their neighbours and would silently
    drop a bass assigned earlier; the pipeline therefore calls this once, on the
    engine's finished output.

    Three conditions have to hold before a bass is reported, and each exists to
    make the failure mode "no slash" rather than "a wrong slash":

    * over half the segment's frames must have a clearly voiced bass
      (:data:`BASS_SALIENCE`),
    * those frames must agree on one pitch class (:data:`BASS_AGREEMENT`),
    * and that pitch class must be a chord tone other than the root.

    The last one is the strictest.  A low note outside the chord is a passing
    bass, a pedal, or simply a detector error, and none of those is worth
    printing a confident ``/b6`` over.
    """
    if not segments or features.bass_chroma.size == 0:
        return segments

    bass_chroma = features.bass_chroma
    energy = bass_chroma.sum(axis=0)
    salience = bass_chroma.max(axis=0) / np.maximum(energy, 1e-9)
    voiced = (salience >= BASS_SALIENCE) & (~features.silent)
    strongest = bass_chroma.argmax(axis=0)

    result: list[ChordSegment] = []
    for segment in segments:
        bass = _segment_bass(segment, features.times, voiced, strongest)
        result.append(segment if bass is None else replace(segment, bass=bass))
    return result


def _segment_bass(
    segment: ChordSegment,
    times: np.ndarray,
    voiced: np.ndarray,
    strongest: np.ndarray,
) -> str | None:
    """The bass pitch-class name for one segment, or ``None`` — see the caller."""
    parsed = parse_chord_label(segment.label)
    if parsed is None:
        return None
    root, quality = parsed

    in_segment = (times >= segment.start) & (times < segment.end)
    total = int(in_segment.sum())
    usable = in_segment & voiced
    n_usable = int(usable.sum())
    if not total or n_usable * 2 < total:
        return None

    # Every frame spans one hop, so counting frames *is* weighting by duration.
    counts = np.bincount(strongest[usable], minlength=12)
    candidate = int(counts.argmax())
    if counts[candidate] / float(n_usable) < BASS_AGREEMENT:
        return None

    interval = (candidate - root) % 12
    if not interval or interval not in CHORD_QUALITIES.get(quality, ()):
        return None
    return PITCH_CLASSES[candidate]


#: Bass-driven re-spellings, keyed by ``(detected quality, bass degree)`` and
#: giving ``(new quality, semitones to move the root)``.  The rule lands on a
#: root-position chord whose root is the sounding bass: `A:min7` over C is
#: exactly `C:maj6`.
_RESPELLINGS: dict[tuple[str, str], tuple[str, int]] = {
    ("min7", "b3"): ("maj6", 3),
}


def respell_with_bass(segments: list[ChordSegment]) -> list[ChordSegment]:
    """Rename chords whose bass reveals a better spelling of the same notes.

    Some pitch-class sets have two equally good names and no chroma-only decoder
    can choose between them: `C:maj6` and `A:min7` are the same four notes.  Only
    one spelling of the pair is a decoder state (see
    :data:`~chord_key_analyzer.models.RESPELLED_QUALITIES`); when the detected
    bass says the other one is what is actually sounding, this rewrites it.

    The result is root position by construction — the bass has *become* the root
    — so the bass field is cleared rather than left to print a slash.  Every
    other bass result is passed through exactly as
    :func:`detect_inversions` reported it, which is where the bass comes from and
    therefore what this has to run after.
    """
    result: list[ChordSegment] = []
    for segment in segments:
        parsed = parse_chord_label(segment.label)
        rule = _RESPELLINGS.get((parsed[1], segment.bass_degree)) if parsed else None
        if parsed is None or rule is None:
            result.append(segment)
            continue

        root, _ = parsed
        quality, shift = rule
        result.append(replace(segment, label=chord_label(root + shift, quality), bass=None))
    return result


def get_engine(name: str, triads_only: bool = False, beat_snap: bool = True) -> ChordEngine:
    """Resolve an engine name to an instance.

    ``deep`` is reserved for the future neural backend; it raises a pointed
    error rather than silently falling back, so a user who asks for it knows
    they did not get it.
    """
    normalised = name.lower()
    if normalised == "template":
        return TemplateHMMEngine(triads_only=triads_only, beat_snap=beat_snap)
    if normalised in {"crema", "deep"}:
        raise ValueError(
            f"Engine '{name}' is not implemented in this version. "
            "Only 'template' is available; see docs/PLAN.md for the roadmap."
        )
    raise ValueError(f"Unknown engine '{name}'. Available engines: template")
