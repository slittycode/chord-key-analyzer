"""Key detection from pitch-class profiles and chord evidence.

Two published key profiles are correlated against the observed pitch-class
distribution and their results averaged:

* Krumhansl-Schmuckler (probe-tone ratings, Krumhansl 1990)
* Temperley (corpus counts from the Kostka-Payne corpus, Temperley 2001)

Averaging the two is a cheap hedge — KS over-weights the tonic on sustained
material, Temperley is better behaved on tonal pop but weaker on modal writing.

Profiles alone are not enough.  A key and its relative (A minor / C major)
contain exactly the same pitch classes, so *no* pitch-class histogram can
separate them — in testing, profile correlation alone put every one of the 12
minor keys under its relative major.  What separates them is which chord acts as
home, so the decoded chord track is folded in as a second evidence term (see
:func:`chord_evidence_scores`).  Chord decoding needs no key, so the dependency
runs one way only.
"""

from __future__ import annotations

import numpy as np

from .chords import viterbi_decode
from .features import Features
from .models import (
    PITCH_CLASSES,
    ChordSegment,
    KeyCandidate,
    KeyEstimate,
    Modulation,
    parse_chord_label,
)

KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

PROFILE_SETS = (
    (KRUMHANSL_MAJOR, KRUMHANSL_MINOR),
    (TEMPERLEY_MAJOR, TEMPERLEY_MINOR),
)

MODES = ("major", "minor")

#: Softmax temperature used to turn the standardised scores into a 0-1
#: confidence.  Tuned so an unambiguous key lands around 0.85 and genuinely
#: ambiguous material (whole-tone, chromatic planing) falls below 0.4.
_CONFIDENCE_TEMPERATURE = 0.6

#: Sliding-window geometry for the modulation scan.
MODULATION_WINDOW = 20.0
MODULATION_HOP = 5.0
#: A local key must hold for this many consecutive windows to count as a real
#: modulation rather than a passing tonicisation.
MODULATION_MIN_WINDOWS = 3
#: Probability that the key is unchanged from one window to the next.  Keys
#: change far more rarely than chords do, so this is much stickier than the
#: chord decoder's self-transition.
MODULATION_SELF_PROB = 0.985


def _zscore(vector: np.ndarray) -> np.ndarray:
    centred = vector - vector.mean()
    norm = np.linalg.norm(centred)
    if norm < 1e-12:
        return np.zeros_like(centred)
    return centred / norm


def _build_template_matrix() -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Pre-compute z-scored, rotated profiles for all 24 keys.

    Returns the (24, 12) matrix and the parallel list of ``(tonic, mode)`` names.
    Rows are the mean of the z-scored KS and Temperley profiles, which makes a
    dot product against a z-scored chroma vector equal to the mean of the two
    Pearson correlations.
    """
    rows: list[np.ndarray] = []
    names: list[tuple[str, str]] = []
    for mode_index, mode in enumerate(MODES):
        for tonic in range(12):
            stacked = [
                _zscore(np.roll(profiles[mode_index], tonic)) for profiles in PROFILE_SETS
            ]
            rows.append(np.mean(stacked, axis=0))
            names.append((PITCH_CLASSES[tonic], mode))
    return np.stack(rows, axis=0), names


_TEMPLATES, _TEMPLATE_NAMES = _build_template_matrix()


def _standardise(scores: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance across the 24 candidate keys.

    Profile correlations and chord-evidence scores have unrelated natural
    ranges; expressing both in standard deviations over the candidate set is
    what lets :data:`KEY_CHORD_WEIGHT` mean "how much this evidence counts"
    instead of silently encoding a unit conversion.
    """
    spread = float(scores.std())
    if spread < 1e-12:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / spread


def score_chroma_vector(chroma_vector: np.ndarray) -> np.ndarray:
    """Correlation score for each of the 24 keys, ordered like ``_TEMPLATE_NAMES``."""
    return _TEMPLATES @ _zscore(np.asarray(chroma_vector, dtype=float))


def _confidence(scores: np.ndarray) -> float:
    """Softmax probability of the winning key — a bounded, monotone margin."""
    shifted = (scores - scores.max()) / _CONFIDENCE_TEMPERATURE
    weights = np.exp(shifted)
    return float(weights.max() / weights.sum())


def estimate_key_from_chroma(
    chroma_vector: np.ndarray,
    n_alternatives: int = 3,
    chord_scores: np.ndarray | None = None,
) -> tuple[KeyCandidate, float, list[KeyCandidate]]:
    """Best key, its confidence, and the runners-up for one pooled chroma vector.

    When ``chord_scores`` is supplied (from :func:`chord_evidence_scores`) it is
    blended into the profile correlations.  This is what separates a key from
    its relative major/minor: the two share an identical pitch-class content, so
    no amount of chroma pooling can tell them apart, but which chord functions as
    home is decisive and shows up plainly in the chord track.
    """
    scores = _standardise(score_chroma_vector(chroma_vector))
    if chord_scores is not None:
        scores = scores + KEY_CHORD_WEIGHT * _standardise(chord_scores)
    order = np.argsort(scores)[::-1]

    best_index = int(order[0])
    tonic, mode = _TEMPLATE_NAMES[best_index]
    best = KeyCandidate(tonic=tonic, mode=mode, score=float(scores[best_index]))

    alternatives = []
    for index in order[1 : n_alternatives + 1]:
        alt_tonic, alt_mode = _TEMPLATE_NAMES[int(index)]
        alternatives.append(
            KeyCandidate(tonic=alt_tonic, mode=alt_mode, score=float(scores[int(index)]))
        )

    return best, _confidence(scores), alternatives


def _diatonic_chords(tonic: int, mode: str) -> set[tuple[int, str]]:
    """The ``(root_pc, quality)`` pairs that belong to a key.

    The minor set covers both natural and harmonic minor, so a v and a V (and a
    subtonic bVII alongside a leading-tone vii°) all count as in-key — which is
    how minor-key music actually behaves.
    """
    if mode == "major":
        degrees = [
            (0, ("maj", "maj7")),
            (2, ("min", "min7")),
            (4, ("min", "min7")),
            (5, ("maj", "maj7")),
            (7, ("maj", "7")),
            (9, ("min", "min7")),
            (11, ("dim",)),
        ]
    else:
        degrees = [
            (0, ("min", "min7")),
            (2, ("dim",)),
            (3, ("maj", "maj7")),
            (5, ("min", "min7")),
            (7, ("min", "min7", "maj", "7")),
            (8, ("maj", "maj7")),
            (10, ("maj", "7")),
            (11, ("dim",)),
        ]
    return {
        ((tonic + degree) % 12, quality)
        for degree, qualities in degrees
        for quality in qualities
    }


def _tonic_qualities(mode: str) -> tuple[str, ...]:
    return ("maj", "maj7") if mode == "major" else ("min", "min7")


#: How much the chord evidence counts relative to the profile correlation, once
#: both have been standardised across the 24 candidate keys (see
#: :func:`_standardise`).  Standardising first is what makes this a genuine
#: relative weight rather than an artefact of the two scores' natural ranges.
KEY_CHORD_WEIGHT = 2.0

#: Extra credit for time spent on the tonic triad itself, and for the
#: progression starting or ending there.  Relative major/minor pairs share every
#: diatonic chord, so *which* chord acts as home is the only thing that
#: separates them — a pitch-class histogram alone genuinely cannot.
TONIC_TIME_WEIGHT = 0.8
TONIC_EDGE_WEIGHT = 0.35


def chord_evidence_scores(
    chords: list[ChordSegment], use_edges: bool = True
) -> np.ndarray:
    """Score all 24 keys by how well a detected chord sequence fits them.

    Ordered like :data:`_TEMPLATE_NAMES` so it can be added to the profile
    correlations directly.

    ``use_edges`` controls the "starts or ends on the tonic" bonus.  That signal
    is strong for a whole piece, where the first and last chords really are
    structural, but meaningless inside an arbitrary sliding window — there the
    window's edges fall wherever the hop puts them, and the bonus flips a key
    between itself and its relative from one window to the next.
    """
    scores = np.zeros(len(_TEMPLATE_NAMES), dtype=float)

    pitched = [c for c in chords if not c.is_no_chord and parse_chord_label(c.label)]
    total = sum(c.duration for c in pitched)
    if total <= 0:
        return scores

    parsed = [(parse_chord_label(c.label), c.duration) for c in pitched]
    first = parsed[0][0]
    last = parsed[-1][0]

    for index, (tonic_name, mode) in enumerate(_TEMPLATE_NAMES):
        tonic = PITCH_CLASSES.index(tonic_name)
        diatonic = _diatonic_chords(tonic, mode)
        tonic_set = {(tonic, quality) for quality in _tonic_qualities(mode)}

        diatonic_time = sum(d for chord, d in parsed if chord in diatonic)
        tonic_time = sum(d for chord, d in parsed if chord in tonic_set)

        edge = 0.0
        if use_edges:
            edge = 0.5 * float(first in tonic_set) + 0.5 * float(last in tonic_set)

        scores[index] = (
            diatonic_time / total
            + TONIC_TIME_WEIGHT * (tonic_time / total)
            + TONIC_EDGE_WEIGHT * edge
        )

    return scores


def _pooled_chroma(features: Features) -> np.ndarray:
    """Loudness-weighted average chroma over the non-silent part of the track."""
    mask = ~features.silent
    if not mask.any():
        mask = np.ones(features.n_frames, dtype=bool)

    chroma = features.chroma[:, mask]
    weights = features.rms[mask]
    if weights.sum() <= 0:
        return chroma.mean(axis=1)
    return chroma @ (weights / weights.sum())


def _chords_in_window(
    chords: list[ChordSegment] | None, start: float, end: float
) -> list[ChordSegment] | None:
    """Chords clipped to ``[start, end)``, so window scores use window-local time."""
    if not chords:
        return None
    clipped = []
    for chord in chords:
        overlap_start = max(chord.start, start)
        overlap_end = min(chord.end, end)
        if overlap_end > overlap_start:
            clipped.append(
                ChordSegment(
                    start=overlap_start,
                    end=overlap_end,
                    label=chord.label,
                    confidence=chord.confidence,
                )
            )
    return clipped or None


def detect_modulations(
    features: Features,
    global_key: tuple[str, str],
    window: float = MODULATION_WINDOW,
    hop: float = MODULATION_HOP,
    chords: list[ChordSegment] | None = None,
) -> list[Modulation]:
    """Sliding-window key scan, reporting stable departures from the global key."""
    duration = float(features.times[-1]) if features.n_frames else 0.0
    if duration < window * 1.5:
        return []

    starts = np.arange(0.0, max(duration - window, 0.0) + 1e-6, hop)
    if starts.size < MODULATION_MIN_WINDOWS:
        return []

    # Per-window scores over all 24 keys.  Windows that are mostly silent score
    # flat, contributing nothing either way.
    spans: list[tuple[float, float]] = []
    score_rows: list[np.ndarray] = []
    for start in starts:
        end = min(start + window, duration)
        spans.append((start, end))

        mask = (features.times >= start) & (features.times < end) & (~features.silent)
        if mask.sum() < 4:
            score_rows.append(np.zeros(len(_TEMPLATE_NAMES)))
            continue

        pooled = features.chroma[:, mask].mean(axis=1)
        scores = _standardise(score_chroma_vector(pooled))

        local_chords = _chords_in_window(chords, start, end)
        if local_chords:
            # No edge bonus here: see chord_evidence_scores().
            scores = scores + KEY_CHORD_WEIGHT * _standardise(
                chord_evidence_scores(local_chords, use_edges=False)
            )
        score_rows.append(scores)

    # The local key is a piecewise-constant latent observed through noisy
    # windows, so decode it the same way we decode chords rather than trusting
    # each window's argmax.  Neighbouring windows overlap heavily and a key and
    # its relative score almost identically, which makes raw per-window winners
    # flicker exactly where the key is in fact steady.
    emissions = np.stack(score_rows, axis=0) / _CONFIDENCE_TEMPERATURE
    path = viterbi_decode(emissions, self_prob=MODULATION_SELF_PROB)

    posterior = np.exp(emissions - emissions.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    window_confidence = posterior[np.arange(path.size), path]

    modulations: list[Modulation] = []
    index = 0
    while index < len(path):
        run_end = index
        while run_end + 1 < len(path) and path[run_end + 1] == path[index]:
            run_end += 1

        run_key = _TEMPLATE_NAMES[int(path[index])]
        length = run_end - index + 1
        if run_key != global_key and length >= MODULATION_MIN_WINDOWS:
            modulations.append(
                Modulation(
                    start=spans[index][0],
                    end=spans[run_end][1],
                    tonic=run_key[0],
                    mode=run_key[1],
                    confidence=float(np.mean(window_confidence[index : run_end + 1])),
                )
            )
        index = run_end + 1

    return modulations


def detect_key(
    features: Features,
    scan_modulations: bool = True,
    chords: list[ChordSegment] | None = None,
) -> KeyEstimate:
    """Global key estimate for a track, with alternatives and modulations.

    Pass ``chords`` (the decoded chord track) to enable the chord-evidence term;
    without it this falls back to pure profile correlation, which cannot resolve
    relative major/minor.
    """
    pooled = _pooled_chroma(features)
    chord_scores = chord_evidence_scores(chords) if chords else None
    best, confidence, alternatives = estimate_key_from_chroma(pooled, chord_scores=chord_scores)

    modulations: list[Modulation] = []
    if scan_modulations:
        modulations = detect_modulations(features, (best.tonic, best.mode), chords=chords)

    return KeyEstimate(
        tonic=best.tonic,
        mode=best.mode,
        confidence=confidence,
        alternatives=alternatives,
        modulations=modulations,
    )
