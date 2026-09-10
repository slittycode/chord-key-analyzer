"""Structural section detection by self-similarity novelty.

Foote's checkerboard method: build a self-similarity matrix over the track's own
features, slide a checkerboard kernel down its diagonal, and read the peaks of
the resulting novelty curve as boundaries.  Where the music stays the same the
kernel sees one uniform block and scores low; where it changes, the kernel's four
quadrants disagree and it spikes.

Chosen over agglomerative clustering because it needs no pre-chosen number of
segments — which is the weakness of every clustering approach on material whose
structure is the thing you are trying to discover.  It costs one matrix multiply
and needs no dependency that is not already here.

Blocks are a fixed 1 s rather than beat-synchronous.  Sections are +/-2-4 s
objects, so beat resolution buys nothing real, and beat-syncing would couple this
to beat-tracker quality on exactly the rubato material where the grid is least
trustworthy — the same reasoning that keeps the chord decoder frame-based.

What comes out is a repetition analysis and nothing more.  Sections are labelled
``A``, ``B``, ``A'`` and never "verse" or "chorus": which repeated span is the
chorus is a question about song form that no self-similarity matrix can answer.
"""

from __future__ import annotations

import numpy as np

from .features import Features
from .key import estimate_key_from_chroma
from .models import ChordSegment, KeyEstimate, Section, clip_segments
from .progression import summarise_progression

#: Feature blocks are averaged over this many seconds.  Fine enough to place a
#: boundary well inside the few-second tolerance a section boundary deserves,
#: coarse enough that a five-minute track is a 300x300 matrix.
SECTION_BLOCK_SECONDS = 1.0

#: Half-width of the checkerboard kernel, in blocks.  Sets the timescale the
#: novelty curve is sensitive to: 16 s either side of a candidate boundary is
#: about one phrase of popular music, so a passing chord change does not register
#: but a change of section does.
SECTION_KERNEL_BLOCKS = 16

#: Sections shorter than this are folded into their neighbour.  Below about eight
#: seconds a "section" is a fill or a turnaround, not a structural unit.
SECTION_MIN_DURATION = 8.0

#: Tracks shorter than this get no sections at all.  Two sections need somewhere
#: for both to live, and an excerpt has structure the excerpting destroyed.
#: Mirrors the modulation scan's guard, for the same reason.
SECTION_MIN_TRACK = 30.0

#: Cosine similarity between two sections' average chroma at which they are
#: called the same section, and — lower — a variation on it (``A'``).
SECTION_SAME_LABEL = 0.92
SECTION_PRIME_LABEL = 0.80

#: Loudness matters to structure — a breakdown is a section even when its harmony
#: does not move — but it is one row against chroma's twelve.  It is z-scored
#: onto the same footing and then deliberately held below full weight.
SECTION_RMS_WEIGHT = 0.5

#: Peak-picking threshold, as a fraction of the novelty curve's own spread.  A
#: relative threshold is what lets one setting work on both a track that changes
#: constantly and one that barely moves.
SECTION_PEAK_DELTA = 0.10


def _checkerboard_kernel(half: int) -> np.ndarray:
    """A ``2*half`` square checkerboard, Gaussian-tapered away from its centre.

    The ``+ - / - +`` sign pattern asks "do the stretches either side of this
    point look like each other, and unlike each other?".  The taper keeps distant
    blocks — which may belong to a different section entirely — from dominating a
    score that is supposed to be about this boundary.
    """
    size = 2 * half
    axis = np.arange(size) - (size - 1) / 2.0
    grid_x, grid_y = np.meshgrid(axis, axis)
    taper = np.exp(-0.5 * (grid_x**2 + grid_y**2) / (half / 2.0) ** 2)
    return np.sign(grid_x * grid_y) * taper


def _block_features(features: Features) -> tuple[np.ndarray, int]:
    """Chroma plus a loudness row, averaged into fixed-length blocks."""
    block_frames = max(int(round(SECTION_BLOCK_SECONDS / features.frame_duration)), 1)
    n_blocks = features.n_frames // block_frames
    if n_blocks < 2:
        return np.zeros((13, 0)), block_frames

    usable = n_blocks * block_frames
    chroma = features.chroma[:, :usable].reshape(12, n_blocks, block_frames).mean(axis=2)
    rms = features.rms[:usable].reshape(n_blocks, block_frames).mean(axis=1)

    spread = float(rms.std())
    loudness = (rms - rms.mean()) / spread if spread > 1e-9 else np.zeros(n_blocks)
    return np.vstack([chroma, SECTION_RMS_WEIGHT * loudness[None, :]]), block_frames


def novelty_curve(blocks: np.ndarray) -> np.ndarray:
    """Foote novelty over the blocks' self-similarity matrix, scaled to ``[0, 1]``."""
    if blocks.shape[1] < 2:
        return np.zeros(blocks.shape[1])

    normalised = blocks / np.maximum(np.linalg.norm(blocks, axis=0, keepdims=True), 1e-9)
    similarity = normalised.T @ normalised

    half = SECTION_KERNEL_BLOCKS
    kernel = _checkerboard_kernel(half)
    # Edge padding lets the curve be defined at the ends without the kernel
    # falling off the matrix; those positions are never boundaries in practice
    # because a section has to be long enough to survive the minimum duration.
    padded = np.pad(similarity, half, mode="edge")
    novelty = np.array(
        [
            float((padded[index : index + 2 * half, index : index + 2 * half] * kernel).sum())
            for index in range(similarity.shape[0])
        ]
    )

    novelty = np.maximum(novelty, 0.0)
    peak = float(novelty.max())
    # Material with no internal variation gives a curve that is zero to within
    # floating-point noise, and scaling *that* by its own maximum would blow the
    # noise up into a full-scale signal with peaks all over it.  A genuine
    # novelty peak is of order one before normalising, so anything this small is
    # the flat curve it looks like.
    return novelty / peak if peak > 1e-9 else np.zeros_like(novelty)


def _boundaries(novelty: np.ndarray, min_blocks: int) -> list[int]:
    """Block indices where the novelty curve peaks, as section starts."""
    import librosa

    if novelty.size < 2 * min_blocks:
        return [0]

    # A curve with no spread has no peaks in it — but the threshold below is a
    # fraction *of that spread*, so on a flat curve it becomes zero, and
    # peak_pick then reads every position as a peak and returns one every
    # ``wait`` blocks.  The relative threshold that makes one setting work on
    # both a busy track and a static one is exactly what inverts here, so the
    # degenerate case has to be caught before it rather than by it.  Digital
    # silence is the input that reaches this.
    spread = float(novelty.std())
    if spread <= 1e-9:
        return [0]

    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=4,
        post_max=4,
        pre_avg=8,
        post_avg=8,
        delta=SECTION_PEAK_DELTA * spread,
        wait=min_blocks,
    )

    starts = [0]
    for peak in peaks:
        index = int(peak)
        # Both the new section and what is left of the previous one have to clear
        # the minimum, or the boundary buys nothing.
        if index - starts[-1] >= min_blocks and novelty.size - index >= min_blocks:
            starts.append(index)
    return starts


def _letter(index: int) -> str:
    """``A``, ``B``, ... ``Z``, ``AA`` — more than any real track will need."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _label_spans(spans: list[tuple[int, int]], blocks: np.ndarray) -> list[str]:
    """Greedy letters: how alike is each span's average chroma to one already seen?"""
    representatives: list[np.ndarray] = []
    letters: list[str] = []
    labels: list[str] = []
    silent_letter: str | None = None

    for start, end in spans:
        average = blocks[:12, start:end].mean(axis=1)
        norm = float(np.linalg.norm(average))

        if norm <= 1e-9:
            # Cosine similarity is undefined for a zero vector, and a silent
            # span dotted against any representative always scores exactly
            # 0.0 — which can never beat the strict `> best_score` floor
            # below, so it would otherwise never match a previous silent span
            # and always mint a fresh letter.  Treat every silent span as the
            # same as every other silent span instead.
            if silent_letter is None:
                silent_letter = _letter(len(representatives))
                representatives.append(average)
                letters.append(silent_letter)
            labels.append(silent_letter)
            continue

        average = average / norm

        best_index, best_score = -1, 0.0
        for index, previous in enumerate(representatives):
            score = float(average @ previous)
            if score > best_score:
                best_index, best_score = index, score

        if best_index >= 0 and best_score >= SECTION_SAME_LABEL:
            labels.append(letters[best_index])
        elif best_index >= 0 and best_score >= SECTION_PRIME_LABEL:
            labels.append(f"{letters[best_index]}'")
        else:
            letter = _letter(len(representatives))
            representatives.append(average)
            letters.append(letter)
            labels.append(letter)

    return labels


def _section_key(
    features: Features, chords: list[ChordSegment], start: float, end: float
) -> tuple[str | None, str | None, float | None]:
    """Local key for one span, or all-``None`` when there is nothing to judge."""
    mask = (features.times >= start) & (features.times < end) & (~features.silent)
    if mask.sum() < 4:
        return None, None, None

    pooled = features.chroma[:, mask].mean(axis=1)
    # No edge bonus: a section's first and last chords are where the boundary
    # detector put them, not where the music necessarily begins and ends.
    best, confidence, _ = estimate_key_from_chroma(
        pooled, chords=clip_segments(chords, start, end), use_edges=False
    )
    return best.tonic, best.mode, confidence


def detect_sections(
    features: Features,
    chords: list[ChordSegment],
    global_key: KeyEstimate,
) -> list[Section]:
    """Split a track into repeated structural spans, each with its own harmony.

    Roman numerals inside a section stay relative to ``global_key``, not to the
    section's own key hint — the point of listing them per section is to compare
    sections with each other, and renumbering each one against its own tonic
    would make two identical progressions look different.
    """
    duration = float(features.times[-1]) + features.frame_duration if features.n_frames else 0.0
    if duration < SECTION_MIN_TRACK:
        return []

    blocks, block_frames = _block_features(features)
    if blocks.shape[1] < 2:
        return []

    block_seconds = block_frames * features.frame_duration
    min_blocks = max(int(round(SECTION_MIN_DURATION / block_seconds)), 1)
    starts = _boundaries(novelty_curve(blocks), min_blocks)

    spans = [
        (start, end) for start, end in zip(starts, [*starts[1:], blocks.shape[1]], strict=True)
    ]
    labels = _label_spans(spans, blocks)

    sections: list[Section] = []
    for (start_block, end_block), label in zip(spans, labels, strict=True):
        start = start_block * block_seconds
        # The last section runs to the end of the track, not to the end of the
        # last whole block, so the sections always tile the whole duration.
        end = duration if end_block == blocks.shape[1] else end_block * block_seconds
        tonic, mode, confidence = _section_key(features, chords, start, end)
        sections.append(
            Section(
                start=start,
                end=end,
                label=label,
                tonic=tonic,
                mode=mode,
                key_confidence=confidence,
                progression=summarise_progression(clip_segments(chords, start, end), global_key),
            )
        )
    return sections
