"""Dataclasses describing an analysis result.

These types are the contract between the analysis pipeline, the CLI, the web UI
and the JSON export.  Anything added here that should survive a round trip
through JSON needs a matching entry in :meth:`AnalysisResult.to_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

#: Chord qualities understood by the v1 template engine, mapped to the pitch
#: class intervals (in semitones above the root) that make up the chord.
CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
}

# Four further qualities were measured against this vocabulary and rejected.
# Each is recorded here so the next person does not have to measure it twice:
#
# * ``sus4`` (0,5,7) is collision-free, decodes its own renders 7 roots out of 7,
#   and changes no label on any existing fixture — and still fails.  A suspended
#   template is too good a match for melody: two stepwise notes blurred together
#   by the chroma median filter, plus their fifths, *are* a sus chord, so a bare
#   scale decodes as a sus4 on every degree and the key evidence collapses with
#   it (A minor read as C major, D minor as F major).  Held out until the
#   emission model can tell a sounding fourth from a passing one.
# * ``hdim7`` (0,3,6,10) is collision-free but loses its own renders to the plain
#   diminished triad at all 12 roots — the dim triad's partials already energise
#   the flat seventh's bin.
# * ``dim7`` (0,3,6,9) is symmetric: three distinct sets across twelve roots, so
#   the canonical root is arbitrary without a confident bass.
# * ``min6`` is enharmonically ``hdim7``, which is not here to be re-spelled from.

#: Qualities that exist only as *labels*, never as decoder states, because each
#: shares its exact pitch-class set with a state: C:maj6 is A:min7 at every root.
#: A template decoder cannot choose between two identical templates — it would
#: keep whichever spelling the deduplication happened to see first — so these are
#: assigned afterwards from the bass, the one thing that does tell them apart.
#: See :func:`~chord_key_analyzer.chords.respell_with_bass`.
RESPELLED_QUALITIES: dict[str, tuple[int, ...]] = {
    "maj6": (0, 4, 7, 9),
}

#: The subset used when ``--triads-only`` is passed.
TRIAD_QUALITIES = ("maj", "min", "dim", "aug")

#: Label for the "no chord" state (silence, percussion, ambiguous texture).
NO_CHORD = "N"

#: Interval above the root (in semitones) -> the MIREX degree that names it, for
#: slash chords.  0 has no entry on purpose: a bass on the root is root position,
#: which is written without a slash at all.
BASS_DEGREES = {
    1: "b2",
    2: "2",
    3: "b3",
    4: "3",
    5: "4",
    6: "b5",
    7: "5",
    8: "b6",
    9: "6",
    10: "b7",
    11: "7",
}


def chord_label(root: int, quality: str) -> str:
    """Build a chord label such as ``C:maj`` from a pitch class and quality."""
    return f"{PITCH_CLASSES[root % 12]}:{quality}"


def parse_chord_label(label: str) -> tuple[int, str] | None:
    """Inverse of :func:`chord_label`.  Returns ``None`` for the no-chord label."""
    if label == NO_CHORD or ":" not in label:
        return None
    root_name, quality = label.split(":", 1)
    try:
        return PITCH_CLASSES.index(root_name), quality
    except ValueError:
        return None


@dataclass(frozen=True)
class KeyCandidate:
    """One entry in the ranked list of plausible keys."""

    tonic: str
    mode: str
    score: float

    @property
    def name(self) -> str:
        return f"{self.tonic} {self.mode}"

    def to_dict(self) -> dict[str, Any]:
        return {"tonic": self.tonic, "mode": self.mode, "score": round(self.score, 4)}


@dataclass(frozen=True)
class Modulation:
    """A stretch of the track whose local key differs from the global key."""

    start: float
    end: float
    tonic: str
    mode: str
    confidence: float

    @property
    def name(self) -> str:
        return f"{self.tonic} {self.mode}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "tonic": self.tonic,
            "mode": self.mode,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class KeyEstimate:
    """Global key estimate plus alternatives and any detected modulations."""

    tonic: str
    mode: str
    confidence: float
    alternatives: list[KeyCandidate] = field(default_factory=list)
    modulations: list[Modulation] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.tonic} {self.mode}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tonic": self.tonic,
            "mode": self.mode,
            "confidence": round(self.confidence, 4),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "modulations": [m.to_dict() for m in self.modulations],
        }


@dataclass(frozen=True)
class ChordSegment:
    """A single chord occupying ``[start, end)`` seconds of the track.

    ``label`` never carries the bass.  Everything that reads a label — the
    parser, the Roman-numeral writer, the key evidence term, the web timeline —
    is about the chord itself, and a slash in there would mean teaching all of
    them to strip it.  The bass lives in its own field, and the two are combined
    only where a combined form is wanted (see :attr:`mirex_label` and
    :attr:`display_label`).
    """

    start: float
    end: float
    label: str
    confidence: float = 0.0
    #: Pitch class of the sounding bass note, when it is a chord tone other than
    #: the root and the low end was clear enough to be sure.  ``None`` otherwise
    #: — including for root-position chords, which need no slash.  Last, and with
    #: a default, so existing positional constructors keep working.
    bass: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_no_chord(self) -> bool:
        return self.label == NO_CHORD

    @property
    def bass_degree(self) -> str | None:
        """The bass as a degree above the root (``"3"``, ``"b7"``), or ``None``."""
        if self.bass is None:
            return None
        parsed = parse_chord_label(self.label)
        if parsed is None:
            return None
        try:
            bass_pc = PITCH_CLASSES.index(self.bass)
        except ValueError:
            return None
        return BASS_DEGREES.get((bass_pc - parsed[0]) % 12)

    @property
    def mirex_label(self) -> str:
        """Label for a ``.lab`` file, where a slash bass is written as a *degree*.

        ``C:maj/E`` is not a label mir_eval will parse; ``C:maj/3`` is.  Degrees
        are what the annotation format speaks, so exports use this form and
        round-trip cleanly through mir_eval.
        """
        degree = self.bass_degree
        return self.label if degree is None else f"{self.label}/{degree}"

    @property
    def display_label(self) -> str:
        """Label as a musician writes it, ``C:maj/E`` — for humans, not files."""
        return self.label if self.bass is None else f"{self.label}/{self.bass}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bass": self.bass,
        }


@dataclass(frozen=True)
class Loop:
    """The dominant repeating chord cycle found in the track."""

    labels: list[str]
    roman: list[str]
    repeats: int
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "roman": list(self.roman),
            "repeats": self.repeats,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }


@dataclass(frozen=True)
class ProgressionSummary:
    """Roman-numeral view of the chord timeline."""

    roman: list[str]
    labels: list[str]
    main_loop: Loop | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "roman": list(self.roman),
            "labels": list(self.labels),
            "main_loop": self.main_loop.to_dict() if self.main_loop else None,
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Everything the pipeline produces for one track."""

    file: str
    duration: float
    key: KeyEstimate
    chords: list[ChordSegment]
    progression: ProgressionSummary
    tempo: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "file": self.file,
            "duration": round(self.duration, 3),
            "tempo": round(self.tempo, 2) if self.tempo else None,
            "key": self.key.to_dict(),
            "chords": [c.to_dict() for c in self.chords],
            "progression": self.progression.to_dict(),
            "meta": dict(self.meta),
        }
