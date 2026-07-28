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

#: The subset used when ``--triads-only`` is passed.
TRIAD_QUALITIES = ("maj", "min", "dim", "aug")

#: Label for the "no chord" state (silence, percussion, ambiguous texture).
NO_CHORD = "N"


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
    """A single chord occupying ``[start, end)`` seconds of the track."""

    start: float
    end: float
    label: str
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_no_chord(self) -> bool:
        return self.label == NO_CHORD

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "label": self.label,
            "confidence": round(self.confidence, 4),
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
