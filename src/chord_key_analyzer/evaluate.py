"""Accuracy evaluation against reference annotations.

Scores this analyzer's output against ground-truth chord and key labels using
:mod:`mir_eval`, the reference implementation of the MIREX metrics — so the
numbers are comparable with published results instead of being a private
invention of this project.

Reference annotations are never downloaded.  Point ``cka eval`` at a directory
you assembled yourself: audio files, each beside a same-stem ``.lab`` of chord
labels and optionally a ``.key`` file.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Extensions treated as audio when matching a ``.lab`` to its recording.
AUDIO_EXTENSIONS = (
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".mp3",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
)

#: The chord metrics reported, in display order.
#:
#: ``root`` ignores quality entirely; ``majmin`` collapses everything to major
#: or minor; ``sevenths`` is the strictest, holding the analyzer to the exact
#: seventh; ``mirex`` counts a chord correct when it shares at least three
#: pitch classes with the reference, so it reads highest; ``seg`` scores only
#: where the boundaries fall, ignoring the labels.
CHORD_METRICS = ("root", "majmin", "sevenths", "mirex", "seg")

_TONIC = re.compile(r"^[A-G][#b]?$")

_MODE_ALIASES = {
    "major": "major",
    "maj": "major",
    "": "major",
    "minor": "minor",
    "min": "minor",
    "m": "minor",
}

#: Segment labels in a key annotation that carry no key.
_NON_KEY_LABELS = frozenset({"silence", "s", "n", "none", "x", "no key"})


class EvalExtraMissing(RuntimeError):
    """Raised when the [eval] extra is not installed."""

    MESSAGE = (
        "Accuracy evaluation needs mir_eval, which is not installed.\n"
        "Install it with: pip install 'chord-key-analyzer[eval]'"
    )

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)


def _require_mir_eval():
    """Import mir_eval, or explain how to install it."""
    try:
        import mir_eval
    except ImportError as exc:  # pragma: no cover - exercised by uninstalling the extra
        raise EvalExtraMissing() from exc
    return mir_eval


def require_eval_extra() -> None:
    """Raise :class:`EvalExtraMissing` unless the [eval] extra is importable.

    The public, side-effect-free door for callers that want to fail before doing
    any work — ``cka eval`` checks it before walking the dataset, so a missing
    extra prints the install hint instead of a discovery listing followed by an
    error.  Mirrors :func:`~chord_key_analyzer.web.require_web_extra`.
    """
    _require_mir_eval()


@dataclass(frozen=True)
class EvalPair:
    """One evaluable track: audio plus the references that describe it."""

    name: str
    audio: Path
    chords: Path
    key: Path | None = None


@dataclass
class TrackEvaluation:
    """Scores for a single track, or the reason it could not be scored."""

    name: str
    duration: float = 0.0
    chord_scores: dict[str, float] = field(default_factory=dict)
    key_score: float | None = None
    reference_key: str | None = None
    estimated_key: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration": round(self.duration, 3),
            "chords": {k: round(v, 4) for k, v in self.chord_scores.items()},
            "key_score": None if self.key_score is None else round(self.key_score, 4),
            "reference_key": self.reference_key,
            "estimated_key": self.estimated_key,
            "error": self.error,
        }


def _extension_of(path: Path, stem: str) -> str | None:
    """The lowercased extension ``path`` carries past ``stem``, or ``None``.

    ``Track.key.lab`` past stem ``Track`` is ``key.lab``.  The stem is compared
    case-sensitively on purpose: on a case-sensitive filesystem ``Song.lab`` and
    ``song.lab`` are two different annotations, and quietly handing one of them
    the other's audio is worse than reporting an orphan.  Only the extension is
    case-folded, because ``.WAV`` really is a wav.
    """
    if not path.name.startswith(f"{stem}."):
        return None
    return path.name[len(stem) + 1 :].lower()


def _audio_for(lab: Path) -> Path | None:
    """The recording a ``.lab`` describes: same directory, same stem.

    Preference follows :data:`AUDIO_EXTENSIONS`, then name, so a directory
    holding both ``track.wav`` and ``track.mp3`` resolves the same way twice.
    """
    candidates = [
        (AUDIO_EXTENSIONS.index(f".{extension}"), path.name, path)
        for path in lab.parent.iterdir()
        if path.is_file()
        and (extension := _extension_of(path, lab.stem)) is not None
        and f".{extension}" in AUDIO_EXTENSIONS
    ]
    if not candidates:
        return None
    return min(candidates)[2]


def _key_for(lab: Path) -> Path | None:
    """The key annotation beside a chord ``.lab``, if there is one.

    Same case rule as :func:`_audio_for`: exact stem, any spelling of the
    extension.  A bare ``.key`` wins over Isophonics' ``.key.lab``.
    """
    beside = sorted(path for path in lab.parent.iterdir() if path.is_file())
    for extension in ("key", "key.lab"):
        for path in beside:
            if _extension_of(path, lab.stem) == extension:
                return path
    return None


def discover_pairs(root: str | Path) -> tuple[list[EvalPair], list[Path]]:
    """Find evaluable tracks under ``root``.

    Returns the pairs and, separately, any chord annotations whose audio is
    missing.  A dataset where one recording could not be shared is still worth
    evaluating on the rest, so those are reported rather than raised.
    """
    root = Path(root)
    pairs: list[EvalPair] = []
    orphans: list[Path] = []

    # Walked rather than globbed for `*.lab` so `Track.LAB` is found too: a
    # dataset assembled on a case-preserving filesystem should not silently
    # contribute nothing.
    for lab in sorted(root.rglob("*")):
        if not lab.is_file() or lab.suffix.lower() != ".lab":
            continue
        # `song.key.lab` is a key reference, not a chord annotation.
        if lab.name.lower().endswith(".key.lab"):
            continue
        audio = _audio_for(lab)
        if audio is None:
            orphans.append(lab)
            continue
        pairs.append(
            EvalPair(
                name=str(lab.relative_to(root).with_suffix("")),
                audio=audio,
                chords=lab,
                key=_key_for(lab),
            )
        )

    return pairs, orphans


def _normalise_key_string(text: str) -> str | None:
    """``E:minor`` / ``Key\tE`` / ``C`` → ``"E minor"`` / ``"E major"`` / ``"C major"``.

    ``None`` for anything this project cannot score: modal annotations
    (``G:mixolydian``) and unparseable junk alike.  A dataset with one odd line
    should lose that track's key score, not crash the run.
    """
    cleaned = text.strip().replace(":", " ").replace("\t", " ")
    if not cleaned:
        return None

    tokens = cleaned.split()
    # Isophonics key files label the segment "Key" before naming it.
    if tokens and tokens[0].lower() == "key":
        tokens = tokens[1:]
    if not tokens:
        return None
    if " ".join(tokens).lower() in _NON_KEY_LABELS:
        return None

    tonic = tokens[0].capitalize() if len(tokens[0]) == 1 else tokens[0][0].upper() + tokens[0][1:]
    if not _TONIC.match(tonic):
        return None

    mode = _MODE_ALIASES.get(" ".join(tokens[1:]).lower())
    if mode is None:
        return None
    return f"{tonic} {mode}"


def load_key_reference(path: str | Path) -> str | None:
    """Read a key annotation as ``"<tonic> <mode>"``, or ``None`` if unusable.

    Handles both a bare key on one line and Isophonics-style segment files, where
    the tonality holding the most time wins and non-key segments are ignored.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None

    lines = [line for line in (raw.strip() for raw in text.splitlines()) if line]
    if not lines:
        return None

    durations: dict[str, float] = {}
    for line in lines:
        fields = line.split()
        # A segment line starts with two timestamps; anything else is a bare key.
        if len(fields) >= 3:
            try:
                start, end = float(fields[0]), float(fields[1])
            except ValueError:
                key = _normalise_key_string(line)
                if key:
                    durations[key] = durations.get(key, 0.0) + 1.0
                continue
            key = _normalise_key_string(" ".join(fields[2:]))
            if key:
                durations[key] = durations.get(key, 0.0) + max(end - start, 0.0)
        else:
            key = _normalise_key_string(line)
            if key:
                durations[key] = durations.get(key, 0.0) + 1.0

    if not durations:
        return None
    return max(durations.items(), key=lambda item: item[1])[0]


def evaluate_track(
    pair: EvalPair,
    engine: str = "template",
    triads_only: bool = False,
) -> TrackEvaluation:
    """Analyse one track and score it against its references."""
    import numpy as np

    from .ingest import IngestError
    from .pipeline import analyze_source

    mir_eval = _require_mir_eval()
    # Not a ValueError: mir_eval raises this straight off Exception, so it has to
    # be named explicitly wherever a reference label reaches the scorer.
    invalid_chord = mir_eval.chord.InvalidChordException
    evaluation = TrackEvaluation(name=pair.name)

    try:
        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(str(pair.chords))
    except (OSError, ValueError) as exc:
        evaluation.error = f"unreadable chord reference: {exc}"
        return evaluation

    try:
        result = analyze_source(str(pair.audio), engine=engine, triads_only=triads_only)
    except (IngestError, ValueError) as exc:
        # One unreadable file must not take the whole corpus run down with it.
        evaluation.error = str(exc)
        return evaluation

    evaluation.duration = result.duration

    if result.chords:
        est_intervals = np.array([[c.start, c.end] for c in result.chords], dtype=float)
        est_labels = [c.label for c in result.chords]
    else:
        est_intervals = np.array([[0.0, max(result.duration, 1e-3)]], dtype=float)
        est_labels = ["N"]

    # mir_eval pads and aligns the two interval sets itself, so no hand-rolled
    # boundary matching is needed here.  It does, however, parse every reference
    # label for the first time right here — `load_labeled_intervals` above reads
    # the file without validating it — so one unparseable line in a hand-made
    # `.lab` surfaces as an exception at this point.  Record it against the track
    # and keep going, exactly as an undecodable audio file does.
    try:
        scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
    except (ValueError, invalid_chord) as exc:
        evaluation.error = f"chord scoring failed: {exc}"
        return evaluation

    evaluation.chord_scores = {
        metric: float(scores[metric]) for metric in CHORD_METRICS if metric in scores
    }

    if pair.key is not None:
        reference_key = load_key_reference(pair.key)
        if reference_key is not None:
            evaluation.reference_key = reference_key
            evaluation.estimated_key = result.key.name
            # load_key_reference() already rejects what it cannot normalise, so
            # this is close to unreachable — but mir_eval owns the final say on
            # what a key string is, and a corpus run must not die on its verdict.
            try:
                evaluation.key_score = float(
                    mir_eval.key.weighted_score(reference_key, result.key.name)
                )
            except ValueError as exc:
                evaluation.error = f"key scoring failed: {exc}"

    return evaluation


def summarise(tracks: list[TrackEvaluation]) -> dict[str, Any]:
    """Corpus totals: duration-weighted for chords, plain mean for key.

    Weighting the chord metrics by track length makes the corpus number a
    weighted chord symbol recall — the standard MIREX figure — rather than an
    average of per-track averages that lets a 20 s clip outvote a 6 min song.
    """
    scored = [t for t in tracks if t.ok]
    summary: dict[str, Any] = {
        "tracks": len(tracks),
        "scored": len(scored),
        "failed": len(tracks) - len(scored),
    }

    total_duration = sum(t.duration for t in scored)
    summary["duration"] = round(total_duration, 3)

    for metric in CHORD_METRICS:
        values = [(t.chord_scores[metric], t.duration) for t in scored if metric in t.chord_scores]
        weight = sum(duration for _, duration in values)
        if not values or weight <= 0:
            summary[metric] = None
            continue
        summary[metric] = round(
            sum(score * duration for score, duration in values) / weight, 4
        )

    key_scores = [t.key_score for t in scored if t.key_score is not None]
    summary["key"] = round(sum(key_scores) / len(key_scores), 4) if key_scores else None
    summary["key_tracks"] = len(key_scores)
    return summary


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_report(
    tracks: list[TrackEvaluation],
    summary: dict[str, Any],
    console: Any = None,
) -> None:
    """Print the per-track table and the corpus summary."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from .output import format_time

    console = console or Console()

    table = Table(title="Accuracy vs reference annotations", header_style="bold")
    table.add_column("Track", overflow="fold")
    table.add_column("Dur", justify="right")
    for metric in CHORD_METRICS:
        table.add_column(metric.capitalize() if metric != "mirex" else "MIREX", justify="right")
    table.add_column("Key", justify="right")

    for track in tracks:
        if not track.ok:
            table.add_row(track.name, "—", *["—"] * len(CHORD_METRICS), "—", style="red")
            continue
        table.add_row(
            track.name,
            format_time(track.duration),
            *[_format_score(track.chord_scores.get(m)) for m in CHORD_METRICS],
            _format_score(track.key_score),
        )

    console.print(table)

    lines = [
        f"Tracks: {summary['scored']} scored"
        + (f", [red]{summary['failed']} failed[/red]" if summary["failed"] else "")
    ]
    lines.append(
        "  ".join(
            f"{metric if metric != 'mirex' else 'MIREX'} {_format_score(summary.get(metric))}"
            for metric in CHORD_METRICS
        )
    )
    if summary.get("key") is not None:
        lines.append(f"Key: {_format_score(summary['key'])} over {summary['key_tracks']} track(s)")
    console.print(Panel("\n".join(lines), title="Summary", border_style="cyan"))

    for track in tracks:
        if not track.ok:
            console.print(Text.assemble((f"{track.name}: ", "red"), track.error or ""))


def report_payload(tracks: list[TrackEvaluation], summary: dict[str, Any]) -> dict[str, Any]:
    return {"summary": summary, "tracks": [t.to_dict() for t in tracks]}


def write_report_json(
    tracks: list[TrackEvaluation], summary: dict[str, Any], destination: str
) -> None:
    """Write the report as JSON; ``-`` means stdout."""
    payload = json.dumps(report_payload(tracks, summary), indent=2)
    if destination == "-":
        sys.stdout.write(payload + "\n")
        return
    Path(destination).write_text(payload + "\n", encoding="utf-8")


def write_report_csv(
    tracks: list[TrackEvaluation], summary: dict[str, Any], destination: str
) -> None:
    """Write one row per track, then a ``__summary__`` row; ``-`` means stdout."""
    columns = ["name", "duration", *CHORD_METRICS, "key_score", "error"]

    def rows() -> list[dict[str, Any]]:
        out = []
        for track in tracks:
            row: dict[str, Any] = {
                "name": track.name,
                "duration": round(track.duration, 3),
                "key_score": track.key_score,
                "error": track.error or "",
            }
            for metric in CHORD_METRICS:
                row[metric] = track.chord_scores.get(metric)
            out.append(row)
        out.append(
            {
                "name": "__summary__",
                "duration": summary.get("duration"),
                "key_score": summary.get("key"),
                "error": "",
                **{metric: summary.get(metric) for metric in CHORD_METRICS},
            }
        )
        return out

    if destination == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows())
        return

    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows())
