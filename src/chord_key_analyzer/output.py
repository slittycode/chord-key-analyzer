"""Rendering: rich terminal output, JSON export, MIREX ``.lab`` export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import AnalysisResult
from .progression import chord_histogram, roman_numeral

#: Chord timeline rows longer than this get truncated with a summary line.
MAX_TIMELINE_ROWS = 40


def format_time(seconds: float) -> str:
    """``m:ss.d`` — compact but precise enough to scrub to."""
    minutes, remainder = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes)}:{remainder:04.1f}"


def to_json(result: AnalysisResult, indent: int = 2) -> str:
    return json.dumps(result.to_dict(), indent=indent)


def _write_text(payload: str, destination: str) -> None:
    """Write ``payload`` to ``destination`` verbatim; ``-`` means stdout."""
    if destination == "-":
        sys.stdout.write(payload)
        return
    Path(destination).write_text(payload, encoding="utf-8")


def write_json(result: AnalysisResult, destination: str) -> None:
    """Write JSON to ``destination``; ``-`` means stdout."""
    _write_text(to_json(result) + "\n", destination)


def to_lab(result: AnalysisResult) -> str:
    """MIREX-style chord annotation: ``start<TAB>end<TAB>label`` per line.

    Labels are written in the ``.lab`` dialect, so a detected bass appears as a
    degree slash (``C:maj/3``) — the spelling mir_eval parses.
    """
    return "".join(f"{c.start:.3f}\t{c.end:.3f}\t{c.mirex_label}\n" for c in result.chords)


def write_lab(result: AnalysisResult, destination: str) -> None:
    _write_text(to_lab(result), destination)


def _confidence_style(confidence: float) -> str:
    if confidence >= 0.66:
        return "green"
    if confidence >= 0.4:
        return "yellow"
    return "red"


def _key_panel(result: AnalysisResult) -> Panel:
    key = result.key
    body = Text()
    body.append(f"{key.name}\n", style="bold cyan")
    body.append("confidence  ", style="dim")
    body.append(f"{key.confidence:.0%}\n", style=_confidence_style(key.confidence))

    if key.alternatives:
        body.append("alternatives  ", style="dim")
        body.append(", ".join(a.name for a in key.alternatives) + "\n")

    if result.tempo:
        body.append("tempo  ", style="dim")
        beats = "" if result.meta.get("beats_reliable") else " (unsteady)"
        body.append(f"{result.tempo:.1f} BPM{beats}\n")

    body.append("duration  ", style="dim")
    body.append(format_time(result.duration))

    if key.modulations:
        body.append("\n\nmodulations", style="dim")
        for modulation in key.modulations:
            body.append(
                f"\n  {format_time(modulation.start)}–{format_time(modulation.end)}  "
                f"{modulation.name}"
            )

    return Panel(body, title="Key", border_style="cyan", expand=False)


def _sections_panel(result: AnalysisResult) -> Panel | None:
    """One line per structural section: letter, span, local key, progression."""
    if not result.sections:
        return None

    body = Text()
    for index, section in enumerate(result.sections):
        if index:
            body.append("\n")
        body.append(f"{section.label:<3}", style="bold cyan")
        body.append(f" {format_time(section.start)}–{format_time(section.end)}  ", style="dim")
        if section.key_name:
            body.append(f"{section.key_name}  ")
        if section.progression and section.progression.roman:
            body.append(" – ".join(section.progression.roman[:8]), style="magenta")

    return Panel(body, title="Sections", border_style="blue", expand=False)


def _chord_table(result: AnalysisResult) -> Table:
    table = Table(title="Chords", header_style="bold", expand=False)
    table.add_column("Start", justify="right", style="dim")
    table.add_column("End", justify="right", style="dim")
    table.add_column("Chord", style="bold")
    table.add_column("Roman")
    table.add_column("Conf", justify="right")

    chords = result.chords
    for chord in chords[:MAX_TIMELINE_ROWS]:
        numeral = (
            ""
            if chord.is_no_chord
            else roman_numeral(chord.label, result.key.tonic, result.key.mode)
        )
        table.add_row(
            format_time(chord.start),
            format_time(chord.end),
            chord.display_label,
            numeral,
            Text(f"{chord.confidence:.0%}", style=_confidence_style(chord.confidence)),
        )
    if len(chords) > MAX_TIMELINE_ROWS:
        remaining = len(chords) - MAX_TIMELINE_ROWS
        table.add_row("…", "…", f"[dim]{remaining} more segments[/dim]", "", "")
    return table


def _progression_panel(result: AnalysisResult) -> Panel | None:
    loop = result.progression.main_loop
    body = Text()

    if loop:
        body.append(" – ".join(loop.roman), style="bold magenta")
        body.append(f"  ×{loop.repeats}\n", style="bold")
        body.append(" – ".join(loop.labels) + "\n", style="dim")
        body.append(f"{format_time(loop.start)}–{format_time(loop.end)}\n", style="dim")
    else:
        body.append("no repeating loop detected\n", style="dim")

    histogram = chord_histogram(result.chords)[:6]
    if histogram:
        total = sum(duration for _, duration in histogram) or 1.0
        body.append("\nmost played  ", style="dim")
        body.append(", ".join(f"{label} ({duration / total:.0%})" for label, duration in histogram))

    return Panel(body, title="Progression", border_style="magenta", expand=False)


def render(result: AnalysisResult, console: Console | None = None) -> None:
    """Pretty-print a full analysis to the terminal."""
    console = console or Console()
    console.print()
    console.print(Text(result.file, style="bold white"))
    console.print(_key_panel(result))
    sections = _sections_panel(result)
    if sections is not None:
        console.print(sections)
    if result.chords:
        console.print(_chord_table(result))
        console.print(_progression_panel(result))
    else:
        console.print("[yellow]No chords detected.[/yellow]")
    console.print(
        f"[dim]engine: {result.meta.get('engine')} · cka {result.meta.get('version')}[/dim]"
    )
