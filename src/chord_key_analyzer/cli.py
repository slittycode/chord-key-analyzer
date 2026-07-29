"""``cka`` command line interface."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.text import Text

from . import __version__

# Runtime strings — exception messages, file paths — are printed as Text rather
# than interpolated into a markup string.  Rich reads `[...]` in markup as a style
# tag, so the install hint for the [url] extra printed as
# "pip install 'chord-key-analyzer'": the one part the user needed, silently
# eaten.  Text.assemble() styles the prefix and takes the body as literal data.

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "-V", "--version", prog_name="cka")
def main() -> None:
    """Offline key, chord and progression analysis for audio files and URLs."""


@main.command()
@click.argument("song", metavar="SONG")
@click.option("--json", "json_path", metavar="PATH", help="Write JSON result ('-' for stdout).")
@click.option("--lab", "lab_path", metavar="PATH", help="Write MIREX .lab chords ('-' for stdout).")
@click.option("--start", type=float, default=0.0, show_default=True, help="Skip to SEC seconds.")
@click.option("--duration", type=float, default=None, help="Analyse only SEC seconds.")
@click.option("--triads-only", is_flag=True, help="Restrict vocabulary to maj/min/dim/aug.")
# Only 'template' is offered here on purpose: get_engine() still knows 'deep',
# which the web UI's engine field and the Python API can both reach, but the CLI
# should not advertise a backend that is not implemented yet.
@click.option(
    "--engine",
    type=click.Choice(["template"], case_sensitive=False),
    default="template",
    show_default=True,
    help="Chord recognition backend.",
)
@click.option("--no-hpss", is_flag=True, help="Skip harmonic/percussive separation (faster).")
@click.option("--no-beat-snap", is_flag=True, help="Do not snap chord boundaries to beats.")
@click.option("--no-modulations", is_flag=True, help="Skip the sliding-window modulation scan.")
@click.option("--no-sections", is_flag=True, help="Skip the structural section scan.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress the terminal report.")
def analyze(
    song: str,
    json_path: str | None,
    lab_path: str | None,
    start: float,
    duration: float | None,
    triads_only: bool,
    engine: str,
    no_hpss: bool,
    no_beat_snap: bool,
    no_modulations: bool,
    no_sections: bool,
    quiet: bool,
) -> None:
    """Analyse SONG, a local audio file or a yt-dlp-supported URL."""
    # Imported here so `cka --help` and `cka --version` stay instant: librosa
    # and numba together cost a couple of seconds at import time.
    from .ingest import IngestError
    from .output import render, write_json, write_lab
    from .pipeline import analyze_source

    console = Console()
    # Progress and reports go to stderr so `--json -` stays pipeable.
    status_console = Console(stderr=True)

    if start < 0:
        raise click.BadParameter("--start must be >= 0")
    if duration is not None and duration <= 0:
        raise click.BadParameter("--duration must be > 0")

    stage_labels = {
        "loading": "Loading audio",
        "features": "Extracting features",
        "key": "Detecting key",
        "chords": "Recognising chords",
        "progression": "Analysing progression",
        "sections": "Finding sections",
        "done": "Done",
    }

    try:
        if quiet:
            result = analyze_source(
                song,
                engine=engine,
                triads_only=triads_only,
                harmonic=not no_hpss,
                beat_snap=not no_beat_snap,
                scan_modulations=not no_modulations,
                scan_sections=not no_sections,
                start=start,
                duration=duration,
            )
        else:
            with status_console.status("Loading audio…") as status:

                def progress(stage: str, fraction: float) -> None:
                    status.update(f"{stage_labels.get(stage, stage)}…")

                result = analyze_source(
                    song,
                    engine=engine,
                    triads_only=triads_only,
                    harmonic=not no_hpss,
                    beat_snap=not no_beat_snap,
                    scan_modulations=not no_modulations,
                    scan_sections=not no_sections,
                    start=start,
                    duration=duration,
                    progress=progress,
                )
    except IngestError as exc:
        status_console.print(Text.assemble(("Error: ", "red"), str(exc)))
        raise SystemExit(2) from exc

    # No `except ValueError` here: the only ValueError the pipeline raises is
    # get_engine()'s unknown-engine error, which click.Choice already rejects
    # before we get this far.  Catching it broadly only turned real bugs into a
    # bare exit 2.  The web UI's own handler in _run_analysis still needs it —
    # its engine field is not validated by click.
    try:
        if json_path:
            write_json(result, json_path)
        if lab_path:
            write_lab(result, lab_path)
    except OSError as exc:
        status_console.print(
            Text.assemble(("Error: ", "red"), f"cannot write output: {exc}")
        )
        raise SystemExit(2) from exc

    if not quiet:
        render(result, console=console)


@main.command(name="eval")
@click.argument("dataset", metavar="DATASET", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--json", "json_path", metavar="PATH", help="Write the report as JSON ('-' for stdout)."
)
@click.option("--csv", "csv_path", metavar="PATH", help="Write the report as CSV ('-' for stdout).")
@click.option("--triads-only", is_flag=True, help="Restrict vocabulary to maj/min/dim/aug.")
@click.option(
    "--engine",
    type=click.Choice(["template"], case_sensitive=False),
    default="template",
    show_default=True,
    help="Chord recognition backend.",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress the report table.")
def eval_cmd(
    dataset: str,
    json_path: str | None,
    csv_path: str | None,
    triads_only: bool,
    engine: str,
    quiet: bool,
) -> None:
    """Score DATASET against its reference annotations (requires the [eval] extra).

    DATASET is a directory of audio files, each beside a same-stem .lab of
    reference chords and optionally a .key file. Subdirectories are searched.
    Annotations are never downloaded — assemble the directory yourself.
    """
    # Imported here, like `analyze` does, so `cka --help` stays instant.
    from .evaluate import (
        EvalExtraMissing,
        discover_pairs,
        evaluate_track,
        render_report,
        require_eval_extra,
        summarise,
        write_report_csv,
        write_report_json,
    )

    console = Console()
    status_console = Console(stderr=True)

    # Before discovery, not per track: a missing extra should print the install
    # hint on its own, not after a listing of the tracks it will never score.
    try:
        require_eval_extra()
    except EvalExtraMissing as exc:
        raise SystemExit(str(exc)) from exc

    pairs, orphans = discover_pairs(dataset)
    for orphan in orphans:
        status_console.print(
            Text.assemble(("Skipping ", "yellow"), f"{orphan}: no audio file beside it.")
        )

    if not pairs:
        status_console.print(
            f"[red]Error:[/red] no evaluable tracks found in {dataset}.\n"
            "Expected audio files each beside a same-stem .lab of reference chords, e.g.\n"
            "  dataset/song.wav\n"
            "  dataset/song.lab\n"
            "  dataset/song.key   (optional)"
        )
        raise SystemExit(2)

    tracks = []
    for index, pair in enumerate(pairs, start=1):
        if not quiet:
            status_console.print(Text(f"({index}/{len(pairs)}) {pair.name}", style="dim"))
        tracks.append(evaluate_track(pair, engine=engine, triads_only=triads_only))

    summary = summarise(tracks)

    try:
        if json_path:
            write_report_json(tracks, summary, json_path)
        if csv_path:
            write_report_csv(tracks, summary, csv_path)
    except OSError as exc:
        status_console.print(
            Text.assemble(("Error: ", "red"), f"cannot write output: {exc}")
        )
        raise SystemExit(2) from exc

    if not quiet:
        render_report(tracks, summary, console=console)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind.")
@click.option("--port", default=8321, show_default=True, type=int, help="Port to bind.")
@click.option("--no-browser", is_flag=True, help="Do not open a browser window.")
@click.option("--no-urls", is_flag=True, help="Disable URL ingestion (uploads only).")
def web(host: str, port: int, no_browser: bool, no_urls: bool) -> None:
    """Launch the local web UI (requires the [web] extra)."""
    from .web import LOOPBACK_HOSTS, WebExtraMissing, serve

    if host not in LOOPBACK_HOSTS:
        Console(stderr=True).print(
            f"[yellow]Warning:[/yellow] binding to {host} exposes the analyzer beyond "
            "this machine. It has no authentication — only do this on a trusted network. "
            "URL ingestion is disabled on non-loopback binds."
        )

    # None lets serve() key the default off the bind; --no-urls forces it off.
    try:
        serve(
            host=host,
            port=port,
            open_browser=not no_browser,
            allow_urls=False if no_urls else None,
        )
    except WebExtraMissing as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
