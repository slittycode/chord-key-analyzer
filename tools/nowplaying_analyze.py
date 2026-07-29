#!/usr/bin/env python3
"""Analyse whatever Music.app is currently playing.

Asks Music.app what is playing and where the file is, then hands that path to
``cka analyze``.  Standard library only, and it does not import the package: it
drives ``osascript`` and the ``cka`` executable through :mod:`subprocess`, so it
runs from a plain system Python against whatever ``cka`` is on ``PATH``.

    python tools/nowplaying_analyze.py
    python tools/nowplaying_analyze.py --save result.json
    python tools/nowplaying_analyze.py --file some_track.flac

Subscription-protected tracks are detected and refused with an explanation.
There is no fallback that reaches the stream — see ``docs/APPLE-MUSIC.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Where the fuzzy fallback looks when Music.app has no location for a track.
SEARCH_ROOTS = (Path.home() / "Music", Path.home() / "slskd")

#: Extensions worth handing to the analyser.  ``.m4p`` is deliberately absent:
#: it is the protected container, and finding one is a reason to stop.
AUDIO_SUFFIXES = frozenset({".flac", ".wav", ".aiff", ".aif", ".mp3", ".m4a", ".ogg", ".opus"})

#: Minimum share of the track's title/artist words a filename must contain
#: before it is accepted.  Deliberately high: the failure mode of a loose match
#: is analysing the wrong song and reporting the result as if it were right.
MATCH_THRESHOLD = 0.75

#: Words that carry no identifying information and would inflate any score.
STOP_WORDS = frozenset({"the", "a", "an", "and", "feat", "ft", "remix", "edit", "mix", "original"})

#: ASCII unit separator — a field delimiter that cannot occur in a track title.
DELIM = "\x1f"

_TRACK_QUERY = """
tell application "Music"
    set d to character id 31
    try
        set t to current track
    on error
        return "no_track"
    end try
    set trackKind to ""
    try
        set trackKind to (kind of t) as text
    end try
    set trackCloud to ""
    try
        set trackCloud to (cloud status of t) as text
    end try
    set trackLoc to ""
    try
        set trackLoc to POSIX path of (location of t)
    end try
    return "ok" & d & (player state as text) & d & (name of t) & d & ¬
        (artist of t) & d & (album of t) & d & trackKind & d & trackCloud & d & trackLoc
end tell
"""


def _osascript(script: str) -> str:
    """Run ``script`` and return its stdout, or ``""`` if AppleScript failed."""
    try:
        done = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def music_is_running() -> bool:
    """Whether Music.app is already running.

    Asked through System Events rather than by touching Music.app itself, because
    addressing an application in AppleScript is enough to launch it — and a tool
    that reports what is playing should never be the reason something starts.
    """
    query = 'tell application "System Events" to (name of processes) contains "Music"'
    return _osascript(query) == "true"


def current_track() -> dict[str, str] | None:
    """State, title, artist, album, kind, cloud status and path for what is playing.

    One AppleScript call rather than one per field, so the answer describes a
    single track instead of whichever track was playing at each of seven queries.
    """
    if not music_is_running():
        return None

    raw = _osascript(_TRACK_QUERY)
    if not raw.startswith("ok" + DELIM):
        return None

    keys = ("state", "name", "artist", "album", "kind", "cloud", "location")

    # Trailing empty fields do not survive the round trip through osascript, so a
    # track with no location comes back short rather than with a blank last
    # field.  That is the common case, not an error: pad instead of rejecting.
    fields = raw.split(DELIM)[1:]
    if len(fields) < 3:
        return None
    fields += [""] * (len(keys) - len(fields))

    return dict(zip(keys, fields, strict=True))


def _words(text: str) -> list[str]:
    """Lowercase alphanumeric words, minus the ones that identify nothing."""
    found = re.split(r"[^a-z0-9]+", text.lower())
    return [word for word in found if len(word) > 1 and word not in STOP_WORDS]


def _score(wanted: list[str], candidate: Path) -> float:
    """Share of ``wanted`` words appearing in the filename or its parent folder."""
    if not wanted:
        return 0.0
    hay = set(_words(f"{candidate.stem} {candidate.parent.name}"))
    return sum(1 for word in wanted if word in hay) / len(wanted)


def find_by_name(artist: str, title: str) -> tuple[Path, float] | None:
    """Best filename match for ``artist``/``title`` under :data:`SEARCH_ROOTS`.

    Returns ``None`` rather than a weak guess — see :data:`MATCH_THRESHOLD`.
    """
    wanted = _words(f"{artist} {title}")
    if not wanted:
        return None

    best: tuple[Path, float] | None = None
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for dirpath, _, filenames in os.walk(root, followlinks=False):
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.suffix.lower() not in AUDIO_SUFFIXES:
                    continue
                score = _score(wanted, path)
                if best is None or score > best[1]:
                    best = (path, score)

    return best if best and best[1] >= MATCH_THRESHOLD else None


def protection_problem(track: dict[str, str]) -> str | None:
    """Why this track cannot be analysed, or ``None`` if nothing rules it out.

    Three independent signals, because any one of them can be absent depending
    on how the track got into the library.
    """
    location = track.get("location", "")
    kind = track.get("kind", "").lower()
    cloud = track.get("cloud", "").lower()

    if location.endswith(".m4p") or "protected" in kind:
        return "the file is a protected (DRM) container"
    if not location and cloud in {"subscription", "no longer available"}:
        return "it is an Apple Music subscription stream with no local file"
    return None


def resolve(track: dict[str, str]) -> Path:
    """The audio file for ``track``, or :class:`SystemExit` explaining why not."""
    problem = protection_problem(track)
    if problem:
        raise SystemExit(
            f"Cannot analyse {track['artist']} — {track['name']}: {problem}.\n"
            "Apple Music streams are DRM-protected and this tool will not work around\n"
            "that. Analyse a DRM-free copy (a purchase, a rip, a Bandcamp download)."
        )

    location = track.get("location", "")
    if location and Path(location).is_file():
        return Path(location)

    print("Music.app reported no file path; searching locally...", file=sys.stderr)
    match = find_by_name(track.get("artist", ""), track.get("name", ""))
    if match is None:
        raise SystemExit(
            f"Cannot locate a local file for {track['artist']} — {track['name']}.\n"
            f"Searched: {', '.join(str(r) for r in SEARCH_ROOTS)}.\n"
            "If you own a DRM-free copy elsewhere, pass it with --file."
        )

    path, score = match
    print(f"Fuzzy match ({score:.0%} of words): {path}", file=sys.stderr)
    return path


def find_cka() -> str:
    """Path to the ``cka`` executable, preferring the one beside this Python."""
    beside = Path(sys.executable).parent / "cka"
    if beside.is_file():
        return str(beside)
    found = shutil.which("cka")
    if not found:
        raise SystemExit(
            "cka is not on PATH. Activate the project venv, or: pip install chord-key-analyzer"
        )
    return found


def analyze(path: Path, duration: float | None) -> dict:
    """Run ``cka analyze`` on ``path`` and return the parsed JSON result."""
    command = [find_cka(), "analyze", str(path), "--json", "-", "--quiet"]
    if duration:
        command += ["--duration", str(duration)]

    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"cka analyze failed:\n{done.stderr.strip()}")
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"cka produced no usable JSON:\n{done.stdout[:500]}") from None


def summarise(result: dict, track: dict[str, str] | None) -> None:
    """Print the few numbers a listener actually wants, not the whole document."""
    if track:
        print(f"\n  {track['artist']} — {track['name']}")
        if track.get("album"):
            print(f"  {track['album']}")
    print(f"  {Path(result['file']).name}  ({result.get('duration', 0):.0f}s)\n")

    key = result["key"]
    tempo = result.get("tempo")
    print(f"  Key    {key['tonic']} {key['mode']}  (confidence {key['confidence']:.0%})")
    if tempo:
        print(f"  Tempo  {tempo:.1f} BPM")

    loop = (result.get("progression") or {}).get("main_loop")
    if loop:
        print(f"  Loop   {' '.join(loop['roman'])}   [{' '.join(loop['chords'])}]")

    chords = [c for c in result.get("chords", []) if c["label"] != "N"]
    if chords:
        # display_label is a property of the dataclass, not of the JSON; rebuild
        # the slash form here so this keeps working against 0.1.0 output too.
        shown = [c["label"] + (f"/{c['bass']}" if c.get("bass") else "") for c in chords[:8]]
        print(f"  Chords {' '.join(shown)}{' ...' if len(chords) > 8 else ''}")

    sections = result.get("sections") or []
    if sections:
        spans = " ".join(f"{s['label']}({s['start']:.0f}-{s['end']:.0f}s)" for s in sections)
        print(f"  Form   {spans}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, help="Analyse this file, skipping Music.app.")
    parser.add_argument("--save", type=Path, help="Write the full JSON result here.")
    parser.add_argument("--duration", type=float, help="Analyse only the first N seconds.")
    args = parser.parse_args()

    if args.file:
        track, path = None, args.file
        if not path.is_file():
            raise SystemExit(f"No such file: {path}")
    else:
        track = current_track()
        if track is None:
            raise SystemExit(
                "Music.app is not running, or nothing is loaded.\n"
                "Start playback and try again, or pass --file."
            )
        if track["state"] != "playing":
            state = track["state"]
            print(f"(player is {state}; analysing the loaded track anyway)", file=sys.stderr)
        path = resolve(track)

    print(f"Analysing {path.name}...", file=sys.stderr)
    result = analyze(path, args.duration)
    summarise(result, track)

    if args.save:
        args.save.write_text(json.dumps(result, indent=2))
        print(f"  Full result: {args.save}\n")


if __name__ == "__main__":
    main()
