"""The single analysis entry point shared by the CLI and the web UI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import __version__
from .chords import detect_inversions, get_engine, respell_with_bass
from .features import DEFAULT_HOP, extract_features
from .ingest import TARGET_SR, resolve_source
from .key import detect_key
from .models import AnalysisResult
from .progression import harmonic_rhythm, summarise_progression

ProgressHook = Callable[[str, float], None]


def _noop_progress(stage: str, fraction: float) -> None:
    """Default progress hook: do nothing."""


def analyze_audio(
    y: np.ndarray,
    sr: int = TARGET_SR,
    source: str = "<memory>",
    engine: str = "template",
    triads_only: bool = False,
    harmonic: bool = True,
    beat_snap: bool = True,
    scan_modulations: bool = True,
    hop_length: int = DEFAULT_HOP,
    progress: ProgressHook = _noop_progress,
) -> AnalysisResult:
    """Analyse in-memory audio and return the full result.

    This is the one code path; ``cka analyze`` and ``POST /analyze`` both land
    here, so the CLI and the web UI can never drift apart.
    """
    chord_engine = get_engine(engine, triads_only=triads_only, beat_snap=beat_snap)

    progress("features", 0.1)
    features = extract_features(y, sr, hop_length=hop_length, harmonic=harmonic)

    # Chords first: the key stage uses them to resolve relative major/minor,
    # which pitch-class content alone cannot do.  Chord decoding itself needs no
    # key, so there is no circularity here.
    progress("chords", 0.5)
    chords = chord_engine.analyze(features)
    # After the engine rather than inside it: the merge and snap helpers rebuild
    # segments from their neighbours and would drop a bass assigned earlier.
    chords = detect_inversions(chords, features)
    # And the re-spellings need that bass, so they come after it in turn.
    chords = respell_with_bass(chords)

    progress("key", 0.75)
    key = detect_key(features, scan_modulations=scan_modulations, chords=chords)

    progress("progression", 0.9)
    progression = summarise_progression(chords, key)

    progress("done", 1.0)
    return AnalysisResult(
        file=source,
        duration=len(y) / float(sr),
        key=key,
        chords=chords,
        progression=progression,
        tempo=features.tempo,
        meta={
            "engine": chord_engine.name,
            "version": __version__,
            "sample_rate": sr,
            "hop_length": hop_length,
            "tuning": round(features.tuning, 4),
            "beats_reliable": bool(features.beats_reliable),
            "harmonic_rhythm": harmonic_rhythm(chords),
            "triads_only": bool(triads_only),
        },
    )


def analyze_source(
    target: str | Path,
    engine: str = "template",
    triads_only: bool = False,
    harmonic: bool = True,
    beat_snap: bool = True,
    scan_modulations: bool = True,
    start: float = 0.0,
    duration: float | None = None,
    progress: ProgressHook = _noop_progress,
) -> AnalysisResult:
    """Load a file path or URL and analyse it."""
    progress("loading", 0.0)
    loaded = resolve_source(str(target), sr=TARGET_SR, offset=start, duration=duration)

    result = analyze_audio(
        loaded.samples,
        sr=loaded.sr,
        source=loaded.source,
        engine=engine,
        triads_only=triads_only,
        harmonic=harmonic,
        beat_snap=beat_snap,
        scan_modulations=scan_modulations,
        progress=progress,
    )

    if start:
        # All reported times are relative to the excerpt, so record where the
        # excerpt began — a consumer comparing against annotations for the full
        # track adds this to every timestamp.
        result.meta["offset"] = start
    return result
