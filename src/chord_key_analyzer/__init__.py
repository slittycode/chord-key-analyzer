"""chord-key-analyzer — offline key, chord and progression analysis."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "analyze_audio",
    "analyze_source",
    "AnalysisResult",
    "ChordSegment",
    "KeyEstimate",
    "ProgressionSummary",
]


def __getattr__(name: str):
    """Lazily re-export the public API.

    Importing the package should not drag in librosa/numba (a ~2 s import),
    which matters for ``cka --version`` and for the web app's startup time.
    """
    if name in {"analyze_audio", "analyze_source"}:
        from . import pipeline

        return getattr(pipeline, name)
    if name in {"AnalysisResult", "ChordSegment", "KeyEstimate", "ProgressionSummary"}:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
