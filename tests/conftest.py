"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fixtures as fx  # noqa: E402


@pytest.fixture(scope="session")
def pop_audio():
    """I–V–vi–IV in C major, three times through at 2 s per chord."""
    return fx.render_progression(fx.POP_LOOP_C, chord_duration=2.0, repeats=3)


@pytest.fixture(scope="session")
def pop_wav(tmp_path_factory, pop_audio):
    path = tmp_path_factory.mktemp("audio") / "pop.wav"
    return fx.write_wav(path, pop_audio)


@pytest.fixture(scope="session")
def pop_features(pop_audio):
    from chord_key_analyzer.features import extract_features

    return extract_features(pop_audio, fx.SR)


@pytest.fixture(scope="session")
def pop_result(pop_audio):
    from chord_key_analyzer.pipeline import analyze_audio

    return analyze_audio(pop_audio, sr=fx.SR, source="pop.wav")
