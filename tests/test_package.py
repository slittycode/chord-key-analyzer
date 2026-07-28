"""Packaging invariants.

Deliberately dependency-free, so this runs in every CI leg — including the bare
install that has none of the extras.
"""

from __future__ import annotations

import pytest

from chord_key_analyzer import __version__


def test_version_matches_installed_metadata():
    """`__version__` and the version pip installed must not drift apart.

    They live in two files that nothing else connects, so the only thing keeping
    `pyproject.toml` and `__init__.py` in step is noticing when they are not.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("chord-key-analyzer")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        pytest.skip("chord-key-analyzer is not installed in this environment")

    assert installed == __version__


def test_the_public_api_is_importable_without_the_extras():
    """Everything in `__all__` has to resolve from a core install alone."""
    import chord_key_analyzer

    for name in chord_key_analyzer.__all__:
        assert getattr(chord_key_analyzer, name) is not None


def test_an_unknown_attribute_still_raises():
    import chord_key_analyzer

    with pytest.raises(AttributeError):
        _ = chord_key_analyzer.definitely_not_exported
