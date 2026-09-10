"""Synthesised audio fixtures.

Everything the test suite analyses is rendered here with numpy — no audio files
are committed and nothing is ever fetched from the network.  Notes are built
from a handful of harmonic partials with a percussive-ish envelope, which is
close enough to a plucked/struck instrument for the chroma front-end to behave
the way it does on real recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 22050

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "maj6": (0, 4, 7, 9),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
}

#: Relative amplitude of partials 1..6 of a rendered note.
PARTIAL_WEIGHTS = (1.0, 0.5, 0.32, 0.2, 0.12, 0.08)


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def note_name_to_pc(name: str) -> int:
    return PITCH_CLASSES.index(name)


def render_note(midi: float, duration: float, sr: int = SR, amplitude: float = 1.0) -> np.ndarray:
    """One note with a few harmonic partials and a soft attack/decay envelope."""
    n = int(round(duration * sr))
    t = np.arange(n) / sr
    wave = np.zeros(n, dtype=np.float64)

    for index, weight in enumerate(PARTIAL_WEIGHTS, start=1):
        frequency = midi_to_hz(midi) * index
        if frequency >= sr / 2:
            break
        # A per-partial phase offset avoids an unnaturally spiky sum at t=0.
        wave += weight * np.sin(2 * np.pi * frequency * t + index * 0.7)

    attack = max(int(0.01 * sr), 1)
    envelope = np.ones(n)
    envelope[:attack] = np.linspace(0.0, 1.0, attack)
    envelope *= np.exp(-1.6 * t)
    release = max(int(0.02 * sr), 1)
    envelope[-release:] *= np.linspace(1.0, 0.0, release)

    return amplitude * wave * envelope


def render_chord(
    root: str,
    quality: str,
    duration: float,
    sr: int = SR,
    octave: int = 4,
    with_bass: bool = True,
    bass_interval: int | None = None,
    bass_octave: int = 2,
    bass_amplitude: float = 0.8,
) -> np.ndarray:
    """A single chord: close-position voicing plus an optional low bass note.

    ``bass_interval`` (semitones above the root) puts the bass on a chosen chord
    tone down in ``bass_octave``, which is how an inversion is rendered: a C
    major triad over E is ``render_chord("C", "maj", d, bass_interval=4)``.

    By default it sounds at the same level as each note of the voicing, not
    louder.  The main chroma spans the whole spectrum, low register included, so
    a bass hot enough to dominate it moves the *chord* decode too: at 1.25x the
    voicing level a rendered C:maj/E starts decoding as A:min7.  That is a real
    property of a chroma-only recogniser rather than a fixture artefact, but a
    fixture that triggers it is testing the wrong thing.  Down in ``bass_octave``
    this note is alone anyway, which is all the low-register chroma needs.
    ``bass_amplitude`` is there for tests that need to sit either side of that
    line deliberately.

    Without ``bass_interval`` the bass just doubles the root an octave down, as
    before.
    """
    root_midi = 12 * (octave + 1) + note_name_to_pc(root)
    intervals = QUALITY_INTERVALS[quality]

    n = int(round(duration * sr))
    audio = np.zeros(n, dtype=np.float64)
    for interval in intervals:
        audio += render_note(root_midi + interval, duration, sr, amplitude=0.8)
    if bass_interval is not None:
        bass_midi = 12 * (bass_octave + 1) + note_name_to_pc(root) + bass_interval
        audio += render_note(bass_midi, duration, sr, amplitude=bass_amplitude)
    elif with_bass:
        audio += render_note(root_midi - 12, duration, sr, amplitude=1.0)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def render_progression(
    chords: list[tuple[str, str]] | list[tuple[str, str, int]],
    chord_duration: float = 2.0,
    repeats: int = 1,
    sr: int = SR,
    octave: int = 4,
    noise: float = 0.002,
    seed: int = 0,
) -> np.ndarray:
    """Render a chord sequence back to back, optionally repeated.

    Entries are ``(root, quality)``, or ``(root, quality, bass_interval)`` to
    render that chord as an inversion.
    """
    blocks = [
        render_chord(
            chord[0],
            chord[1],
            chord_duration,
            sr=sr,
            octave=octave,
            bass_interval=chord[2] if len(chord) > 2 else None,
        )
        for chord in chords
    ] * repeats
    audio = np.concatenate(blocks) if blocks else np.zeros(0)

    if noise > 0 and audio.size:
        audio = audio + np.random.default_rng(seed).normal(0.0, noise, audio.size)

    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


def render_scale(
    tonic: str, mode: str = "major", note_duration: float = 0.4, sr: int = SR, octave: int = 4
) -> np.ndarray:
    """An ascending-then-descending scale — a melodic-only key-detection case."""
    steps = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    root_midi = 12 * (octave + 1) + note_name_to_pc(tonic)
    degrees = [*steps, 12, *reversed(steps)]

    audio = np.concatenate([render_note(root_midi + step, note_duration, sr) for step in degrees])
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


def render_silence(duration: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(duration * sr)), dtype=np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> Path:
    """Write a mono 16-bit WAV (readable without ffmpeg)."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


# --- Named progressions used across the suite -------------------------------

POP_LOOP_C = [("C", "maj"), ("G", "maj"), ("A", "min"), ("F", "maj")]  # I V vi IV in C
BLUES_TURNAROUND_C = [("C", "maj"), ("F", "maj"), ("G", "maj"), ("C", "maj")]
MINOR_LOOP_A = [("A", "min"), ("F", "maj"), ("C", "maj"), ("G", "maj")]
JAZZ_TURNAROUND_C = [("D", "min7"), ("G", "7"), ("C", "maj7")]  # ii7 V7 Imaj7
