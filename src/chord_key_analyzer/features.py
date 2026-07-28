"""Feature extraction: tuning correction, chroma, beats.

The chord and key stages both consume the same :class:`Features` bundle, so a
track is only ever transformed once per analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: ~93 ms per frame at 22050 Hz.  Fine enough that chord boundaries land well
#: inside the +/-0.25 s tolerance we hold ourselves to, coarse enough that a
#: five-minute track stays in the low thousands of frames.
DEFAULT_HOP = 2048

#: Frames whose RMS falls below this fraction of the track's 95th-percentile RMS
#: are treated as silence and forced to the no-chord state.
SILENCE_RATIO = 0.02

#: The bass chroma covers three octaves up from C1 — the register a bass guitar
#: or a pianist's left hand actually occupies, stopping below where the chord
#: voicing itself sits.  Sharing the main chroma's hop and bins/octave is what
#: makes the two frame-aligned with no resampling in between.
BASS_FMIN_NOTE = "C1"
BASS_OCTAVES = 3


@dataclass
class Features:
    """Chroma and timing information for one track."""

    chroma: np.ndarray  # (12, n_frames), each frame L2-normalised
    bass_chroma: np.ndarray  # (12, n_frames), low register, deliberately unnormalised
    times: np.ndarray  # (n_frames,) frame start times in seconds
    rms: np.ndarray  # (n_frames,) per-frame RMS of the source audio
    silent: np.ndarray  # (n_frames,) bool mask of near-silent frames
    sr: int
    hop_length: int
    tuning: float
    tempo: float | None
    beat_times: np.ndarray
    beats_reliable: bool

    @property
    def n_frames(self) -> int:
        return self.chroma.shape[1]

    @property
    def frame_duration(self) -> float:
        return self.hop_length / float(self.sr)


def estimate_tuning(y: np.ndarray, sr: int) -> float:
    """Deviation from A440 in fractions of a semitone, clamped to a sane range."""
    import librosa

    if y.size < sr // 4:
        return 0.0
    try:
        tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        return 0.0
    if not np.isfinite(tuning) or abs(tuning) > 0.5:
        return 0.0
    return tuning


def _normalise_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return matrix / norms


def harmonic_component(y: np.ndarray, hop_length: int = DEFAULT_HOP) -> np.ndarray:
    """The harmonic part of ``y``, or ``y`` unchanged when HPSS cannot run.

    Percussion spreads energy across every pitch class at once; separating it out
    first is the single biggest chroma-quality win on real drum-heavy material.

    Hoisted out of :func:`compute_chroma` so one separation can feed both the
    main and the bass chroma.  It is the most expensive step in the front-end,
    and running it twice would also let the two chromas disagree about what the
    harmonic part of the track even is.
    """
    if y.size <= hop_length * 4:
        return y

    import librosa

    try:
        return librosa.effects.harmonic(y, margin=3.0)
    except Exception:
        return y


def _median_smooth(chroma: np.ndarray, median_width: int) -> np.ndarray:
    if median_width > 1 and chroma.shape[1] > median_width:
        from scipy.ndimage import median_filter

        chroma = median_filter(chroma, size=(1, median_width), mode="nearest")
    return chroma


def compute_chroma(
    y: np.ndarray,
    sr: int,
    hop_length: int = DEFAULT_HOP,
    tuning: float = 0.0,
    harmonic: bool = True,
    median_width: int = 9,
) -> np.ndarray:
    """CQT chroma at 36 bins/octave, optionally from the harmonic component only.

    ``harmonic=True`` runs the separation here.  :func:`extract_features`
    separates once and passes the result in with ``harmonic=False`` instead, so
    the two chromas share one HPSS pass; the flag stays for callers that reach
    this function directly.
    """
    import librosa

    source = harmonic_component(y, hop_length) if harmonic else y

    chroma = librosa.feature.chroma_cqt(
        y=source,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=36,
        tuning=tuning,
    )

    chroma = _median_smooth(np.asarray(chroma, dtype=np.float64), median_width)
    return _normalise_columns(chroma)


def compute_bass_chroma(
    y: np.ndarray,
    sr: int,
    hop_length: int = DEFAULT_HOP,
    tuning: float = 0.0,
    harmonic: bool = True,
    median_width: int = 9,
) -> np.ndarray:
    """Chroma of the low register only, and deliberately **not** normalised.

    Every other chroma here is column-normalised, which is right when only the
    shape of a frame matters.  It is wrong here: the inversion detector asks
    whether one pitch class *dominates* a frame's bass energy, and unit-norming
    each column erases precisely the difference between a clearly voiced bass
    note and a murky low end with nothing in particular going on.
    """
    import librosa

    source = harmonic_component(y, hop_length) if harmonic else y

    chroma = librosa.feature.chroma_cqt(
        y=source,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.note_to_hz(BASS_FMIN_NOTE),
        n_octaves=BASS_OCTAVES,
        bins_per_octave=36,
        tuning=tuning,
    )

    return _median_smooth(np.asarray(chroma, dtype=np.float64), median_width)


def _match_frame_count(matrix: np.ndarray, n_frames: int) -> np.ndarray:
    """Force ``matrix`` to exactly ``n_frames`` columns.

    The bass CQT shares the main chroma's hop, so in practice the two already
    agree; this is belt and braces against a librosa release rounding the
    shorter filter bank's frame count differently.
    """
    if matrix.shape[1] == n_frames:
        return matrix
    if matrix.shape[1] > n_frames:
        return matrix[:, :n_frames]
    return np.pad(matrix, ((0, 0), (0, n_frames - matrix.shape[1])), mode="edge")


def track_beats(
    y: np.ndarray, sr: int, hop_length: int = DEFAULT_HOP
) -> tuple[float | None, np.ndarray, bool]:
    """Estimate tempo and beat times, plus whether the result looks trustworthy.

    Rubato classical and free-time material produce beat grids that are worse
    than no grid at all, so we report reliability and let callers opt out.
    """
    import librosa

    if y.size < sr * 2:
        return None, np.array([], dtype=float), False

    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames"
        )
    except Exception:
        return None, np.array([], dtype=float), False

    tempo_value = float(np.atleast_1d(tempo)[0]) if tempo is not None else 0.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beat_times = np.asarray(beat_times, dtype=float)

    reliable = False
    if beat_times.size >= 8 and 40.0 <= tempo_value <= 220.0:
        intervals = np.diff(beat_times)
        if intervals.size:
            median_interval = float(np.median(intervals))
            # A steady pulse has tightly clustered inter-beat intervals; rubato
            # and mis-tracked material do not.
            spread = float(np.median(np.abs(intervals - median_interval)))
            reliable = median_interval > 0 and (spread / median_interval) < 0.10

    return (tempo_value or None), beat_times, reliable


def frame_rms(y: np.ndarray, hop_length: int, n_frames: int) -> np.ndarray:
    """Per-frame RMS aligned to the chroma frames."""
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
    if rms.size < n_frames:
        rms = np.pad(rms, (0, n_frames - rms.size), mode="edge")
    return np.asarray(rms[:n_frames], dtype=float)


def extract_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = DEFAULT_HOP,
    harmonic: bool = True,
) -> Features:
    """Run the whole feature front-end over one track."""
    import librosa

    tuning = estimate_tuning(y, sr)

    # One separation, both chromas: see harmonic_component().
    source = harmonic_component(y, hop_length) if harmonic else y
    chroma = compute_chroma(source, sr, hop_length=hop_length, tuning=tuning, harmonic=False)
    n_frames = chroma.shape[1]
    bass_chroma = _match_frame_count(
        compute_bass_chroma(source, sr, hop_length=hop_length, tuning=tuning, harmonic=False),
        n_frames,
    )

    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    rms = frame_rms(y, hop_length, n_frames)

    loud_reference = float(np.percentile(rms, 95)) if rms.size else 0.0
    silent = rms < max(loud_reference * SILENCE_RATIO, 1e-6)

    tempo, beat_times, beats_reliable = track_beats(y, sr, hop_length=hop_length)

    return Features(
        chroma=chroma,
        bass_chroma=bass_chroma,
        times=np.asarray(times, dtype=float),
        rms=rms,
        silent=silent,
        sr=sr,
        hop_length=hop_length,
        tuning=tuning,
        tempo=tempo,
        beat_times=beat_times,
        beats_reliable=beats_reliable,
    )
