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


@dataclass
class Features:
    """Chroma and timing information for one track."""

    chroma: np.ndarray  # (12, n_frames), each frame L2-normalised
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


def compute_chroma(
    y: np.ndarray,
    sr: int,
    hop_length: int = DEFAULT_HOP,
    tuning: float = 0.0,
    harmonic: bool = True,
    median_width: int = 9,
) -> np.ndarray:
    """CQT chroma at 36 bins/octave, optionally from the harmonic component only.

    Percussion spreads energy across every pitch class at once; running HPSS
    first is the single biggest chroma-quality win on real drum-heavy material.
    """
    import librosa

    source = y
    if harmonic and y.size > hop_length * 4:
        try:
            source = librosa.effects.harmonic(y, margin=3.0)
        except Exception:
            source = y

    chroma = librosa.feature.chroma_cqt(
        y=source,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=36,
        tuning=tuning,
    )

    if median_width > 1 and chroma.shape[1] > median_width:
        from scipy.ndimage import median_filter

        chroma = median_filter(chroma, size=(1, median_width), mode="nearest")

    return _normalise_columns(np.asarray(chroma, dtype=np.float64))


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
    chroma = compute_chroma(y, sr, hop_length=hop_length, tuning=tuning, harmonic=harmonic)
    n_frames = chroma.shape[1]

    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    rms = frame_rms(y, hop_length, n_frames)

    loud_reference = float(np.percentile(rms, 95)) if rms.size else 0.0
    silent = rms < max(loud_reference * SILENCE_RATIO, 1e-6)

    tempo, beat_times, beats_reliable = track_beats(y, sr, hop_length=hop_length)

    return Features(
        chroma=chroma,
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
