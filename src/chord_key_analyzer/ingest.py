"""Audio ingestion: local files (soundfile / ffmpeg) and URLs (yt-dlp).

Decoding strategy, in order:

1. ``soundfile`` for containers libsndfile understands (wav, flac, ogg, and mp3
   on modern libsndfile builds).  This keeps the common test/fixture path free
   of any external binary.
2. ``ffmpeg`` piped to raw ``f32le`` PCM for everything else.  Piping avoids
   ``audioread``'s backend roulette and gives us exact control over the sample
   rate and channel count.

Everything is returned as mono float32 at :data:`TARGET_SR`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SR = 22050

URL_SCHEMES = ("http://", "https://")


class IngestError(RuntimeError):
    """Raised when audio cannot be obtained or decoded."""


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded audio plus the human-readable name of where it came from."""

    samples: np.ndarray
    sr: int
    source: str

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sr)


def is_url(target: str) -> bool:
    return target.startswith(URL_SCHEMES)


def _require_ffmpeg(reason: str) -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise IngestError(
            f"ffmpeg is required {reason} but was not found on PATH.\n"
            "Install it with one of:\n"
            "  macOS:         brew install ffmpeg\n"
            "  Debian/Ubuntu: sudo apt install ffmpeg\n"
            "  Windows:       winget install Gyan.FFmpeg"
        )
    return path


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _decode_with_soundfile(
    path: Path, sr: int, offset: float, duration: float | None
) -> np.ndarray | None:
    """Return decoded mono audio, or ``None`` if libsndfile cannot read the file."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception:
        return None

    start_frame = int(round(offset * info.samplerate))
    if start_frame >= info.frames and info.frames > 0:
        raise IngestError(
            f"--start {offset:g}s is past the end of the file ({info.duration:.2f}s)."
        )
    frames = -1 if duration is None else int(round(duration * info.samplerate))

    try:
        with sf.SoundFile(str(path)) as handle:
            handle.seek(start_frame)
            samples = handle.read(frames=frames, dtype="float32", always_2d=False)
    except Exception:
        return None

    samples = _to_mono(np.asarray(samples))
    if info.samplerate != sr:
        import librosa

        samples = librosa.resample(samples, orig_sr=info.samplerate, target_sr=sr)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _decode_with_ffmpeg(
    path: Path, sr: int, offset: float, duration: float | None
) -> np.ndarray:
    ffmpeg = _require_ffmpeg(f"to decode {path.name}")
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error"]
    if offset > 0:
        cmd += ["-ss", f"{offset:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {proc.returncode}"
        raise IngestError(f"ffmpeg failed to decode {path.name}: {tail}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise IngestError(
            f"ffmpeg produced no audio for {path.name} (empty or unsupported stream)."
        )
    return np.ascontiguousarray(samples, dtype=np.float32)


def load_audio_file(
    path: str | Path,
    sr: int = TARGET_SR,
    offset: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    """Decode a local audio file to mono float32 at ``sr``."""
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path}")
    if path.is_dir():
        raise IngestError(f"Expected an audio file but got a directory: {path}")

    samples = _decode_with_soundfile(path, sr, offset, duration)
    if samples is None:
        samples = _decode_with_ffmpeg(path, sr, offset, duration)

    if samples.size == 0:
        raise IngestError(
            f"No audio decoded from {path.name}. "
            "The file may be empty, or --start/--duration may select an empty range."
        )
    # Guard against NaN/inf from damaged files so downstream DSP stays finite.
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


def probe_duration(path: str | Path) -> float | None:
    """Best-effort container duration in seconds without a full decode."""
    path = Path(path)
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def download_url(url: str, dest_dir: str | Path | None = None) -> Path:
    """Fetch ``url`` with yt-dlp and return the path to the downloaded audio.

    yt-dlp is an optional dependency (``pip install 'chord-key-analyzer[url]'``)
    and is imported lazily so the base install never pays for it.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise IngestError(
            "URL input requires yt-dlp, which is not installed.\n"
            "Install it with: pip install 'chord-key-analyzer[url]'"
        ) from exc

    _require_ffmpeg("to extract audio from downloaded media")

    dest = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="cka-"))
    dest.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(dest / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}
        ],
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise IngestError(f"yt-dlp could not download {url}: {exc}") from exc

    if info is None:
        raise IngestError(f"yt-dlp returned no media for {url}")
    if "entries" in info:  # playlist — analyse the first entry
        entries = [e for e in info["entries"] if e]
        if not entries:
            raise IngestError(f"yt-dlp found no playable entries at {url}")
        info = entries[0]

    downloaded = sorted(p for p in dest.iterdir() if p.is_file())
    if not downloaded:
        raise IngestError(f"yt-dlp reported success but produced no file for {url}")
    # The post-processed wav is what we want when both it and the source remain.
    for candidate in downloaded:
        if candidate.suffix.lower() == ".wav":
            return candidate
    return downloaded[0]


def resolve_source(
    target: str,
    sr: int = TARGET_SR,
    offset: float = 0.0,
    duration: float | None = None,
) -> LoadedAudio:
    """Load ``target`` — a local path or a yt-dlp-supported URL — into memory."""
    if is_url(target):
        path = download_url(target)
        samples = load_audio_file(path, sr=sr, offset=offset, duration=duration)
        return LoadedAudio(samples=samples, sr=sr, source=target)

    samples = load_audio_file(target, sr=sr, offset=offset, duration=duration)
    return LoadedAudio(samples=samples, sr=sr, source=str(target))
