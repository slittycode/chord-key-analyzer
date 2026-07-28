"""Ingestion tests: decoding, probing, and URL handling."""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import fixtures as fx
from chord_key_analyzer import ingest


def test_loads_a_wav_without_ffmpeg(pop_wav):
    """WAV goes through libsndfile, so the common path needs no external binary."""
    audio = ingest.load_audio_file(pop_wav)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) / ingest.TARGET_SR == pytest.approx(24.0, abs=0.1)


def test_resamples_to_the_target_rate(tmp_path):
    audio = fx.render_progression(fx.POP_LOOP_C, chord_duration=1.0, repeats=1)
    path = fx.write_wav(tmp_path / "hi.wav", audio, sr=fx.SR)
    loaded = ingest.load_audio_file(path, sr=11025)
    assert len(loaded) == pytest.approx(len(audio) / 2, rel=0.02)


def test_start_and_duration_select_a_window(pop_wav):
    excerpt = ingest.load_audio_file(pop_wav, offset=4.0, duration=2.0)
    assert len(excerpt) / ingest.TARGET_SR == pytest.approx(2.0, abs=0.05)


def test_duration_past_the_end_is_clamped(pop_wav):
    excerpt = ingest.load_audio_file(pop_wav, offset=22.0, duration=60.0)
    assert 0 < len(excerpt) / ingest.TARGET_SR <= 3.0


def test_start_past_the_end_is_an_error(pop_wav):
    with pytest.raises(ingest.IngestError, match="past the end"):
        ingest.load_audio_file(pop_wav, offset=999.0)


def test_start_past_the_end_is_an_error_on_the_ffmpeg_path(monkeypatch, tmp_path, pop_wav):
    """Files libsndfile declines must give the same message, not ffmpeg's vague one."""
    monkeypatch.setattr(ingest, "_decode_with_soundfile", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "probe_duration", lambda path: 10.0)
    monkeypatch.setattr(
        ingest,
        "_decode_with_ffmpeg",
        lambda *args, **kwargs: pytest.fail("ffmpeg must not be reached for an out-of-range start"),
    )

    with pytest.raises(ingest.IngestError, match="past the end"):
        ingest.load_audio_file(pop_wav, offset=999.0)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ingest.IngestError, match="not found"):
        ingest.load_audio_file(tmp_path / "nope.wav")


def test_directory_is_an_error(tmp_path):
    with pytest.raises(ingest.IngestError, match="directory"):
        ingest.load_audio_file(tmp_path)


def test_stereo_is_mixed_to_mono(tmp_path):
    import soundfile as sf

    stereo = np.stack(
        [
            fx.render_progression(fx.POP_LOOP_C, 0.5, 1),
            fx.render_progression(fx.MINOR_LOOP_A, 0.5, 1),
        ],
        axis=1,
    )
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, fx.SR)
    assert ingest.load_audio_file(path).ndim == 1


def test_probe_duration_of_a_wav(pop_wav):
    assert ingest.probe_duration(pop_wav) == pytest.approx(24.0, abs=0.1)


def test_probe_duration_of_a_missing_file(tmp_path):
    assert ingest.probe_duration(tmp_path / "nope.wav") is None


def test_url_detection():
    assert ingest.is_url("https://example.com/song")
    assert ingest.is_url("http://example.com/song")
    assert not ingest.is_url("/home/user/song.mp3")
    assert not ingest.is_url("song.wav")


def test_resolve_source_reads_local_files(pop_wav):
    loaded = ingest.resolve_source(str(pop_wav))
    assert loaded.sr == ingest.TARGET_SR
    assert loaded.duration == pytest.approx(24.0, abs=0.1)
    assert loaded.source == str(pop_wav)


def test_missing_ytdlp_gives_an_install_hint(monkeypatch, tmp_path):
    """The [url] extra is optional, so its absence must be explained, not raised
    as a bare ImportError."""
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    with pytest.raises(ingest.IngestError, match=r"chord-key-analyzer\[url\]"):
        ingest.download_url("https://example.com/song", dest_dir=tmp_path)


def test_url_download_returns_the_extracted_wav(monkeypatch, tmp_path):
    """Exercise the yt-dlp path with a stub — no network, no real download."""

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            destination = tmp_path / "song.wav"
            fx.write_wav(destination, fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
            return {"id": "song", "title": "Song"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    monkeypatch.setattr(ingest, "_require_ffmpeg", lambda reason: "/usr/bin/ffmpeg")

    result = ingest.download_url("https://example.com/song", dest_dir=tmp_path)
    assert result.suffix == ".wav"


def test_url_download_caps_the_file_size(monkeypatch, tmp_path):
    """An unbounded download is a denial-of-service on the analyzing machine."""
    captured: dict[str, object] = {}

    class CapturingYoutubeDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            fx.write_wav(tmp_path / "song.wav", fx.render_progression(fx.POP_LOOP_C, 0.5, 1))
            return {"id": "song", "title": "Song"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=CapturingYoutubeDL))
    monkeypatch.setattr(ingest, "_require_ffmpeg", lambda reason: "/usr/bin/ffmpeg")

    ingest.download_url("https://example.com/song", dest_dir=tmp_path)
    assert captured["max_filesize"] == ingest.MAX_DOWNLOAD_BYTES


def test_url_download_reports_yt_dlp_failures(monkeypatch, tmp_path):
    class FailingYoutubeDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            raise RuntimeError("video unavailable")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FailingYoutubeDL))
    monkeypatch.setattr(ingest, "_require_ffmpeg", lambda reason: "/usr/bin/ffmpeg")

    with pytest.raises(ingest.IngestError, match="video unavailable"):
        ingest.download_url("https://example.com/song", dest_dir=tmp_path)


def _fake_ytdl_recording_its_dest(monkeypatch, *, write_audio: bool) -> list[Path]:
    """Install a yt-dlp stub that records the directory yt-dlp was told to use.

    ``downloaded_media`` makes up that temporary directory itself and hands it to
    ``download_url``, so reading it back out of ``outtmpl`` is the only way a
    test can then assert the directory was cleaned up.
    """
    recorded: list[Path] = []

    class RecordingYoutubeDL:
        def __init__(self, options):
            recorded.append(Path(options["outtmpl"]).parent)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            if write_audio:
                audio = fx.render_progression(fx.POP_LOOP_C, 0.5, 1)
                fx.write_wav(recorded[-1] / "song.wav", audio)
            return {"id": "song", "title": "Song"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=RecordingYoutubeDL))
    monkeypatch.setattr(ingest, "_require_ffmpeg", lambda reason: "/usr/bin/ffmpeg")
    return recorded


def test_resolve_source_cleans_up_the_download_dir(monkeypatch):
    """A URL analysis must not leave its decoded download behind."""
    recorded = _fake_ytdl_recording_its_dest(monkeypatch, write_audio=True)

    loaded = ingest.resolve_source("https://example.com/song")

    assert loaded.duration > 0
    assert len(recorded) == 1
    assert not recorded[0].exists(), f"download dir {recorded[0]} was left behind"


def test_resolve_source_cleans_up_when_decode_fails(monkeypatch):
    """Cleanup has to survive the failure path, not just the happy one."""
    recorded = _fake_ytdl_recording_its_dest(monkeypatch, write_audio=False)

    with pytest.raises(ingest.IngestError):
        ingest.resolve_source("https://example.com/song")

    assert len(recorded) == 1
    assert not recorded[0].exists(), f"download dir {recorded[0]} was left behind"


def test_missing_ffmpeg_message_names_the_installers(monkeypatch):
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    with pytest.raises(ingest.IngestError) as excinfo:
        ingest._require_ffmpeg("to test")
    message = str(excinfo.value)
    assert "brew install ffmpeg" in message
    assert "apt install ffmpeg" in message


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_decodes_a_compressed_file(tmp_path, pop_wav):
    """Formats libsndfile cannot read must fall through to the ffmpeg pipe."""
    import subprocess

    mp3 = tmp_path / "pop.mp3"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(pop_wav), str(mp3)],
        check=True,
        capture_output=True,
    )
    audio = ingest._decode_with_ffmpeg(mp3, ingest.TARGET_SR, 0.0, None)
    assert audio.dtype == np.float32
    assert len(audio) / ingest.TARGET_SR == pytest.approx(24.0, abs=0.3)


def test_ffmpeg_reports_a_bad_file(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"definitely not audio")
    with pytest.raises(ingest.IngestError):
        ingest._decode_with_ffmpeg(junk, ingest.TARGET_SR, 0.0, None)
