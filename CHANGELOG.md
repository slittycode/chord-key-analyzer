# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-07-28

First release: offline key, chord and progression analysis for audio files and
yt-dlp-supported URLs, as a `cka` CLI, a local web UI, and a Python API.

### Added

- `cka analyze` — key with confidence, alternatives and modulations; a chord
  timeline over a 77-state vocabulary (triads and sevenths, plus no-chord);
  Roman-numeral progression with repeated-loop detection. Rich terminal report,
  versioned JSON (`"schema": 1`) and MIREX `.lab` export.
- `cka web` — a single static vanilla-JS page on FastAPI, no build step, calling
  the same pipeline as the CLI.
- `cka eval` — accuracy scoring against reference annotations via `mir_eval`,
  reporting the MIREX chord metrics (`root`, `majmin`, `sevenths`, `mirex`,
  `seg`) and a weighted key score. Corpus chord scores are duration-weighted.
- `[eval]` extra for `mir_eval`, alongside the existing `[url]` and `[web]`.
- `cka web --no-urls` to disable URL ingestion.
- Pre-download URL screening: non-http(s) schemes, and hosts that are or resolve
  to private, loopback, link-local, multicast, reserved or unspecified
  addresses, are refused. Best-effort, since yt-dlp re-resolves on fetch.
- A 200 MB cap on URL downloads.

### Changed

- URL ingestion in the web UI is now enabled only for loopback binds. Binding to
  a non-loopback address disables it unless you build the app yourself.

### Fixed

- URL analysis no longer leaks a temporary directory per run; the download is
  removed once the audio is decoded, including when decoding fails.
- `cka web` with uvicorn installed but FastAPI missing printed the install hint
  instead of crashing on unresolved FastAPI names.
- Requests rejected by `POST /analyze` no longer consume a slot in the bounded
  job store, where they could evict a finished result still being polled for.
- The modulation scan damps chord evidence by each window's own chord variety,
  matching the global key path; a single sustained chord no longer carries a
  20 s window as strongly as a full progression would.
- A `--start` past the end of a file reports "past the end" on the ffmpeg
  decode path too, rather than ffmpeg's vague "produced no audio".
- An unwritable `--json`/`--lab` path is a clean error instead of a traceback.

[Unreleased]: https://github.com/slittycode/chord-key-analyzer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/slittycode/chord-key-analyzer/releases/tag/v0.1.0
