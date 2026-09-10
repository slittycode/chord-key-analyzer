# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `docs/ROADMAP.md` no longer claims the repo has no issue tracker — GitHub
  Issues is enabled; known defects are kept here on purpose, not because
  there's nowhere else to put them.
- `output.py`'s three copies of "write to stdout, or write to a path" —
  `write_json`, `write_lab`, and `evaluate.py`'s `write_report_json` — are now
  one `_write_text()` helper.

### Fixed

- `_label_spans` no longer mints a fresh section letter for every silent
  span. A silent span's average chroma is the zero vector, which
  cosine-matches nothing — not even another silent span — without a
  dedicated path for it; two quiet stretches in the same track were
  labelled differently (e.g. `B` and `C`) instead of matching. Was
  documented as a known defect in `docs/ROADMAP.md`.

## [0.2.0] — 2026-07-29

Bass awareness, structural sections, and a CI leg that proves the core install
really is standalone.

### Added

- Bass detection over a dedicated low-register chroma. Where one pitch class
  clearly dominates a segment's low end and is a chord tone other than the root,
  it is reported: `chords[].bass` in JSON (a note name), a degree slash in the
  `.lab` export (`C:maj/3` — the spelling mir_eval parses), and `C:maj/E` in the
  terminal and the browser. The `label` field itself stays plain.
- `maj6` as a re-spelling. `C:maj6` and `A:min7` are the same four notes, so only
  `min7` is a decoder state; a detected third in the bass re-spells it as the
  sixth on that bass, in root position.
- Structural sections on tracks over 30 seconds, via Foote checkerboard novelty
  over a self-similarity matrix. Each section carries a local key hint and its own
  progression, and is labelled `A`/`B`/`A'` — repetition, never "verse"/"chorus".
  Additive `sections` key in the JSON; new `--no-sections` flag.
- A `bare-install` CI job that installs only the core dependencies, checks every
  skip is one of the extras it deliberately left out, and asserts each optional
  feature prints its install hint rather than a traceback.
- `require_eval_extra()`, mirroring `require_web_extra()`.

### Changed

- `estimate_key_from_chroma` takes `chords`/`use_edges` instead of
  `chord_scores`/`chord_weight`; the evidence damping now happens inside it
  rather than at each call site.
- `download_url` requires `dest_dir`. The temporary-directory fallback was dead
  code and a standing invitation to reintroduce a leak.
- `_diatonic_chords` is derived from the scales rather than hand-listed. The
  derived sets reproduce the old tables chord for chord, and a test pins that.
- The JSON schema stays at `1`. All additions are additive, and the README now
  states the contract: consumers must ignore keys they do not recognise.

### Fixed

- One unparseable label in a reference `.lab` no longer ends a whole `cka eval`
  run. mir_eval parses labels when it scores, not when it loads, and raises from
  `Exception` rather than `ValueError` — which is how this escaped the handler.
- `cka eval` without the `[eval]` extra prints the install hint before walking
  the dataset, instead of after listing tracks it cannot score.
- Dataset discovery matches audio and label extensions case-insensitively, so
  `Track.WAV` beside `Track.LAB` is a pair rather than two orphans.
- The web module and README said the endpoint calls `analyze_audio()`; it calls
  `analyze_source()`. The README also claimed the server binds to localhost
  "only" rather than by default, and its `meta` example was missing three keys.

## [0.1.0] — 2026-07-29

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

[Unreleased]: https://github.com/slittycode/chord-key-analyzer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/slittycode/chord-key-analyzer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/slittycode/chord-key-analyzer/releases/tag/v0.1.0
