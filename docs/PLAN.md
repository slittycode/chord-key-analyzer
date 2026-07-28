# chord-key-analyzer — Initial Build Plan

## Context

`slittycode/chord-key-analyzer` is an empty repo (zero commits). The goal is a **lean, fully offline, LLM-API-key-free** music analyzer inspired by `slittycode/ableton-sonic-analyzer` (ASA) — but deliberately *not* a port of it. ASA is a three-layer FastAPI + React + Essentia + Gemini stack; this project keeps only its best idea (deterministic DSP measurements are authoritative) and drops everything that made it heavy: no web stack, no Gemini/cloud interpretation, no stem separation, no Essentia (notoriously painful to install).

What it does: given a song — a local audio file or a URL from yt-dlp-supported sites (YouTube, SoundCloud, Bandcamp) — report:
1. **Key** of the track (tonic + mode, confidence, alternatives, modulations),
2. **Chords** on a timeline (label + start/end + confidence),
3. **Progression** (Roman-numeral analysis relative to the detected key, repeated-loop summary).

Genre-general: rock, classical, jazz, electronic, etc. — not tuned only to EDM.

### Decisions (confirmed with the user)
- **Engine**: hybrid — DSP core (numpy/librosa) as the always-works default, with a *pluggable engine interface* so a local neural backend (e.g. crema) can be added later as an optional extra. No heavy ML deps in the base install.
- **Interface**: CLI (`cka analyze <file-or-url>`) **plus a minimal local web UI** (`cka web`): drag-drop a file or paste a URL, see the key card and a chord timeline. Kept lean: one static vanilla-JS/HTML page served by a small local FastAPI app — **no React/Vite/node toolchain**, unlike ASA.
- **Sources**: local files (anything ffmpeg decodes) + yt-dlp URLs. **Spotify / Apple Music streams are out of scope**: they are DRM-protected and cannot be decoded offline without circumvention. Document this clearly in the README; DRM-free purchased files (Bandcamp downloads, iTunes-store DRM-free m4a) work as local files.
- **Chord vocabulary**: triads + sevenths — maj, min, dim, aug, maj7, min7, 7 (dom7), plus an explicit `N` (no-chord) state. Richer vocab (sus/6ths/inversions) deferred; it hurts accuracy more than it helps at this stage.

## Tech stack

- **Python ≥ 3.11**, packaged with `pyproject.toml` (hatchling or setuptools), installable via `pip`/`uv`. Package name `chord_key_analyzer`, CLI entry point **`cka`**.
- **Core deps (kept minimal)**: `numpy`, `librosa` (chroma/CQT, beat tracking, tuning estimation; pulls scipy/soundfile/numba — accepted as the pragmatic floor for correct DSP), `rich` (terminal output), `click` (CLI).
- **Optional extras**:
  - `cka[url]` → `yt-dlp` (URL ingestion)
  - `cka[web]` → `fastapi` + `uvicorn` (local web UI; lazy-imported, clear error if `cka web` is run without it)
  - `cka[deep]` → reserved for a future neural chord backend (e.g. `crema`); **not implemented in v1**, but the engine interface must make it a drop-in.
- **Runtime requirement**: `ffmpeg` on PATH (decoding mp3/m4a/etc. and as yt-dlp's post-processor). Detect and fail with a friendly install hint.
- **No network calls ever** except when the user explicitly passes a URL to fetch.

## Architecture / module layout

```
chord_key_analyzer/
  __init__.py
  cli.py            # click CLI: `cka analyze`, `cka --version`
  ingest.py         # local file decode (ffmpeg→mono float32 @22050Hz), URL fetch via yt-dlp (lazy import)
  features.py       # tuning estimation, CQT chroma, beat tracking, beat-synchronous chroma aggregation
  key.py            # Krumhansl-Schmuckler + Temperley profile correlation; global key, alternatives, windowed modulation scan
  chords.py         # ChordEngine protocol; TemplateHMMEngine: chord templates, log-likelihood scoring, Viterbi smoothing, segment merge
  progression.py    # chords → Roman numerals in detected key; loop/repeat detection; progression summary
  output.py         # rich terminal rendering, JSON export, MIREX .lab export
  models.py         # dataclasses: AnalysisResult, KeyEstimate, ChordSegment, ProgressionSummary
  web.py            # `cka web`: local FastAPI app — POST /analyze (file upload or URL), GET / serves static page
  web_static/
    index.html      # single vanilla-JS page: drag-drop/URL input, key card, SVG chord timeline, JSON download
tests/
  fixtures.py       # synthesize WAVs with known chords/keys (numpy-rendered triads/7ths over a progression)
  test_key.py test_chords.py test_progression.py test_ingest.py test_cli.py
.github/workflows/ci.yml   # lint (ruff) + pytest on push/PR
```

### Analysis pipeline (TemplateHMMEngine, the v1 default)
1. **Decode** to mono float32 at 22050 Hz via ffmpeg (subprocess piping to raw PCM — avoids audioread flakiness).
2. **Tuning correction** (`librosa.estimate_tuning`) so slightly-detuned recordings don't smear chroma bins.
3. **Chroma**: `librosa.feature.chroma_cqt` (36 bins/octave folded to 12), median-filtered.
4. **Beat-sync**: `librosa.beat.beat_track`; average chroma per beat (fallback to fixed ~0.5 s windows when beat tracking is unreliable, e.g. rubato classical — decide via beat-strength confidence).
5. **Key**: correlate the global average chroma against all 24 rotated Krumhansl-Schmuckler *and* Temperley profiles; report best key, confidence (correlation margin over runner-up), top-3 alternatives. **Modulation scan**: same correlation over a sliding window (~20 s, 5 s hop); report segments where the winning key changes stably.
6. **Chords**: 12 roots × 7 qualities + N = 85 states. Score each beat's chroma against L1-normalized binary-with-harmonic-weighting templates (log domain); **Viterbi decode** with a self-transition bonus (sticky chords) to suppress frame-level flicker; merge adjacent identical labels into `ChordSegment(start, end, label, confidence)`.
7. **Progression**: map segments to Roman numerals in the detected key (borrowed chords labeled literally, e.g. `bVII`); collapse consecutive repeats; find the dominant repeating loop (e.g. `I–V–vi–IV ×12`) via n-gram counting over the segment sequence.

### Engine interface (the "hybrid" seam)
`ChordEngine` protocol: `analyze(audio: np.ndarray, sr: int) -> list[ChordSegment]`. v1 ships only `TemplateHMMEngine`; CLI flag `--engine template` (default) reserves the namespace so `--engine crema` can arrive later without breaking the CLI or JSON schema.

## CLI surface (v1)

```
cka analyze SONG            # file path or URL; pretty rich output: key card + chord timeline + progression
  --json PATH | --json -    # machine-readable result
  --lab PATH                # MIREX-style chord annotation (start\tend\tlabel)
  --start SEC --duration SEC
  --triads-only             # restrict vocabulary to maj/min/dim/aug
  --engine template
cka web                     # launch the local web UI (default http://127.0.0.1:8321); requires cka[web]
  --port PORT --host HOST
```

### Web UI (minimal, local-only)
- `cka web` starts uvicorn bound to 127.0.0.1 only (never 0.0.0.0 by default — this is a local tool).
- One static page (`web_static/index.html`, vanilla JS + inline CSS, no build step): drop an audio file or paste a URL → `POST /analyze` (multipart upload or `{"url": ...}`) → renders the same JSON the CLI emits as a key card, an SVG chord timeline with time axis, the Roman-numeral progression, and a "download JSON / .lab" link.
- The endpoint calls the exact same pipeline function the CLI uses (`analyze_audio(...) -> AnalysisResult`) — one code path, two frontends. Long analyses report progress via simple polling (`GET /jobs/{id}`); keep it to an in-memory job dict, no database, no queue.

JSON schema (stable, versioned via `"schema": 1`): `{file, duration, key: {tonic, mode, confidence, alternatives[], modulations[]}, chords: [{start, end, label, confidence}], progression: {roman[], main_loop{labels, roman, repeats}}, meta: {engine, version}}`.

## Milestones (implement in order; each ends green on CI)

1. **Scaffold**: pyproject, package skeleton, click CLI stub, ruff + pytest, GitHub Actions CI, README (with the source-support matrix incl. the DRM explanation), MIT license.
2. **Ingest**: ffmpeg decode + probe (duration), ffmpeg-missing detection; yt-dlp URL fetch behind `[url]` extra with lazy import and a clear error when absent.
3. **Features**: tuning, chroma, beat-sync with rubato fallback.
4. **Key detection** + tests against synthesized fixtures (render scales/chord beds in known keys; assert detected key and that relative-major/minor confusion is within tolerance).
5. **Chord detection** (templates + Viterbi) + fixture tests (render `C F G C`, `Am F C G`, a 7th-chord jazz turnaround; assert segment labels and boundary tolerance ±0.25 s).
6. **Progression analysis** + tests.
7. **Output layer**: rich rendering, JSON, .lab; snapshot-test the JSON.
8. **Web UI**: `web.py` + `web_static/index.html` behind `[web]`; FastAPI TestClient tests for `POST /analyze` (fixture WAV upload → expected JSON) and the missing-extra error path.
9. **End-to-end + docs**: `cka analyze` on a fixture WAV in CI; README usage examples (CLI + web); accuracy-expectations section (triadic rock/pop strong; dense jazz voicings and fast classical harmony will be approximate); optional offline eval script that scores against Isophonics/Beatles `.lab` annotations *if the user has downloaded them locally* (never fetched automatically).

Explicitly deferred (Phase 2 candidates, not in this build): neural engine (`crema`) behind `[deep]`, inversion/bass detection, section-aware segmentation, waveform rendering in the web UI.

## Hand-off notes for the implementing session

This document is the source of truth for the build. Work through the milestones in order — each one should land with its tests green on CI before starting the next. The four "Decisions" above were confirmed with the user; don't relitigate them, but do surface anything discovered mid-build that would change one.

## Verification

- **Unit/integration**: `pytest` — all fixture-based tests above; fixtures are synthesized in-test with numpy (no audio files committed, no network).
- **End-to-end**: `cka analyze tests/out/fixture.wav --json -` returns the known key (`C major`) and the rendered progression; run in CI. Web path: FastAPI TestClient posts the same fixture and asserts the same JSON.
- **Real-world spot checks** (manual, post-build): a Beatport-tagged electronic track (compare detected key to Beatport's), a canonical rock song (e.g. a I–V–vi–IV pop track), and a classical piece for the rubato fallback path.
