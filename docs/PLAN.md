# chord-key-analyzer — design plan

This is the plan the project was built from, followed by the decisions that changed
during implementation and why.

## Goal

A lean, fully offline, LLM-API-key-free music analyzer. Given a song — a local audio file
or a URL from a yt-dlp-supported site — report:

1. **Key** (tonic + mode, confidence, alternatives, modulations)
2. **Chords** on a timeline (label + start/end + confidence)
3. **Progression** (Roman numerals relative to the detected key, repeated-loop summary)

Genre-general: rock, classical, jazz, electronic — not tuned only to EDM.

It is inspired by [`ableton-sonic-analyzer`](https://github.com/slittycode/ableton-sonic-analyzer)
(ASA) but is deliberately not a port of it. It keeps ASA's best idea — deterministic DSP
measurements are authoritative — and drops what made ASA heavy: the FastAPI + React web
stack, Gemini/cloud interpretation, stem separation, and Essentia.

## Decisions

- **Engine**: hybrid. A DSP core (numpy/librosa) is the always-works default, behind a
  pluggable engine interface so a local neural backend (e.g. crema) can be added later.
  No heavy ML dependencies in the base install.
- **Interface**: CLI (`cka analyze`) plus a minimal local web UI (`cka web`) — one static
  vanilla-JS page served by a small FastAPI app. No React/Vite/node toolchain.
- **Sources**: local files (anything ffmpeg decodes) + yt-dlp URLs. Spotify and Apple
  Music streams are out of scope: they are DRM-protected and cannot be decoded offline
  without circumvention. DRM-free purchased files work as ordinary local files.
- **Chord vocabulary**: triads + sevenths — maj, min, dim, aug, maj7, min7, 7 — plus an
  explicit `N` (no-chord) state. Richer vocabulary (sus, 6ths, inversions) is deferred;
  at this stage it costs more accuracy than it buys.

## Stack

- Python ≥ 3.11, `pyproject.toml` (hatchling), package `chord_key_analyzer`, CLI `cka`.
- Core deps: `numpy`, `librosa`, `soundfile`, `rich`, `click`.
- Extras: `[url]` → yt-dlp; `[web]` → fastapi + uvicorn; `[deep]` → reserved for a future
  neural backend, empty in v1; `[dev]` → pytest, ruff, httpx.
- ffmpeg on PATH for compressed formats and as yt-dlp's post-processor.
- No network calls except when the user explicitly passes a URL.

## Module layout

```
src/chord_key_analyzer/
  cli.py            click CLI: `cka analyze`, `cka web`
  ingest.py         soundfile/ffmpeg decode, yt-dlp URL fetch (lazy import)
  features.py       tuning, CQT chroma, HPSS, beat tracking
  key.py            KS + Temperley profiles, chord evidence, modulation scan
  chords.py         ChordEngine protocol; TemplateHMMEngine (templates + Viterbi)
  progression.py    Roman numerals, loop detection
  pipeline.py       analyze_audio / analyze_source — the one shared code path
  output.py         rich rendering, JSON export, MIREX .lab export
  models.py         dataclasses + JSON schema
  web.py            FastAPI app: POST /analyze, GET /jobs/{id}
  web_static/index.html
tests/              fixtures.py (numpy-synthesised audio) + per-module tests
```

## Pipeline

1. **Decode** to mono float32 at 22050 Hz.
2. **Tuning correction** (`librosa.estimate_tuning`).
3. **HPSS**, then **CQT chroma** at 36 bins/octave folded to 12, median filtered.
4. **Beat tracking**, with a reliability check for rubato material.
5. **Chords**: 12 roots × 7 qualities + `N`, scored against harmonically-weighted
   templates, Viterbi-decoded with a self-transition bonus, merged into segments.
6. **Key**: KS + Temperley profile correlation over the pooled chroma, combined with
   chord evidence; sliding-window scan for modulations.
7. **Progression**: Roman numerals, collapse repeats, find the dominant loop by n-gram
   coverage.

## Verification

- Fixtures are synthesised in-test with numpy — no audio files committed, no network.
- Key detection is checked across all 12 major and all 12 minor keys.
- Chord decoding is checked on pop, minor, blues, jazz-seventh, and dim/aug progressions,
  with a ±0.25 s boundary tolerance.
- The web path posts the same fixture through `TestClient` and must agree with the CLI.
- CI runs ruff + pytest on push and PR.

---

## Changes made during implementation

Everything below is a deliberate departure from the plan above, made because the
implementation surfaced something the plan had not accounted for.

### 1. Chord decoding is frame-level, not beat-synchronous

The plan called for beat-synchronous chroma aggregation before chord decoding. That was
dropped in favour of decoding at frame resolution (~93 ms), because it couples the chord
track to beat-tracker quality precisely where beat tracking is least trustworthy — the
rubato and classical material the plan itself flagged as a fallback case. Frame-level
decoding also keeps boundaries comfortably inside the ±0.25 s tolerance, which one beat at
most tempos exceeds on its own.

Beats are still tracked, reported, and used — but only as an optional post-hoc snap of
chord boundaries, and only when the grid earns it (see below).

### 2. Beat snapping requires prior agreement with the grid

Snapping boundaries to the nearest beat made results measurably *worse* on the first
fixture: maximum boundary error went from 0.081 s to 0.204 s. The beat grid was perfectly
steady and still wrong — locked to a subdivision and offset from the real chord changes.
Steadiness is not correctness, so the original "is the grid steady?" check was not enough.

Snapping is now applied only when the decoded boundaries *already* agree with the grid
(median deviation under 20% of a beat). It can refine an alignment that is already right;
it can no longer invent one.

### 3. Augmented triads are deduplicated

`C:aug`, `E:aug` and `G#:aug` are the same pitch-class set. Chroma has no bass
information to distinguish them, so keeping all three only split probability mass between
identical templates and diluted the confidence of every neighbouring chord. Only the
lowest-root spelling is kept, taking the state count from 85 to **77**.

### 4. Chord-tone weighting, without which sevenths collapse to triads

The plan's binary-with-harmonic-weighting templates weighted every chord tone equally.
That systematically mislabelled `Dm7` as `D:min` and `Cmaj7` as `C:maj`: a four-note
template spreads its unit norm over more bins than a triad's, so root-heavy audio matches
the triad better even when the seventh is plainly sounding.

Chord tones are now weighted by role — root 1.0, fifth 0.8, third 0.75, seventh 0.5 —
which reflects how present each voice actually is in a recording. All six test
progressions pass across a broad band of settings.

### 5. Key detection uses chord evidence, and the pipeline order is reversed

This was the largest change. Profile correlation alone got **0 of 12 minor keys right** —
every single one was reported as its relative major. That is not tuning noise: a key and
its relative share an identical pitch-class content, so no pitch-class histogram can
separate them, whichever profile set is used.

What separates them is which chord acts as home, and the chord track already shows that
unambiguously. So the pipeline now decodes **chords before key**, and the key stage adds a
chord-evidence term: diatonic membership weighted by sounding time, plus credit for time
spent on the tonic triad and for the piece starting or ending there. Chord decoding needs
no key, so there is no circularity.

Both score vectors are standardised across the 24 candidate keys before being combined —
without that, the mixing weight silently encodes a unit conversion rather than a relative
importance. Result: **36 of 36** key fixtures correct (12 major, 24 minor across two
progression shapes).

`detect_key()` still works without chords, and still cannot resolve relative
major/minor in that mode. That is documented rather than hidden.

### 6. The modulation scan is Viterbi-smoothed

Per-window argmax flickered between a key and its relative in regions where the key was in
fact perfectly steady, which broke the "N consecutive identical windows" rule and hid real
modulations. The local key is a piecewise-constant latent observed through noisy
overlapping windows — the same problem shape as chord decoding — so it is now decoded with
the same `viterbi_decode`, with a much stickier self-transition. The "first/last chord is
the tonic" bonus is also disabled for windows, where the edges fall wherever the hop puts
them rather than at structural boundaries.

### 7. soundfile-first decoding

The plan specified ffmpeg for all decoding. Reading WAV/FLAC/OGG through libsndfile first
and falling back to the ffmpeg pipe means the entire test suite — and the common case of
analysing a WAV — needs no external binary. ffmpeg remains required for mp3/m4a and URLs.

### 8. FastAPI imports live at module scope

Importing FastAPI's symbols inside `create_app()` left pydantic unable to resolve
`UploadFile | None`, and every route returned 422. FastAPI resolves endpoint annotations
against module globals, so the import is now module-level behind a `FASTAPI_AVAILABLE`
flag — the module still imports without the `[web]` extra, and `create_app()` raises a
helpful error instead.

---

## Deferred (phase 2)

- Neural chord engine (crema) behind `[deep]`
- Inversion and bass-note detection
- Section-aware segmentation (verse/chorus)
- Waveform rendering in the web UI
- Sus, 6th, 9th and altered chord qualities
- An offline eval script scoring against Isophonics/Beatles `.lab` annotations, if the
  user has downloaded them locally — never fetched automatically
