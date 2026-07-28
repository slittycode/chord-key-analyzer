# chord-key-analyzer

Tell me the **key**, the **chords**, and the **progression** of a song — offline, with
no API keys and no cloud services.

Point it at an audio file (or a YouTube/SoundCloud/Bandcamp URL) and it reports the
key with its confidence and alternatives, a chord timeline, and the Roman-numeral
progression including the dominant repeating loop.

```console
$ cka analyze track.wav

track.wav
╭──── Key ─────╮
│ C major      │
│ confidence   │
│ 84%          │
│ alternatives │
│ A minor, G   │
│ major, F     │
│ major        │
│ tempo        │
│ 120.0 BPM    │
╰──────────────╯
                Chords
 Start   End   Chord   Roman   Conf
 0:00.0 0:02.0 C:maj   I        41%
 0:02.0 0:04.0 G:maj   V        43%
 0:04.0 0:06.0 A:min   vi       40%
 0:06.0 0:08.0 F:maj   IV       45%
 …
╭──────── Progression ────────╮
│ I – V – vi – IV  ×3         │
│ C:maj – G:maj – A:min – F…  │
│ 0:00.0–0:24.0               │
│                             │
│ most played  C:maj (26%), … │
╰─────────────────────────────╯
```

## Why this exists

It is inspired by [`ableton-sonic-analyzer`](https://github.com/slittycode/ableton-sonic-analyzer),
and keeps its one genuinely good idea — **deterministic DSP measurements are
authoritative** — while dropping everything that made it heavy. There is no web stack
to build, no LLM in the loop interpreting results, no stem separation, and no Essentia
to fight with at install time. What you get is a small Python package that runs the
same analysis from a CLI or a local page in your browser.

## Install

Requires **Python ≥ 3.11**.

```bash
pip install chord-key-analyzer                 # core: local audio files
pip install 'chord-key-analyzer[url]'          # + YouTube / SoundCloud / Bandcamp
pip install 'chord-key-analyzer[web]'          # + the local web UI
pip install 'chord-key-analyzer[url,web]'      # everything
```

**ffmpeg** must be on your PATH to decode compressed formats (mp3, m4a, …) and for URL
downloads. WAV, FLAC and OGG are read directly and need no ffmpeg at all.

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian / Ubuntu
winget install Gyan.FFmpeg   # Windows
```

## Usage

### CLI

```bash
cka analyze song.mp3                      # pretty terminal report
cka analyze song.mp3 --json result.json   # machine-readable
cka analyze song.mp3 --json -             # JSON to stdout (report goes to stderr)
cka analyze song.mp3 --lab song.lab       # MIREX-style chord annotation
cka analyze song.mp3 --start 30 --duration 60   # analyse an excerpt
cka analyze song.mp3 --triads-only        # maj/min/dim/aug only, no sevenths
cka analyze 'https://www.youtube.com/watch?v=...'   # needs the [url] extra
```

Useful flags: `--no-hpss` (skip harmonic/percussive separation — faster, less accurate on
drum-heavy material), `--no-beat-snap`, `--no-modulations`, `--quiet`.

### Web UI

```bash
cka web            # opens http://127.0.0.1:8321
```

Drag in a file or paste a URL and you get the key card, an SVG chord timeline, the
progression, and download links for the JSON and `.lab`. It is one static page with
inline vanilla JS — no React, no node, no build step — and it binds to localhost only.
The endpoint calls the exact same `analyze_audio()` the CLI does, so the two can never
disagree.

### Python

```python
from chord_key_analyzer import analyze_source

result = analyze_source("song.mp3")
print(result.key.name, result.key.confidence)
for chord in result.chords:
    print(f"{chord.start:6.2f}–{chord.end:6.2f}  {chord.label}")
print(result.progression.main_loop.roman)
```

## Source support

| Source | Supported | Notes |
| --- | --- | --- |
| Local WAV / FLAC / OGG | ✅ | Read directly, no ffmpeg needed |
| Local MP3 / M4A / AAC / OPUS / … | ✅ | Anything ffmpeg decodes |
| YouTube, SoundCloud, Bandcamp, … | ✅ | Via `[url]` extra (yt-dlp) |
| Bandcamp / iTunes purchased downloads | ✅ | DRM-free files are just local files |
| **Spotify / Apple Music streams** | ❌ | **Out of scope — see below** |

**Why Spotify and Apple Music are out of scope.** Their streams are DRM-protected. There
is no way to decode them offline without circumventing that protection, which this
project will not do. If you own a DRM-free copy of a track (a Bandcamp download, a
DRM-free m4a), analyse that file directly — it works like any other local file.

## How it works

```
audio ──▶ decode 22.05 kHz mono ──▶ tuning estimate ──▶ HPSS ──▶ CQT chroma
                                                                     │
                        ┌────────────────────────────────────────────┤
                        ▼                                            ▼
              chord templates + Viterbi                       beat tracking
                        │                                            │
                        ├──────────────▶ chord timeline ◀────────────┘
                        │                                (optional boundary snap)
                        ▼
       key: profile correlation + chord evidence ──▶ Roman numerals, loop detection
```

1. **Decode** to mono 22.05 kHz — via libsndfile where possible, otherwise an ffmpeg pipe
   to raw PCM (which sidesteps `audioread`'s backend roulette).
2. **Tuning estimate** so slightly detuned recordings don't smear the chroma bins.
3. **Harmonic/percussive separation**, then **CQT chroma** at 36 bins/octave. Percussion
   otherwise splashes energy across every pitch class at once.
4. **Chords**: each frame is scored against harmonically-weighted templates for 12 roots ×
   7 qualities (plus a no-chord state), then **Viterbi-decoded** with a sticky
   self-transition so the output doesn't flicker frame to frame.
5. **Key**: Krumhansl-Schmuckler and Temperley profiles are correlated against the pooled
   chroma **and combined with evidence from the detected chords**. That second term is
   what resolves relative major/minor — A minor and C major contain identical pitch
   classes, so no pitch-class histogram can separate them, but which chord acts as home
   is decisive.
6. **Modulations**: the same key scoring runs over a sliding 20 s window, and the
   resulting sequence is itself Viterbi-smoothed, so overlapping windows don't produce
   phantom key changes.
7. **Progression**: chords become Roman numerals in the detected key (borrowed chords are
   written literally, e.g. `bVII`), and n-gram scanning finds the dominant repeating loop.

### Chord vocabulary

`maj`, `min`, `dim`, `aug`, `maj7`, `min7`, `7` (dominant), plus `N` for no-chord.
Labels are MIREX-style: `C:maj`, `F#:min7`, `N`.

Augmented triads repeat every four semitones, so `C:aug`, `E:aug` and `G#:aug` are one
and the same pitch-class set. Chroma carries no bass information to tell them apart, so
only the lowest-root spelling is a decoder state.

## Accuracy expectations

This is a chroma-template system, and it is honest about what that means:

- **Strong**: triadic rock, pop and electronic music with a clear harmonic rhythm.
  Key detection on tonal material is reliable.
- **Reasonable**: jazz with sevenths, though dense voicings, altered chords and
  substitutions will be approximated by the nearest vocabulary entry.
- **Approximate**: fast classical harmony, heavy chromaticism, and rubato playing where
  beat tracking gives up (the pipeline detects this and falls back rather than trusting a
  bad grid).
- **Not modelled**: inversions and bass notes, suspensions, 6ths, 9ths and other
  extensions, key changes shorter than about 30 seconds.

Confidence values are real signals, not decoration — treat anything below ~0.4 as the
analyzer telling you it is unsure. Chord confidences are posterior probabilities across
the whole 77-state vocabulary, so they are naturally lower than a yes/no score: many
chords genuinely share most of their pitch classes.

## Engines

The `--engine` flag selects the chord backend. v1 ships `template` (the default).
`ChordEngine` is a small protocol — `analyze(features) -> list[ChordSegment]` — so a local
neural backend can be added later without changing the CLI or the JSON schema. Asking for
an unimplemented engine fails loudly rather than silently falling back.

## JSON schema

Versioned via `"schema": 1`.

```json
{
  "schema": 1,
  "file": "song.wav",
  "duration": 24.0,
  "tempo": 120.0,
  "key": {
    "tonic": "C", "mode": "major", "confidence": 0.84,
    "alternatives": [{"tonic": "A", "mode": "minor", "score": 1.42}],
    "modulations": [{"start": 25.0, "end": 60.0, "tonic": "E", "mode": "major",
                     "confidence": 0.71}]
  },
  "chords": [{"start": 0.0, "end": 2.04, "label": "C:maj", "confidence": 0.41}],
  "progression": {
    "roman": ["I", "V", "vi", "IV"],
    "labels": ["C:maj", "G:maj", "A:min", "F:maj"],
    "main_loop": {"labels": ["C:maj", "G:maj", "A:min", "F:maj"],
                  "roman": ["I", "V", "vi", "IV"], "repeats": 3,
                  "start": 0.0, "end": 24.0}
  },
  "meta": {"engine": "template", "version": "0.1.0", "tuning": 0.0,
           "beats_reliable": true, "harmonic_rhythm": 2.0}
}
```

## Privacy

No network calls, ever — except when you explicitly pass a URL to fetch. There is no
telemetry, no model download, and no API key anywhere in this project.

## Development

```bash
pip install -e '.[dev,web,url]'
pytest
ruff check .
```

Tests synthesise their own audio with numpy — no audio files are committed and nothing is
downloaded. See [`docs/PLAN.md`](docs/PLAN.md) for the design rationale and roadmap.

## License

MIT — see [LICENSE](LICENSE).
