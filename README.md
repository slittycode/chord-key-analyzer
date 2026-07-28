# chord-key-analyzer

A lean, fully offline music analyzer: point it at a song — a local audio file
or a YouTube/SoundCloud/Bandcamp URL — and it reports the track's **key**, its
**chords** on a timeline, and the **chord progression** (Roman-numeral
analysis). No cloud APIs, no LLM keys, works across genres (rock, classical,
electronic, jazz).

Inspired by [ableton-sonic-analyzer](https://github.com/slittycode/ableton-sonic-analyzer),
but deliberately smaller: a single Python package with a `cka` CLI and a
minimal local web UI, instead of a FastAPI + React + Essentia + Gemini stack.

## Status

**Planning.** The build has not started yet — see [`docs/PLAN.md`](docs/PLAN.md)
for the full design and milestone plan.

## Planned usage

```
cka analyze song.mp3                 # key, chord timeline, progression
cka analyze https://youtube.com/...  # via yt-dlp (pip install "cka[url]")
cka web                              # local drag-drop UI (pip install "cka[web]")
```

Note: Spotify and Apple Music streams are DRM-protected and out of scope;
DRM-free files you own (e.g. Bandcamp downloads) work as local files.
