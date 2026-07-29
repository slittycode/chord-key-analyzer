# Apple Music integration

Analysing what you are currently listening to, without weakening the project's
position on DRM.

The short version: Apple Music can tell us *which* track is playing and, for
anything you actually own, *where the file is*. That is enough. Everything
downstream is the existing `cka analyze` on a local path.

## The metadata-only reality

`applemusic-mcp` (a local `pipx` install at `~/.local/bin/applemusic-mcp`) exposes
metadata and transport: search the catalogue, read the library, play, pause, skip,
queue. What it does not expose — and could not — is a file path or audio data.
It is a remote control, not a decoder.

That is not a gap to work around. It is the correct shape for that interface.
The path has to come from somewhere else.

## Resolution strategy

Three steps, in order, each falling through to the next only on failure.

### 1. Ask Music.app directly (primary)

AppleScript reaches the running application rather than the web API, and the
running application knows where its files live:

```applescript
tell application "Music" to get location of current track
```

For a DRM-free track — a purchased download, a CD rip, an imported file — this
returns an alias that converts straight to a POSIX path. This is the path that
matters: it is exact, it needs no guessing, and it is the file the user is
literally hearing.

For a subscription stream it returns `missing value` or raises. That failure is
informative, not a problem to route around; see "DRM stance".

**In practice this path is less reliable than it reads.** On the library this was
developed against — 19,492 tracks, iCloud Music Library on — *not one track*
returned a location, including MP3s sitting in Music.app's own
`Media.localized` folder. Their `cloud status` is `uploaded`: the library has
synced them, and the running app no longer volunteers a local path over
AppleScript.

So step 2 is not a rare fallback. On a synced library it is the only path that
works, which is why it is built rather than stubbed, and why its match is always
printed rather than silently trusted.

### 2. Fuzzy filename search (fallback)

Music.app can be closed, or the track can be playing somewhere else entirely
(a browser, Spotify, a DJ set on a USB stick). When there is a title and artist
but no location, search the local corpora — `~/Music` and `~/slskd` — for a
filename that looks like the track, and score candidates on token overlap.

The failure mode of a fuzzy match is **analysing the wrong song and saying
nothing**, which is worse than not matching at all: a key and a chord chart for
some other track are confidently, silently wrong. So the threshold is set high,
the match is always printed for the user to sanity-check, and a near-miss is
reported as a miss.

### 3. Give up, clearly

No location and no confident filename match means saying so, with the reason.

## DRM stance

This is the part the design exists to protect. From `README.md`:

> Their streams are DRM-protected. There is no way to decode them offline
> without circumventing that protection, which this project will not do.

The integration holds that line exactly:

- A `.m4p`, a `kind` containing "Protected", or a subscription track with no
  local location is **detected and named**: this file is subscription-protected
  and cannot be analysed. Buy it, or analyse a DRM-free copy.
- There is no fallback that records audio output, no decryption, no key
  extraction, no attempt to reach the stream. The tool stops.

The current behaviour is worse than that only in its manners: a protected file
handed to `cka analyze` today dies inside `ingest.py` with a generic ffmpeg
error that names nothing. Making that message actionable is a near-term item on
`ROADMAP.md`, and it is the same message this tool prints.

## The MVP

`tools/nowplaying_analyze.py` — stdlib only, no dependency on the package
itself, driving `osascript` and `cka` through `subprocess`:

```
$ python tools/nowplaying_analyze.py
$ python tools/nowplaying_analyze.py --save out.json   # keep the full result
$ python tools/nowplaying_analyze.py --file some.flac  # skip resolution entirely
```

One AppleScript call fetches state, title, artist, album, kind, cloud status and
location together — atomically, so the answer describes one track rather than
whichever track was playing at each of six separate queries.

Both paths, against a real library:

```
$ python tools/nowplaying_analyze.py --duration 60
Music.app reported no file path; searching locally...
Fuzzy match (100% of words): …/Two Shell/Two Shell Demos/03 Pixel Heart.mp3
Analysing 03 Pixel Heart.mp3...

  Two Shell — Pixel Heart
  Two Shell Demos
  03 Pixel Heart.mp3  (60s)

  Key    F# major  (confidence 98%)
  Tempo  129.2 BPM
  Chords F#:maj F#:maj7 F#:dim D#:7 F#:dim B:maj7 D#:min B:dim ...
```

```
$ python tools/nowplaying_analyze.py
Cannot analyse A. G. Cook — Silver Thread Golden Needle: it is an Apple Music
subscription stream with no local file.
Apple Music streams are DRM-protected and this tool will not work around
that. Analyse a DRM-free copy (a purchase, a rip, a Bandcamp download).
```

One implementation note worth keeping: a trailing empty field does not survive
the round trip through `osascript`, so a track with no location comes back with
seven fields rather than eight. That is the common case rather than an error,
and the parser pads instead of rejecting.

## Sketches

Two things this shape makes cheap, neither built.

**Batch library analysis.** The resolution step is the only Apple-specific part;
past it, everything is a list of paths. Walking a playlist through
`analyze_audio()` in-process — rather than paying CLI startup per track — gives
a keys-and-tempos table for a whole crate. The interesting output is not the
per-track rows but the aggregate: what key is this collection in, which tracks
beat-match, where the harmonic neighbours are.

**`cka` as an MCP server.** `analyze_audio(y, sr, ...)` in `pipeline.py` is
already the clean in-memory seam, and `cka analyze --json -` already emits clean
JSON on stdout with every status line on stderr. An MCP server exposing
`analyze(path)` would let any Claude session call this directly instead of
shelling out — and, combined with `applemusic-mcp` in the same session, would
close the loop: ask what is playing, resolve it, analyse it, all without leaving
the conversation.
