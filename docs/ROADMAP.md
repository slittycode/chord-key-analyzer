# Roadmap

Where the project goes after 0.2.0. This is the surviving half of `PLAN.md`'s
deferred list — the items Stage 3 claimed have been struck out and replaced with
what they actually turned into, because a roadmap that still promises finished
work is worse than no roadmap.

## Status after 0.2.0

| Deferred in `PLAN.md` | Where it landed |
| --- | --- |
| Inversion and bass-note detection | Shipped in 0.2.0 — `chords[].bass`, `/3` in `.lab`, `C:maj/E` for humans |
| Section-aware segmentation | Shipped in 0.2.0, but as **repetition only** (`A`/`B`/`A'`), never verse/chorus |
| Offline eval against `.lab` annotations | Shipped in 0.1.0 as `cka eval` |
| 6th qualities | Half done: `maj6` exists as a bass-driven re-spelling. `min6` does not — see below |
| Sus qualities | **Measured and rejected.** See "Vocabulary" |
| Neural chord engine behind `[deep]` | Not started. The seam is pre-wired |
| Waveform rendering in the web UI | Not started |
| 9th and altered qualities | Not started |

## Near term

1. **Merge PR #4 and tag `v0.2.0`.** The branch is green and the changelog entry
   is written; this is a user action, not a code change.
2. **Phantom sections on a flat novelty curve.** A track whose novelty curve is
   exactly flat — digital silence is the reachable case — gets a boundary every
   `wait + 1` blocks instead of none, because `librosa.util.peak_pick` on an
   all-zero array with `delta = 0` treats every position as a peak. Sixty
   seconds of silence currently yields six sections, `A` through `F`. Guard
   `_boundaries` on the curve's spread. See "Known defects" below.
3. **`min6` re-spelling.** Documented out of scope for Stage 3 and still open.
   Unlike `maj6`, `min6` is enharmonically `hdim7`, which is not a decoder state
   — so there is nothing to re-spell *from*. Doing it properly means either
   adding `hdim7` as a state (it currently loses its own renders to the plain
   diminished triad at all twelve roots) or driving the re-spelling from
   something other than an existing state.
4. **DRM-aware ingest errors.** A subscription-protected `.m4p` currently dies
   with a generic ffmpeg error that tells the user nothing. It should name the
   problem: this file is protected, that is not something the tool will work
   around, analyse a DRM-free copy instead. See `APPLE-MUSIC.md`.

## Next

1. **Neural chord engine (crema) behind `[deep]`.** The extra is already declared
   and deliberately empty, and `ChordEngine` in `chords.py` is the protocol the
   CLI drives — the seam exists precisely so this can be added without touching
   the pipeline. This is the single biggest available accuracy win on real
   material, where the template decoder's weakness is exactly the thing a
   learned emission model fixes.
2. **Waveform rendering in the web UI.** The chord timeline is already there;
   what is missing is the signal underneath it, which is what makes a boundary
   look right or wrong at a glance.
3. **9th and altered qualities.** Gated on the same thing as `sus4` below: a
   template decoder over chroma cannot reliably tell an extension from a passing
   note. This is downstream of the neural engine, not parallel to it.

### Vocabulary: what was measured and rejected

Recorded in `models.py` and repeated here so nobody re-derives it. The
vocabulary stayed at 77 states through Stage 3 — the plan proposed adding `sus4`
and the measurement said no:

- **`sus4`** is collision-free and decodes its own renders 7 roots out of 7, and
  still fails. A suspended template is too good a match for melody: two stepwise
  notes blurred by the chroma median filter, plus their fifths, *are* a sus
  chord. A bare scale decodes as a sus4 on every degree and the key evidence
  collapses with it — A minor read as C major. Held until the emission model can
  distinguish a sounding fourth from a passing one.
- **`hdim7`** loses its own renders to the plain diminished triad at all twelve
  roots; the dim triad's partials already energise the flat seventh's bin.
- **`dim7`** is symmetric — three distinct sets across twelve roots — so the
  canonical root is arbitrary without a confident bass.

## Integration track

1. **Apple Music resolution.** Design and a working MVP are in
   `APPLE-MUSIC.md` and `tools/nowplaying_analyze.py`: ask Music.app for the
   current track's file path and analyse that file. Metadata and transport only;
   protected files are refused with a clear message, never circumvented.
2. **`cka` as an MCP server**, so any Claude session can call `analyze` directly
   rather than shelling out. `analyze_audio(y, sr, ...)` in `pipeline.py` is the
   clean in-memory seam, and `--json -` already emits clean JSON on stdout with
   every status line on stderr, so the transport work is small.

## Known defects

Carried here rather than in an issue tracker because the repo does not have one.

- **Flat-novelty sections** (above). One-line guard in `sections.py::_boundaries`:
  return `[0]` when `novelty.std()` is zero to within floating point. Verified to
  fix the silence case without moving a boundary on real material.
- **`_label_spans` on zero vectors.** Related, and moot once the above is fixed
  for the only input that reaches it: every span's average chroma is the zero
  vector, every similarity is `0.0`, and the comparison is strictly greater —
  so no span ever matches a previous one and each gets a fresh letter. If the
  labeller is ever reached with degenerate input by another path, it should treat
  a zero-norm average as "same as anything", not "different from everything".
