# Stage 3 plan: pre-flight → bass/inversions → vocabulary → sections → 0.2.0

This is the implementation plan for the third round of work. Like `PLAN.md` before it,
it is the source of truth for the build: work the milestones in order (except where
parallelism is called out), land each as its own PR with CI green, and document any
departure this plan did not anticipate.

## Context

Stages 1–2 are merged (PR #1: v1 analyzer; PR #2: hardening + `cka eval`). Verified state of `main` (`2901c90`): 237 tests pass + 2 skip locally and in CI (3.11/3.12), ruff clean, zero TODO/FIXME, docs closely synced.

**Why a feature round and not a cleanup round:** a deep review of `main` found only ~100 lines of genuine residue — folded in here as pre-flight milestone M0, not its own round. The round carries **all three deferred features** as separate PR-sized milestones ending at a tagged 0.2.0.

**Sequencing:** M0 → M1 → M2 (M2 *hard-depends* on M1: verified pitch-class collisions — C:sus2 ≡ G:sus4, C:maj6 ≡ A:min7, C:min6 ≡ A:hdim7 — mean the new qualities are indistinguishable without bass information). M3 is independent, may run in parallel with M1/M2 (conflicts confined to `models.py:to_dict`, `pipeline.py`, README schema example, `index.html`). M4 last. Each milestone = one PR against `main`, CI green.

**Schema policy (applies to M1+M3):** `SCHEMA_VERSION` stays 1; all additions are strictly additive (`chords[].bass`, top-level `sections`); README gains one contract sentence: consumers must ignore unknown keys.

**Release facts:** version 0.1.0 in `pyproject.toml`+`__init__.py`; CHANGELOG's `v0.1.0` tag links 404 — **no tags exist**. Stage 3 creates both v0.1.0 (retro, at `2901c90`) and v0.2.0.

**User action item (repo settings, not agent-doable):** default branch is still `claude/music-analyzer-plan-1gz9jx` — switch to `main`; delete stale `claude/chord-key-analyzer-build-oczump` and `claude/chord-key-analyzer-stage-2-8gbk1z`.

All claims marked [verified] were checked against `2901c90` or executed with mir_eval 0.8.2 in a venv.

---

## M0 — Pre-flight fixes (~100 lines, one PR)

1. **Guard mir_eval scoring in `evaluate_track`** — `evaluate.py:276` `mir_eval.chord.evaluate` is unguarded; `InvalidChordException` derives from `Exception` not `ValueError` [verified], so one unparseable label in a user `.lab` kills the corpus run, contradicting `evaluate.py:261`, the README, and the audio-only test `test_undecodable_audio_is_recorded_not_raised`. `mir_eval.io.load_labeled_intervals` does NOT parse labels, so bad labels reach `evaluate()`. Fix: bind `InvalidChord = mir_eval.chord.InvalidChordException` after `_require_mir_eval()`; wrap the evaluate call in `except (ValueError, InvalidChord)` → `error = f"chord scoring failed: {exc}"`; wrap `weighted_score` (`:286`) in `except ValueError` similarly (it raises ValueError on bad key strings [verified]). Tests in `tests/test_evaluate.py`: two-track dataset (one good, one with `C:notachord`) → bad track errors, run continues, `summarise` shows 1 scored/1 failed; monkeypatched `weighted_score` raising → recorded not raised.
2. **Move evidence damping into the key seam** — `key.py:132` `estimate_key_from_chroma` defaults to undamped `chord_weight=KEY_CHORD_WEIGHT`; damping is applied at both call sites (`detect_key` ~:405, `detect_modulations` ~:347) instead of inside. Fix: new `_blend_chord_evidence(scores, chords, use_edges)` = `scores + KEY_CHORD_WEIGHT * _evidence_support(chords) * _standardise(chord_evidence_scores(chords, use_edges))` (must call module-global `_evidence_support` so the existing monkeypatch test at `tests/test_key.py:153` still observes it). Change `estimate_key_from_chroma(chroma_vector, n_alternatives=3, chords=None, use_edges=True)` — **drop** `chord_scores`/`chord_weight` (0.x break); both callers simplify. New test: pure C-major profile chroma + single sustained `ChordSegment(0, 45, "G:maj")` → winner stays C major (undamped 2.0 flips it; margins pinned at `test_key.py:179–205`).
3. **Check `[eval]` extra up-front in CLI** — `_require_mir_eval` fires per-track after discovery output. Add public `require_eval_extra()` (mirror `web.py:122 require_web_extra`); `cli.py:eval_cmd` calls it before `discover_pairs`, `except EvalExtraMissing → SystemExit(str(exc))`. Test in `tests/test_cli.py` (NOT the importorskip-gated file): `monkeypatch.setitem(sys.modules, "mir_eval", None)` → nonzero exit, hint present, no discovery output.
4. **Make `download_url`'s `dest_dir` required** — `ingest.py:197/:217` mkdtemp fallback is dead code (only caller `downloaded_media` always passes it) and a leak-reintroduction trap. Delete the fallback, require the parameter, rewrite docstring. Update `tests/test_ingest.py:113` (pass `dest_dir=tmp_path`) and the stale fake docstring at `:190`. Cleanup pinning tests (`:217`, `:228`) unchanged.
5. **Case-insensitive dataset discovery** — `evaluate.py:148` rglobs `*.lab`, lowercase-only `AUDIO_EXTENSIONS`; `Track.WAV`/`.LAB` silently orphan. Fix: iterate `sorted(root.rglob("*"))` filtering `suffix.lower() == ".lab"`; skip key labs via `name.lower().endswith(".key.lab")`; `_audio_for` scans `lab.parent.iterdir()` for `p.stem == lab.stem` (stem stays case-sensitive — document why) with `p.suffix.lower() in AUDIO_EXTENSIONS`, priority by extension order then name; `_key_for` likewise. Tests: `song.LAB`+`song.WAV` → one pair; `.Wav` variant; existing orphan test pins the miss path.
6. **Docs/dead-code** — README "binds to localhost only" → "by default"; README + `web.py:5` claim endpoint calls `analyze_audio()` (it calls `analyze_source`, `web.py:201`); README JSON example `meta` gains `sample_rate`/`hop_length`/`triads_only` + note about conditional `offset` [verified vs `pipeline.py:66–76,108–112`]; delete dead `@runtime_checkable` on `ChordEngine` (`chords.py:49`) + its import (no isinstance use anywhere).

**Verify:** `ruff check .`; `pytest -q`; both CI e2e gates unchanged. Risk: `estimate_key_from_chroma` signature break (accepted, 0.x); a key-scoring failure voids that track's chord scores in the summary (nearly unreachable; note in PR).

---

## M1 — Bass salience + inversion (slash-chord) detection

**Goal:** low-register chroma in `Features`; per-segment bass pitch class; MIREX slash labels for chord-tone basses ≠ root; murky bass → no slash (never a wrong one).

**Verified design facts:** bass chroma = `chroma_cqt(..., fmin=note_to_hz("C1"), n_octaves=3, bins_per_octave=36, tuning=tuning)` frame-aligns with the main chroma at hop 2048 (no resampling; defensively truncate/pad). On synthesized inversions (chord @ octave 4, bass @ octave 2) per-frame argmax recovers the bass pc with 100% agreement for C:maj/3, C:maj/5, F:maj7/7, A:min/b3; root-position reports the root. mir_eval parses **degree** slashes (`C:maj/3`, `A:min7/b7`) but rejects note-name slashes (`C:maj/E`) [verified]. `root/majmin/sevenths/mirex` all ignore bass (`C:maj` vs `C:maj/3` = 1.0 both ways) [verified] — the CI eval gate cannot move.

**JSON contract (decided):** `label` stays plain; additive `bass: str | null` field (pitch-class name); slash form appears in `.lab` (degree notation), terminal, and web renderings. Schema stays 1. Rationale: every existing `label` consumer (`parse_chord_label`, `roman_numeral`, `chord_evidence_scores`, web timeline) is untouched.

**Changes (ordered):**
1. `features.py`: hoist HPSS out of `compute_chroma` into `extract_features` (one `librosa.effects.harmonic(y, margin=3.0)` run feeds both chromas; `compute_chroma` keeps its `harmonic` param for direct callers). New `compute_bass_chroma(...)` — median-filtered (width 9), **unnormalised** so per-frame energy supports the salience test. `Features` gains `bass_chroma: np.ndarray  # (12, n_frames)`.
2. `models.py`: `ChordSegment.bass: str | None = None` (last-with-default keeps positional constructors valid); `to_dict` adds `"bass"` unconditionally. `BASS_DEGREES = {1:"b2",2:"2",3:"b3",4:"3",5:"4",6:"b5",7:"5",8:"b6",9:"6",10:"b7",11:"7"}`; `ChordSegment.mirex_label` (degree slash) and `display_label` (`C:maj/E`) properties.
3. `chords.py`: `BASS_SALIENCE = 0.30` (strongest bass bin ≥ 30% of frame's bass energy; uniform = 1/12≈0.083), `BASS_AGREEMENT = 0.6`. New `detect_inversions(segments, features)`: per pitched segment, use non-silent bass-voiced frames; < half the segment's frames voiced → no bass; bass pc = duration-weighted mode of per-frame argmax needing the agreement share; emit `bass` only when it's a chord tone (from `CHORD_QUALITIES`) and ≠ root; return via `dataclasses.replace`. **Must run after all merge/snap helpers** (they rebuild segments and would drop `bass`) — document in docstring.
4. `pipeline.py`: `chords = detect_inversions(chords, features)` right after the engine, before `detect_key`.
5. `output.py`: `to_lab` uses `mirex_label`; chord table uses `display_label`.
6. `web_static/index.html`: table shows `label + "/" + bass` when present; `.lab` download converts via a JS mirror of `BASS_DEGREES`.
7. `tests/fixtures.py`: `render_chord(bass_interval: int | None = None, bass_octave: int = 2)` (bass note = root+interval @ octave 2, amp ~1.2); `render_progression` accepts 3-tuples.
8. README: `"bass": null` in schema example + additive-fields contract sentence; rewrite "Not modelled: inversions"; short Inversions paragraph (degree `.lab` vs `bass` JSON).

**Tests** (new `tests/test_inversions.py` + edits): parametrised first/second inversions over ≥4 roots (maj/min) + `/7` on maj7 → expected `bass` and `mirex_label`; root-position and no-bass renders → `bass is None`; `.lab` round-trip through `mir_eval.io.load_labeled_intervals` + `chord.encode` (importorskip-guarded); extend `test_every_vocabulary_label_parses_in_mir_eval` over quality × chord-tone degrees; eval round-trip with `C:maj/3` reference lines ≥ 0.75 root/majmin; `test_output.py` sample with `bass="E"` → JSON/`.lab`/table forms.

**Risks:** real recordings fail salience more often than fixtures — designed failure = no slash. HPSS hoist must be behavior-identical (full suite is the check; 12 `extract_features` call sites in tests). CI e2e asserts plain labels on root-position audio — unaffected.

---

## M2 — Extended vocabulary, gated by measurement (depends on M1)

**Collision policy (all computed, not assumed):**

| Candidate | Verdict | Why |
|---|---|---|
| `sus4` (0,5,7) | **Add as template states** (77 → 89) | Collision-free; existing six fixture progressions decode identically with it added; own renders detected [verified] |
| `sus2` | **Never a state; bass-driven re-spell** | C:sus2 ≡ G:sus4 — dedup would keep an arbitrary spelling. Detected `X:sus4` with bass = its 4th → relabel `(X+5):sus2` |
| `maj6` (`6` isn't valid mir_eval [verified]) | **Bass-driven relabel of min7** | C:maj6 ≡ A:min7 at every root. `X:min7` with bass = its b3 → `(X+3):maj6`, `bass=None` (re-spelled chords are root-position by construction) |
| `min6` | **Out of scope** | ≡ hdim7, which isn't added; documented follow-up |
| `hdim7` | **Do not add** | Collision-free but *fails its own render*: B:hdim7 render scores B:dim 0.9163 vs B:hdim7 0.9098 (dim-triad partials energise the b7 bin); raising 7th weight worsens it (0.8614) [measured]. Record margins in PR |
| `dim7` | **Do not add** | Symmetric (3 sets/12 roots); canonical root arbitrary without confident bass |
| `7sus4` | **Do not add** | mir_eval rejects the label [verified] — breaks `.lab` round-trips |

**Changes:** `models.py`: `CHORD_QUALITIES["sus4"] = (0,5,7)`; `RESPELLED_QUALITIES = {"maj6": (0,4,7,9), "sus2": (0,2,7)}` (label-only, never decoder states); `TRIAD_QUALITIES`/`--triads-only` unchanged (documented contract = maj/min/dim/aug). `chords.py` or sibling `respell_with_bass(segments)` after `detect_inversions` in the pipeline: the two relabel rules above; all other bass results keep M1 slash behavior. `progression.py`: `_QUALITY_SUFFIX += {"sus4":"sus4","sus2":"sus2","maj6":"6"}` (uppercase numerals: `Isus4`, `I6`). `key.py:_diatonic_chords`: refactor to scale-membership derivation (`MAJOR_SCALE={0,2,4,5,7,9,11}`, `MINOR_SCALE={0,2,3,5,7,8,10,11}` natural∪harmonic), **excluding `aug`**; hand-verified to reproduce the existing tables exactly, then extends naturally to sus4/sus2/maj6; precompute at import. `tests/fixtures.py`: render support for the three qualities. README: vocabulary section, 77→89 state-count mentions.

**Gate (CI-enforced) — a quality ships only if all three hold:** (a) own renders decode to it (the sus fixture case `C:maj C:sus4 G:sus4 G:maj` passes today [verified]; re-spell test per bass-driven quality); (b) zero label changes on the six existing fixture progressions; (c) CI eval gate holds (`majmin ≥ 0.75`, `key == 1.0` — mir_eval treats sus4/maj6 as non-comparable under majmin/sevenths so the POP gate can't move [verified]). One bounded tuning attempt allowed (only `BASS_AGREEMENT`/`BASS_SALIENCE`; never `TONE_WEIGHTS`); otherwise **drop the quality** and record the measurement in the PR — as already done for hdim7.

**Tests:** sus progression case in `test_chords.py` (state-count test self-updates — it computes from the dict [verified]); re-spell suite: `A:min7`+bass C → `C:maj6`/roman `I6`; `G:sus4`+bass C → `C:sus2`; `A:min7`+bass A → unchanged; `A:min7`+bass E → `A:min7/5` (slash, not relabel); `_diatonic_chords` derivation-pinning test freezing the old literal sets; roman tests; vocabulary-parses test extended over `RESPELLED_QUALITIES`.

**Risk:** the maj6 relabel changes the reported root — vs an `A:min7` reference, `C:maj6` scores 0 on root/majmin/sevenths (mirex still 1.0) [verified]. High `BASS_AGREEMENT` is the guard; drop criteria the escape hatch.

---

## M3 — Section-aware analysis (independent; parallel-safe with M1/M2)

**Method (decided):** Foote checkerboard novelty on 1 s block-aggregated chroma (+ z-scored rms row @ 0.5 weight), SSM `X.T @ X` on column-normalised blocks, Gaussian-tapered checkerboard kernel L=16 blocks, `librosa.util.peak_pick(pre_max=4, post_max=4, pre_avg=8, post_avg=8, delta=0.10*novelty.std(), wait=8)` (≥8 s sections). Rationale: no pre-chosen segment count (agglomerative's weakness), trivial cost, existing deps only; fixed blocks not beat-sync (sections are ±2–4 s objects; avoids coupling to beat-tracker quality — same reasoning as PLAN.md change #1). Track < 30 s → `[]` (mirrors modulation guard). **Neutral labels**: greedy letters by cosine of section-representative chroma (≥0.92 same letter, ≥0.80 primed `A'`, else next letter) — never "verse"/"chorus".

**Changes:** `models.py`: promote `key.py:_chords_in_window` to public `clip_segments(segments, start, end)` in models (key.py delegates); frozen `Section(start, end, label, tonic|None, mode|None, key_confidence|None, progression)` with `to_dict`; `AnalysisResult.sections: list[Section] = []`, additive in `to_dict`. New `sections.py`: `detect_sections(features, chords, global_key)` with constants `SECTION_BLOCK_SECONDS=1.0`, `SECTION_KERNEL_BLOCKS=16`, `SECTION_MIN_DURATION=8.0`, `SECTION_MIN_TRACK=30.0`, `SECTION_SAME_LABEL=0.92`, `SECTION_PRIME_LABEL=0.80` (house-style why-comments). Per-section key hint via post-M0 `estimate_key_from_chroma(pooled, chords=clipped, use_edges=False)`; per-section progression via `summarise_progression(clipped, global_key)` — numerals stay relative to the **global** key for comparability. `pipeline.py`: `scan_sections: bool = True`, `progress("sections", 0.95)`. `cli.py`: `--no-sections`, stage label. `output.py`: `_sections_panel` (one line per section: `A  0:00.0–0:32.0  C major  I – V – vi – IV`) between key panel and chord table. `index.html`: sections lane atop the timeline SVG (height → 140), `result.sections || []` guard for pre-M3 results. README: sections paragraph (explicit honesty re structural-not-semantic labels), schema example, flags.

**Tests** (new `tests/test_sections.py`): A/B/A fixture (POP_LOOP_C ×4 = 32 s, then E-major progression ×4, then A again) → exactly 3 sections, boundaries ~32 s/~64 s ±4.0 s, `base_letter(0)==base_letter(2)!=base_letter(1)`, section 2 key hint tonic E; homogeneous 8-repeat loop → 1 section `A`; <30 s → `[]`; `test_output.py` sample with a Section round-trips; `--no-sections` → `"sections": []`; assert once that the 24 s `pop_result` conftest fixture stays `sections == []` (below guard, so existing tests untouched).

**Risks:** `peak_pick` params are the tuning surface — widen `delta` before touching the kernel if flaky; check `test_web.py` result assertions for exact-key comparisons.

---

## M4 — Release 0.2.0 + bare-install CI leg

1. `.github/workflows/ci.yml` new job `bare-install` (3.12 only; extras-absence is version-independent): `pip install -e '.[dev]'`; `pytest -q -rs | tee`, then a check script asserting exit 0, every skip reason matches `requires the \[(eval|web)\] extra|ffmpeg not installed`, **skip floor ≥ 40** and passed ≥ 150 (catches an extra silently becoming a core dep; allowlist catches unexplained skips). CLI hint assertions (offline-safe: `download_url` imports yt_dlp before any I/O [verified]; M0 fix 3 makes the eval hint precede discovery): `cka eval .` → nonzero + `chord-key-analyzer[eval]`; `cka web --no-browser` → `[web]` hint; `cka analyze 'https://example.invalid/song'` → `[url]` hint (RFC-2606 domain — can't touch the network even if misordered).
2. Version → 0.2.0 in `pyproject.toml` + `__init__.py`; new `tests/test_package.py::test_version_matches_installed_metadata` via `importlib.metadata` (skip on PackageNotFoundError) — runs in every leg including bare.
3. CHANGELOG `[0.2.0]`: Added (bass/inversions + `bass` field + slash `.lab`; sus4 + maj6/sus2 re-spelling; sections + `--no-sections`; bare-install CI leg) / Changed (`estimate_key_from_chroma` signature; `download_url` requires `dest_dir`; 77→89 states) / Fixed (M0 items). Fix link refs (`[Unreleased]` compare, add `[0.2.0]`).
4. README coherence pass over all M1–M3 edits.
5. **Tags (post-merge steps; none exist today):** `git tag -a v0.1.0 2901c90 -m "chord-key-analyzer 0.1.0"` (the state its CHANGELOG entry describes); `git tag -a v0.2.0 <M4 merge sha>`; `git push origin v0.1.0 v0.2.0`. Both CHANGELOG links must then resolve.

**Risk:** `-rs` parsing — keep to reason-regex + floors, never exact counts.

---

## Whole-round verification

Per PR and again after M4:
1. `ruff check .` + `pytest -q` on 3.11/3.12 (expect ~237 + 35–45 new tests).
2. CI e2e **analyze** gate unchanged at every milestone (`schema == 1`, C major, `["C:maj","G:maj","A:min","F:maj"]`, `I V vi IV`) — the standing no-regression tripwire.
3. CI e2e **eval** gate (`majmin ≥ 0.75`, `key == 1.0`) — M2's measurement gate rides on it.
4. Bare-install leg green (M4).
5. Manual smoke: M0 — `cka eval` over a dir with one bad `.lab` completes and reports the failure; M1 — inverted render shows `/3` in `.lab` and `"bass"` in JSON; M2 — sus4 render shows `Isus4`; M3 — A/B/A render shows the sections panel + web sections lane; M4 — bare venv reproduces all three install hints.
6. Post-merge: both tags pushed and CHANGELOG links resolve.
