# Contributing

## Setup

```bash
pip install -e '.[dev,web,url,eval]'
```

All extras, so `pytest` doesn't skip anything and `ruff check .` sees the whole
tree. `ffmpeg` on `PATH` is only needed for the ffmpeg-fallback decode path and
for URL ingestion; tests that need it skip cleanly when it's absent.

## Before opening a PR

```bash
ruff check .
ruff format --check .
pytest
```

Tests synthesise their own audio with numpy — nothing is downloaded and no
audio fixtures are committed. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for
what's planned and known-broken, and [`docs/PLAN.md`](docs/PLAN.md) for the
original design rationale.

## Scope

Known defects and planned work are tracked in `docs/ROADMAP.md` rather than
GitHub Issues — check there before filing something that might already be
on it.
