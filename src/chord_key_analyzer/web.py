"""Local web UI: a single static page plus a JSON analysis endpoint.

Deliberately small — one FastAPI app, one HTML file, an in-memory job dict.  No
database, no queue, no build step.  ``POST /analyze`` calls exactly the same
:func:`~chord_key_analyzer.pipeline.analyze_audio` the CLI uses, so the two
frontends cannot disagree about results.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# FastAPI resolves endpoint annotations against this module's globals, so these
# names have to live at module scope — importing them inside create_app() leaves
# pydantic unable to resolve `UploadFile | None` and every route 422s.  The
# import stays optional: without the [web] extra the module still imports and
# create_app() raises a helpful error instead.
try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on install extras
    FASTAPI_AVAILABLE = False

STATIC_DIR = Path(__file__).parent / "web_static"

#: Uploads above this size are rejected before anything touches the disk.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

#: Finished jobs are kept only so the page can poll for them once.
MAX_JOBS = 32


class WebExtraMissing(RuntimeError):
    """Raised when the [web] extra is not installed."""

    MESSAGE = (
        "The web UI needs FastAPI and uvicorn, which are not installed.\n"
        "Install them with: pip install 'chord-key-analyzer[web]'"
    )

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)


@dataclass
class Job:
    """One analysis in flight or completed."""

    id: str
    status: str = "pending"  # pending | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "progress": round(self.progress, 3),
                "result": self.result,
                "error": self.error,
            }


class JobStore:
    """Bounded, thread-safe in-memory job table."""

    def __init__(self, capacity: int = MAX_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._capacity = capacity

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._capacity:
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


def _run_analysis(job: Job, target: str, options: dict[str, Any], cleanup: Path | None) -> None:
    """Worker body: analyse ``target`` and record the outcome on ``job``."""
    from .ingest import IngestError
    from .pipeline import analyze_source

    def progress(stage: str, fraction: float) -> None:
        with job.lock:
            job.stage = stage
            job.progress = fraction

    try:
        with job.lock:
            job.status = "running"
        result = analyze_source(target, progress=progress, **options)
        with job.lock:
            job.status = "done"
            job.stage = "done"
            job.progress = 1.0
            job.result = result.to_dict()
    except (IngestError, ValueError) as exc:
        with job.lock:
            job.status = "error"
            job.error = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        with job.lock:
            job.status = "error"
            job.error = f"Unexpected failure: {exc}"
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def create_app(allow_urls: bool = True):
    """Build the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise WebExtraMissing()

    app = FastAPI(title="chord-key-analyzer", version="0.1.0", docs_url=None, redoc_url=None)
    jobs = JobStore()
    app.state.jobs = jobs

    def _options(triads_only: bool, engine: str) -> dict[str, Any]:
        return {"engine": engine, "triads_only": triads_only}

    @app.get("/")
    def index() -> Any:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        from . import __version__

        return {"status": "ok", "version": __version__}

    @app.post("/analyze")
    async def analyze(
        background: BackgroundTasks,
        file: UploadFile | None = File(default=None),
        url: str | None = Form(default=None),
        triads_only: bool = Form(default=False),
        engine: str = Form(default="template"),
    ) -> Any:
        if file is None and not url:
            raise HTTPException(status_code=400, detail="Provide either a file or a url.")
        if file is not None and url:
            raise HTTPException(status_code=400, detail="Provide a file or a url, not both.")

        job = jobs.create()

        if file is not None:
            temp_dir = Path(tempfile.mkdtemp(prefix="cka-web-"))
            suffix = Path(file.filename or "upload").suffix or ".audio"
            destination = temp_dir / f"upload{suffix}"

            written = 0
            with destination.open("wb") as handle:
                while chunk := await file.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        handle.close()
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                        )
                    handle.write(chunk)

            if written == 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            background.add_task(
                _run_analysis, job, str(destination), _options(triads_only, engine), temp_dir
            )
        else:
            if not allow_urls:
                raise HTTPException(status_code=403, detail="URL input is disabled on this server.")
            assert url is not None
            if not url.startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400, detail="URL must start with http:// or https://"
                )
            background.add_task(_run_analysis, job, url, _options(triads_only, engine), None)

        return JSONResponse({"job": job.id}, status_code=202)

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> Any:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        return job.to_dict()

    return app


def serve(host: str = "127.0.0.1", port: int = 8321, open_browser: bool = True) -> None:
    """Run the web UI with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise WebExtraMissing() from exc

    app = create_app()

    if open_browser:
        import webbrowser

        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{display_host}:{port}")).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
