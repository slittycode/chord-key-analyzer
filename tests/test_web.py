"""Web UI tests via FastAPI's TestClient."""

from __future__ import annotations

import socket
import sys
import types
from pathlib import Path

import pytest

import fixtures as fx

fastapi = pytest.importorskip("fastapi", reason="requires the [web] extra")
from fastapi.testclient import TestClient  # noqa: E402

from chord_key_analyzer import web as web_module  # noqa: E402
from chord_key_analyzer.web import (  # noqa: E402
    STATIC_DIR,
    JobStore,
    create_app,
    url_rejection_reason,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "chord-key-analyzer" in response.text


def test_static_page_is_self_contained():
    """No CDN, no build step — the page must not reach out to the network."""
    html = (STATIC_DIR / "index.html").read_text()
    assert "<script" in html
    assert "src=\"http" not in html
    assert "cdn." not in html


def test_health(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"


def test_analyze_an_uploaded_file(client, pop_wav):
    """The endpoint runs the same pipeline as the CLI, so it must agree with it."""
    with open(pop_wav, "rb") as handle:
        response = client.post(
            "/analyze", files={"file": ("pop.wav", handle, "audio/wav")}
        )
    assert response.status_code == 202
    job_id = response.json()["job"]

    # TestClient runs background tasks before returning, so the job is finished.
    status = client.get(f"/jobs/{job_id}").json()
    assert status["status"] == "done", status.get("error")

    result = status["result"]
    assert result["key"]["tonic"] == "C"
    assert result["key"]["mode"] == "major"
    assert [c["label"] for c in result["chords"]][:4] == ["C:maj", "G:maj", "A:min", "F:maj"]
    assert result["progression"]["main_loop"]["roman"] == ["I", "V", "vi", "IV"]


def test_web_and_cli_agree(client, pop_wav, pop_result):
    with open(pop_wav, "rb") as handle:
        response = client.post("/analyze", files={"file": ("pop.wav", handle, "audio/wav")})
    job = client.get(f"/jobs/{response.json()['job']}").json()
    web_result = job["result"]

    assert web_result["key"]["tonic"] == pop_result.key.tonic
    assert web_result["key"]["mode"] == pop_result.key.mode
    assert [c["label"] for c in web_result["chords"]] == [c.label for c in pop_result.chords]


def test_triads_only_is_honoured(client, pop_wav):
    with open(pop_wav, "rb") as handle:
        response = client.post(
            "/analyze",
            files={"file": ("pop.wav", handle, "audio/wav")},
            data={"triads_only": "true"},
        )
    job = client.get(f"/jobs/{response.json()['job']}").json()
    assert job["result"]["meta"]["triads_only"] is True


def test_analyze_requires_an_input(client):
    response = client.post("/analyze")
    assert response.status_code == 400
    assert "either a file or a url" in response.json()["detail"].lower()


def test_analyze_rejects_both_inputs(client, pop_wav):
    with open(pop_wav, "rb") as handle:
        response = client.post(
            "/analyze",
            files={"file": ("pop.wav", handle, "audio/wav")},
            data={"url": "https://example.com/song"},
        )
    assert response.status_code == 400


def test_analyze_rejects_an_empty_upload(client):
    response = client.post("/analyze", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 400


def test_analyze_rejects_a_non_http_url(client):
    response = client.post("/analyze", data={"url": "file:///etc/passwd"})
    assert response.status_code == 400


def test_url_input_can_be_disabled():
    disabled = TestClient(create_app(allow_urls=False))
    response = disabled.post("/analyze", data={"url": "https://example.com/song"})
    assert response.status_code == 403


def test_undecodable_upload_reports_an_error(client):
    response = client.post(
        "/analyze", files={"file": ("junk.wav", b"not audio at all", "audio/wav")}
    )
    assert response.status_code == 202
    job = client.get(f"/jobs/{response.json()['job']}").json()
    assert job["status"] == "error"
    assert job["error"]


def test_unknown_job_is_a_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/song",
        "http://10.0.0.5/song",
        "http://192.168.1.10/song",
        "http://172.16.0.1/song",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://[::1]/song",
        "http://[::ffff:10.0.0.1]/song",  # private v4 wearing an IPv6 hat
        "http://[::]/song",
        "http://0.0.0.0/song",
    ],
)
def test_guard_rejects_non_public_addresses(url):
    """IP literals are judged without touching DNS, so this test needs no network."""
    assert url_rejection_reason(url) is not None


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/song", "http:///nohost", "not-a-url"],
)
def test_guard_rejects_bad_schemes_and_missing_hosts(url):
    assert url_rejection_reason(url) is not None


def test_guard_accepts_a_public_ip_literal():
    assert url_rejection_reason("https://93.184.216.34/song") is None


def _fake_getaddrinfo(address: str):
    return lambda *args, **kwargs: [(socket.AF_INET, None, None, "", (address, 80))]


def test_guard_accepts_a_hostname_resolving_to_a_public_address(monkeypatch):
    monkeypatch.setattr(web_module.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34"))
    assert url_rejection_reason("https://example.com/song") is None


def test_guard_rejects_a_hostname_resolving_to_a_private_address(monkeypatch):
    """The DNS-rebinding shape: a public name pointing at an internal host."""
    monkeypatch.setattr(web_module.socket, "getaddrinfo", _fake_getaddrinfo("10.1.2.3"))
    reason = url_rejection_reason("https://internal.example.com/song")
    assert reason is not None and "non-public" in reason


def test_guard_rejects_an_unresolvable_host(monkeypatch):
    def boom(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(web_module.socket, "getaddrinfo", boom)
    reason = url_rejection_reason("https://nope.invalid/song")
    assert reason is not None and "resolve" in reason


def test_analyze_rejects_a_private_url(client):
    response = client.post("/analyze", data={"url": "http://169.254.169.254/latest/meta-data/"})
    assert response.status_code == 400
    assert "non-public" in response.json()["detail"]


def test_url_job_runs_and_leaves_no_temp_dir(monkeypatch, tmp_path):
    """End-to-end web URL analysis: it succeeds and leaves nothing on disk."""
    recorded: list[Path] = []

    class RecordingYoutubeDL:
        def __init__(self, options):
            recorded.append(Path(options["outtmpl"]).parent)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            audio = fx.render_progression(fx.POP_LOOP_C, 1.0, 2)
            fx.write_wav(recorded[-1] / "song.wav", audio)
            return {"id": "song", "title": "Song"}

    from chord_key_analyzer import ingest

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=RecordingYoutubeDL))
    monkeypatch.setattr(ingest, "_require_ffmpeg", lambda reason: "/usr/bin/ffmpeg")
    monkeypatch.setattr(web_module, "url_rejection_reason", lambda url: None)

    local = TestClient(create_app())
    response = local.post("/analyze", data={"url": "https://example.com/song"})
    assert response.status_code == 202

    job = local.get(f"/jobs/{response.json()['job']}").json()
    assert job["status"] == "done", job.get("error")
    assert len(recorded) == 1
    assert not recorded[0].exists(), f"download dir {recorded[0]} was left behind"


def test_serve_disables_urls_off_loopback(monkeypatch):
    """The bind decides the default, and an explicit --no-urls still wins."""
    seen: list[bool] = []

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))

    def fake_create_app(allow_urls: bool = True):
        seen.append(allow_urls)
        return object()

    monkeypatch.setattr(web_module, "create_app", fake_create_app)

    web_module.serve(host="0.0.0.0", open_browser=False)
    web_module.serve(host="127.0.0.1", open_browser=False)
    web_module.serve(host="127.0.0.1", open_browser=False, allow_urls=False)

    assert seen == [False, True, False]


def test_job_store_is_bounded():
    store = JobStore(capacity=3)
    created = [store.create() for _ in range(5)]
    assert store.get(created[0].id) is None
    assert store.get(created[-1].id) is not None


def test_job_store_ids_are_unique():
    store = JobStore()
    ids = {store.create().id for _ in range(20)}
    assert len(ids) == 20
