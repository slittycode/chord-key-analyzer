"""Web UI tests via FastAPI's TestClient."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="requires the [web] extra")
from fastapi.testclient import TestClient  # noqa: E402

from chord_key_analyzer.web import STATIC_DIR, JobStore, create_app  # noqa: E402


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


def test_job_store_is_bounded():
    store = JobStore(capacity=3)
    created = [store.create() for _ in range(5)]
    assert store.get(created[0].id) is None
    assert store.get(created[-1].id) is not None


def test_job_store_ids_are_unique():
    store = JobStore()
    ids = {store.create().id for _ in range(20)}
    assert len(ids) == 20
