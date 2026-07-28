"""CLI tests driven through click's runner."""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

from chord_key_analyzer import __version__
from chord_key_analyzer.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_the_commands(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output
    assert "web" in result.output


def test_analyze_help_documents_the_flags(runner):
    result = runner.invoke(main, ["analyze", "--help"])
    assert result.exit_code == 0
    for flag in ("--json", "--lab", "--start", "--duration", "--triads-only", "--engine",
                 "--no-sections"):
        assert flag in result.output


def test_web_help_documents_no_urls(runner):
    result = runner.invoke(main, ["web", "--help"])
    assert result.exit_code == 0
    assert "--no-urls" in result.output


def test_web_without_fastapi_prints_the_install_hint(runner, monkeypatch):
    """uvicorn present but FastAPI missing used to crash on the FastAPI names."""
    web_module = pytest.importorskip("chord_key_analyzer.web")
    monkeypatch.setattr(web_module, "FASTAPI_AVAILABLE", False)

    result = runner.invoke(main, ["web"])
    assert result.exit_code != 0
    assert "chord-key-analyzer[web]" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output


def test_web_without_uvicorn_prints_the_install_hint(runner, monkeypatch):
    pytest.importorskip("chord_key_analyzer.web")
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    result = runner.invoke(main, ["web"])
    assert result.exit_code != 0
    assert "chord-key-analyzer[web]" in result.output
    assert "Traceback" not in result.output


def test_eval_without_mir_eval_prints_the_install_hint_first(runner, monkeypatch, tmp_path):
    """The hint must land before discovery, not after a listing of tracks
    the run was never going to be able to score."""
    monkeypatch.setitem(sys.modules, "mir_eval", None)

    result = runner.invoke(main, ["eval", str(tmp_path)])
    assert result.exit_code != 0
    assert "chord-key-analyzer[eval]" in result.output
    assert "no evaluable tracks" not in result.output.lower()
    assert "Traceback" not in result.output


def test_the_url_install_hint_survives_rich_markup(runner, monkeypatch, tmp_path):
    """Regression: rich reads `[...]` in a markup string as a style tag.

    The hint went through `console.print(f"[red]Error:[/red] {exc}")`, so
    "pip install 'chord-key-analyzer[url]'" reached the user as
    "pip install 'chord-key-analyzer'" — with the one part they needed silently
    eaten.  Runtime strings are printed as Text now, not interpolated markup.
    """
    monkeypatch.setitem(sys.modules, "yt_dlp", None)

    result = runner.invoke(main, ["analyze", "https://example.invalid/song", "--quiet"])
    assert result.exit_code == 2
    assert "chord-key-analyzer[url]" in result.output


def test_a_bracketed_path_is_not_eaten_by_markup(runner, tmp_path):
    """Same failure mode, reached through a filename instead of a hint."""
    awkward = tmp_path / "album [remastered]"
    awkward.mkdir()

    result = runner.invoke(main, ["analyze", str(awkward / "nope.wav")])
    assert result.exit_code == 2
    # Soft-wrapped rather than mangled: rich breaks a long path across lines
    # without inserting anything, so dropping the newlines restores it.
    assert "album [remastered]" in result.output.replace("\n", "")


def test_analyze_renders_a_report(runner, pop_wav):
    result = runner.invoke(main, ["analyze", str(pop_wav)])
    assert result.exit_code == 0, result.output
    assert "C major" in result.output


def test_analyze_json_to_stdout(runner, pop_wav):
    """`--json -` must emit clean JSON: progress and reports go to stderr."""
    result = runner.invoke(main, ["analyze", str(pop_wav), "--json", "-", "--quiet"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["key"]["tonic"] == "C"
    assert payload["key"]["mode"] == "major"
    assert payload["schema"] == 1


def test_analyze_json_to_a_file(runner, pop_wav, tmp_path):
    destination = tmp_path / "result.json"
    result = runner.invoke(
        main, ["analyze", str(pop_wav), "--json", str(destination), "--quiet"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(destination.read_text())
    assert [c["label"] for c in payload["chords"]][:4] == [
        "C:maj",
        "G:maj",
        "A:min",
        "F:maj",
    ]


def test_analyze_lab_export(runner, pop_wav, tmp_path):
    destination = tmp_path / "result.lab"
    result = runner.invoke(main, ["analyze", str(pop_wav), "--lab", str(destination), "--quiet"])
    assert result.exit_code == 0, result.output
    first = destination.read_text().splitlines()[0].split("\t")
    assert len(first) == 3
    assert first[2] == "C:maj"


def test_analyze_start_and_duration(runner, pop_wav, tmp_path):
    destination = tmp_path / "excerpt.json"
    result = runner.invoke(
        main,
        ["analyze", str(pop_wav), "--start", "2", "--duration", "4", "--json",
         str(destination), "--quiet"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text())["duration"] == pytest.approx(4.0, abs=0.1)


def test_analyze_triads_only(runner, pop_wav, tmp_path):
    destination = tmp_path / "triads.json"
    runner.invoke(
        main, ["analyze", str(pop_wav), "--triads-only", "--json", str(destination), "--quiet"]
    )
    payload = json.loads(destination.read_text())
    assert payload["meta"]["triads_only"] is True
    for chord in payload["chords"]:
        if chord["label"] != "N":
            assert chord["label"].split(":")[1] in {"maj", "min", "dim", "aug"}


def test_analyze_missing_file_exits_with_an_error(runner, tmp_path):
    result = runner.invoke(main, ["analyze", str(tmp_path / "nope.wav")])
    assert result.exit_code == 2
    assert "not found" in result.output.lower()


def test_analyze_reports_an_unwritable_output_path(runner, pop_wav, tmp_path):
    """An unwritable --json must be a clean error, not a traceback."""
    result = runner.invoke(main, ["analyze", str(pop_wav), "--json", str(tmp_path), "--quiet"])
    assert result.exit_code == 2
    assert "cannot write output" in result.output.lower()
    assert "Traceback" not in result.output


def test_analyze_rejects_a_negative_start(runner, pop_wav):
    result = runner.invoke(main, ["analyze", str(pop_wav), "--start", "-5"])
    assert result.exit_code != 0


def test_analyze_rejects_a_zero_duration(runner, pop_wav):
    result = runner.invoke(main, ["analyze", str(pop_wav), "--duration", "0"])
    assert result.exit_code != 0


def test_analyze_rejects_an_unknown_engine(runner, pop_wav):
    result = runner.invoke(main, ["analyze", str(pop_wav), "--engine", "crema"])
    assert result.exit_code != 0


def test_no_hpss_still_works(runner, pop_wav, tmp_path):
    destination = tmp_path / "nohpss.json"
    result = runner.invoke(
        main, ["analyze", str(pop_wav), "--no-hpss", "--json", str(destination), "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text())["key"]["tonic"] == "C"


def test_no_sections_flag_skips_the_scan(runner, pop_wav, tmp_path):
    destination = tmp_path / "nosections.json"
    result = runner.invoke(
        main, ["analyze", str(pop_wav), "--no-sections", "--json", str(destination), "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text())["sections"] == []


def test_no_modulations_flag_skips_the_scan(runner, pop_wav, tmp_path):
    destination = tmp_path / "nomod.json"
    runner.invoke(
        main,
        ["analyze", str(pop_wav), "--no-modulations", "--json", str(destination), "--quiet"],
    )
    assert json.loads(destination.read_text())["key"]["modulations"] == []


def test_quiet_suppresses_the_report(runner, pop_wav):
    result = runner.invoke(main, ["analyze", str(pop_wav), "--quiet"])
    assert result.exit_code == 0
    assert result.output.strip() == ""
