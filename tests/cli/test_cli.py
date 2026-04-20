"""CLI smoke tests."""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from laser_ai.cli import cli


def test_cli_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "info" in result.output


def test_generate_command_writes_ilda_file(tiny_wav_path: Path, tmp_path: Path):
    out = tmp_path / "out.ilda"
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", str(tiny_wav_path), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 32  # at least one header


def test_info_command_on_fixture():
    fixture = Path(__file__).parent.parent / "fixtures" / "tiny_show.ilda"
    runner = CliRunner()
    result = runner.invoke(cli, ["info", str(fixture)])
    assert result.exit_code == 0, result.output
    assert "frames" in result.output.lower()
