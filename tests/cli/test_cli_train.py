"""Tests for new ML-related CLI subcommands."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from laser_ai.cli import cli


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _write_wav(path: Path, duration_s: float = 1.0) -> None:
    import soundfile as sf
    sr = 44100
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    s = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    sf.write(path, s, sr, subtype="PCM_16")


def test_train_vae_creates_checkpoint(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    # Re-use the fixture ILDA twice to give the VAE something to train on
    import shutil
    shutil.copy(FIXTURES / "tiny_show.ilda", data / "a.ilda")
    shutil.copy(FIXTURES / "tiny_show.ilda", data / "b.ilda")

    out_ck = tmp_path / "ck.pt"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "train-vae", str(data), "-o", str(out_ck),
        "--epochs", "1", "--n-points", "32",
        "--latent-dim", "4", "--hidden", "8",
    ])
    assert result.exit_code == 0, result.output
    assert out_ck.exists()


def test_generate_with_model_works(tmp_path: Path):
    # Build a checkpoint, then use it to generate
    import shutil
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(FIXTURES / "tiny_show.ilda", data / "a.ilda")

    ck = tmp_path / "ck.pt"
    runner = CliRunner()
    r1 = runner.invoke(cli, [
        "train-vae", str(data), "-o", str(ck),
        "--epochs", "1", "--n-points", "32",
        "--latent-dim", "4", "--hidden", "8",
    ])
    assert r1.exit_code == 0, r1.output

    song = tmp_path / "song.wav"
    _write_wav(song)
    out = tmp_path / "out.ilda"
    r2 = runner.invoke(cli, [
        "generate", str(song), "-o", str(out), "--model", str(ck),
    ])
    assert r2.exit_code == 0, r2.output
    assert out.exists()
