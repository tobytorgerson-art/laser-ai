"""Tests for dataset bundling."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from laser_ai.bundle.pack import pack_dataset


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _write_wav(path: Path) -> None:
    import numpy as np
    import soundfile as sf
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    s = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    sf.write(path, s, sr, subtype="PCM_16")


def test_pack_creates_zip_with_expected_entries(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    _write_wav(data / "song1.wav")
    import shutil
    shutil.copy(FIXTURES / "tiny_show.ilda", data / "song1.ilda")

    out = tmp_path / "bundle.zip"
    pack_dataset(data, out)

    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "index.json" in names
        assert any(n.endswith("song1.wav") for n in names)
        assert any(n.endswith("song1.ilda") for n in names)
        with zf.open("index.json") as f:
            idx = json.load(f)
        assert idx["version"] >= 1
        assert len(idx["pairs"]) == 1
        assert idx["pairs"][0]["stem"] == "song1"


def test_pack_rejects_empty_dataset(tmp_path: Path):
    import pytest
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no audio"):
        pack_dataset(empty, tmp_path / "out.zip")
