"""Tests for audio loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from laser_ai.audio.loader import load_audio


def test_load_returns_mono_float32(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert sr == 44100


def test_load_preserves_duration(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    # Fixture is 1-second 44.1k
    assert abs(len(samples) - sr) < 10


def test_load_missing_file_raises(tmp_path: Path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nope.wav")
