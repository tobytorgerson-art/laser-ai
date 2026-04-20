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


def test_load_resamples_non_44k_audio(tmp_path: Path):
    import soundfile as sf
    src_sr = 22050
    duration_s = 0.5
    t = np.linspace(0, duration_s, int(src_sr * duration_s), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "half_rate.wav"
    sf.write(path, samples, src_sr, subtype="PCM_16")

    out, sr = load_audio(path)
    assert sr == 44100
    # Resampled length should be ~2x original
    assert abs(len(out) - int(duration_s * 44100)) < 100
