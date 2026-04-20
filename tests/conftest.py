"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def tiny_wav_path(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "tiny_audio.wav"
    if not path.exists():
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        sr = 44100
        duration_s = 1.0
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(path, samples, sr, subtype="PCM_16")
    return path
