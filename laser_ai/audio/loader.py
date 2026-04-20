"""Load audio files to mono float32 at 44.1 kHz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SR = 44100


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as (samples, sr). Samples are mono float32.

    Resampled to 44.1 kHz if the source differs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Mix to mono
    samples = samples.mean(axis=1)

    if sr != TARGET_SR:
        # lazy import; librosa is heavy
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return samples.astype(np.float32, copy=False), sr
