"""Tests for audio feature extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from laser_ai.audio.features import FEATURE_DIM, extract_features
from laser_ai.audio.loader import load_audio


def test_extract_returns_2d_array(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats = extract_features(samples, sr, fps=30.0)
    assert feats.ndim == 2


def test_extract_matches_expected_frame_count(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats = extract_features(samples, sr, fps=30.0)
    # 1-second audio at 30 fps → 30 frames (± 1 for boundary rounding)
    assert abs(feats.shape[0] - 30) <= 1


def test_extract_feature_dim_matches_constant(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats = extract_features(samples, sr, fps=30.0)
    assert feats.shape[1] == FEATURE_DIM


def test_extract_is_finite(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats = extract_features(samples, sr, fps=30.0)
    assert np.all(np.isfinite(feats))


def test_extract_is_deterministic(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats1 = extract_features(samples, sr, fps=30.0)
    feats2 = extract_features(samples, sr, fps=30.0)
    np.testing.assert_allclose(feats1, feats2)


def test_extract_handles_multi_beat_audio():
    """Longer percussive-ish audio triggers the beat-phase extrapolation branch."""
    sr = 44100
    duration_s = 4.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # 2 Hz click train (one click every 0.5s) with a decaying envelope per click
    clicks = np.zeros_like(t)
    for click_t in np.arange(0.0, duration_s, 0.5):
        start = int(click_t * sr)
        end = min(start + 2000, len(clicks))
        env = np.exp(-np.linspace(0, 8, end - start))
        clicks[start:end] += env * np.sin(2 * np.pi * 440 * np.linspace(0, (end - start) / sr, end - start))
    samples = clicks.astype(np.float32)

    feats = extract_features(samples, sr, fps=30.0)
    # 4 sec at 30 fps → ~120 frames
    assert abs(feats.shape[0] - 120) <= 2
    # beat_phase column should not be all zero — some frames are after the last detected beat
    beat_phase_col = feats[:, -1]
    assert beat_phase_col.max() > 0.0
