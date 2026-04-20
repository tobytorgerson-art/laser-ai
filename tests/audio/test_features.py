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
