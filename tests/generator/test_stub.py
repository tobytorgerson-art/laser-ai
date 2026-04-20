"""Tests for the rule-based stub generator."""
from __future__ import annotations

import numpy as np

from laser_ai.audio.features import FEATURE_DIM
from laser_ai.generator.stub import StubGenerator


def _random_features(T: int = 60, rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.random((T, FEATURE_DIM), dtype=np.float32)


def test_stub_output_shape():
    feats = _random_features(T=60)
    gen = StubGenerator()
    out = gen.generate(feats, n_points=512)
    assert out.shape == (60, 512, 6)


def test_stub_output_is_in_valid_ranges():
    feats = _random_features(T=30)
    gen = StubGenerator()
    out = gen.generate(feats, n_points=256)
    # x, y in [-1, 1]
    assert out[..., 0].min() >= -1.0 - 1e-6
    assert out[..., 0].max() <= 1.0 + 1e-6
    assert out[..., 1].min() >= -1.0 - 1e-6
    assert out[..., 1].max() <= 1.0 + 1e-6
    # colors in [0, 1]
    assert out[..., 2:5].min() >= 0.0 - 1e-6
    assert out[..., 2:5].max() <= 1.0 + 1e-6
    # is_blank in {0, 1}
    assert set(np.unique(out[..., 5]).tolist()).issubset({0.0, 1.0})


def test_stub_is_deterministic_with_fixed_seed():
    feats = _random_features(T=30)
    gen1 = StubGenerator(seed=42)
    gen2 = StubGenerator(seed=42)
    np.testing.assert_allclose(gen1.generate(feats), gen2.generate(feats))


def test_stub_reacts_to_energy_difference():
    feats_quiet = np.zeros((30, FEATURE_DIM), dtype=np.float32)
    feats_loud = np.zeros((30, FEATURE_DIM), dtype=np.float32)
    # set RMS feature (second-to-last before onset+beat_phase)
    rms_idx = 128 + 12 + 3
    feats_loud[:, rms_idx] = 1.0
    gen = StubGenerator(seed=0)
    quiet = gen.generate(feats_quiet)
    loud = gen.generate(feats_loud)
    # Loud frames should have larger shapes (higher x/y span)
    quiet_extent = quiet[..., :2].std()
    loud_extent = loud[..., :2].std()
    assert loud_extent > quiet_extent
