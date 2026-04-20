"""Tests for TrainedGenerator."""
from __future__ import annotations

import numpy as np
import torch

from laser_ai.audio.features import FEATURE_DIM
from laser_ai.generator.trained import TrainedGenerator
from laser_ai.models.checkpoint import LaserAICheckpoint
from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
from laser_ai.models.vae import FrameVAE, FrameVAEConfig


def _make_checkpoint(n_points: int = 64) -> LaserAICheckpoint:
    vae_cfg = FrameVAEConfig(n_points=n_points, latent_dim=8, hidden=16)
    seq_cfg = SequencerConfig(
        feature_dim=FEATURE_DIM, latent_dim=8, hidden=32,
        n_layers=1, n_heads=2, max_len=256,
    )
    return LaserAICheckpoint(
        vae=FrameVAE(vae_cfg), vae_cfg=vae_cfg,
        sequencer=AudioToLatentSequencer(seq_cfg), seq_cfg=seq_cfg,
        audio_feature_dim=FEATURE_DIM, fps=30.0,
    )


def test_trained_generator_output_shape():
    ck = _make_checkpoint(n_points=64)
    gen = TrainedGenerator(ck)
    feats = np.random.rand(30, FEATURE_DIM).astype(np.float32)
    out = gen.generate(feats, n_points=64)
    assert out.shape == (30, 64, 6)


def test_trained_generator_output_in_valid_ranges():
    ck = _make_checkpoint(n_points=64)
    gen = TrainedGenerator(ck)
    feats = np.random.rand(20, FEATURE_DIM).astype(np.float32)
    out = gen.generate(feats, n_points=64)
    assert out[..., :2].max() <= 1.0 + 1e-5
    assert out[..., :2].min() >= -1.0 - 1e-5
    assert out[..., 2:5].min() >= 0.0 - 1e-5
    assert out[..., 2:5].max() <= 1.0 + 1e-5


def test_trained_generator_requires_matching_feature_dim():
    import pytest
    ck = _make_checkpoint()
    gen = TrainedGenerator(ck)
    bad = np.random.rand(10, FEATURE_DIM - 1).astype(np.float32)
    with pytest.raises(ValueError, match="feature_dim"):
        gen.generate(bad, n_points=64)


def test_trained_generator_overrides_n_points_warns_on_mismatch(recwarn):
    import warnings
    ck = _make_checkpoint(n_points=64)
    gen = TrainedGenerator(ck)
    feats = np.random.rand(5, FEATURE_DIM).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = gen.generate(feats, n_points=128)   # different from checkpoint
        assert any("n_points" in str(x.message) for x in w)
    # It still runs at the model's native n_points
    assert out.shape == (5, 64, 6)
