"""Smoke tests for sequencer training loop."""
from __future__ import annotations

import torch

from laser_ai.models.sequencer import SequencerConfig
from laser_ai.training.train_sequencer import SequencerTrainConfig, train_sequencer


def _synthetic_pairs(count: int, T: int, feat_dim: int, latent_dim: int):
    torch.manual_seed(0)
    return [
        (
            torch.randn(T, feat_dim),
            torch.randn(T, latent_dim),
        )
        for _ in range(count)
    ]


def test_train_sequencer_runs():
    pairs = _synthetic_pairs(count=3, T=32, feat_dim=16, latent_dim=8)
    seq_cfg = SequencerConfig(
        feature_dim=16, latent_dim=8, hidden=32, n_layers=2, n_heads=2,
    )
    train_cfg = SequencerTrainConfig(epochs=1, batch_size=1)
    model, history, latent_mean, latent_std = train_sequencer(
        pairs, seq_cfg=seq_cfg, train_cfg=train_cfg,
    )
    assert model is not None
    assert len(history) == 1
    assert latent_mean.shape == (8,)
    assert latent_std.shape == (8,)
    assert torch.all(latent_std > 0)


def test_train_sequencer_loss_decreases():
    pairs = _synthetic_pairs(count=4, T=32, feat_dim=16, latent_dim=8)
    seq_cfg = SequencerConfig(
        feature_dim=16, latent_dim=8, hidden=32, n_layers=2, n_heads=2,
    )
    train_cfg = SequencerTrainConfig(epochs=5, batch_size=2, lr=3e-3)
    _, history, _, _ = train_sequencer(pairs, seq_cfg=seq_cfg, train_cfg=train_cfg)
    assert history[-1]["mse"] < history[0]["mse"]
