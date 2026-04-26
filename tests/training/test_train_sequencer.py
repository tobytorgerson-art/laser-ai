"""Smoke tests for sequencer training loop."""
from __future__ import annotations

import torch

from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
from laser_ai.models.vae import FrameVAE, FrameVAEConfig
from laser_ai.training.train_sequencer import (
    SequencerE2ETrainConfig,
    SequencerTrainConfig,
    train_sequencer,
    train_sequencer_e2e,
)


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


def _synthetic_e2e_pairs(count: int, T: int, feat_dim: int, n_points: int):
    torch.manual_seed(0)
    return [
        (
            torch.randn(T, feat_dim),
            torch.cat([
                torch.rand(T, n_points, 2) * 2 - 1,    # xy in [-1, 1]
                torch.rand(T, n_points, 3),            # rgb in [0, 1]
                torch.rand(T, n_points, 1),            # travel in [0, 1]
            ], dim=-1),
        )
        for _ in range(count)
    ]


def test_train_sequencer_e2e_runs():
    n_points = 32
    latent_dim = 4
    feat_dim = 8
    vae = FrameVAE(FrameVAEConfig(n_points=n_points, latent_dim=latent_dim, hidden=8))
    seq_cfg = SequencerConfig(
        feature_dim=feat_dim, latent_dim=latent_dim, hidden=16,
        n_layers=1, n_heads=2, max_len=64,
    )
    sequencer = AudioToLatentSequencer(seq_cfg)
    pairs = _synthetic_e2e_pairs(count=2, T=32, feat_dim=feat_dim, n_points=n_points)
    latent_mean = torch.zeros(latent_dim)
    latent_std = torch.ones(latent_dim)
    cfg = SequencerE2ETrainConfig(
        epochs=1, batch_size=1, window=16, samples_per_epoch=2, lr=1e-3,
    )
    model, history = train_sequencer_e2e(
        pairs, sequencer=sequencer, vae=vae,
        latent_mean=latent_mean, latent_std=latent_std, train_cfg=cfg,
    )
    assert model is not None
    assert len(history) == 1
    assert "chamfer" in history[0] and "rgb" in history[0] and "travel" in history[0]


def test_train_sequencer_e2e_freezes_vae():
    n_points = 32
    latent_dim = 4
    feat_dim = 8
    vae = FrameVAE(FrameVAEConfig(n_points=n_points, latent_dim=latent_dim, hidden=8))
    sequencer = AudioToLatentSequencer(SequencerConfig(
        feature_dim=feat_dim, latent_dim=latent_dim, hidden=16,
        n_layers=1, n_heads=2, max_len=64,
    ))
    vae_state_before = {k: v.clone() for k, v in vae.state_dict().items()}
    pairs = _synthetic_e2e_pairs(count=2, T=32, feat_dim=feat_dim, n_points=n_points)
    cfg = SequencerE2ETrainConfig(
        epochs=1, batch_size=1, window=16, samples_per_epoch=2, lr=1e-3,
    )
    train_sequencer_e2e(
        pairs, sequencer=sequencer, vae=vae,
        latent_mean=torch.zeros(latent_dim), latent_std=torch.ones(latent_dim),
        train_cfg=cfg,
    )
    for k, v_before in vae_state_before.items():
        torch.testing.assert_close(vae.state_dict()[k], v_before)
