"""Tests for saving/loading model bundles."""
from __future__ import annotations

from pathlib import Path

import torch

from laser_ai.models.checkpoint import LaserAICheckpoint, load_checkpoint, save_checkpoint
from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
from laser_ai.models.vae import FrameVAE, FrameVAEConfig


def _build_tiny_bundle() -> LaserAICheckpoint:
    vae_cfg = FrameVAEConfig(n_points=32, latent_dim=4, hidden=8)
    seq_cfg = SequencerConfig(feature_dim=8, latent_dim=4, hidden=16,
                              n_layers=1, n_heads=2, max_len=64)
    return LaserAICheckpoint(
        vae=FrameVAE(vae_cfg),
        vae_cfg=vae_cfg,
        sequencer=AudioToLatentSequencer(seq_cfg),
        seq_cfg=seq_cfg,
        audio_feature_dim=8,
        fps=30.0,
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    ck = _build_tiny_bundle()
    path = tmp_path / "bundle.pt"
    save_checkpoint(ck, path)
    assert path.exists()

    ck2 = load_checkpoint(path, map_location="cpu")
    assert ck2.vae_cfg.latent_dim == ck.vae_cfg.latent_dim
    assert ck2.seq_cfg.feature_dim == ck.seq_cfg.feature_dim
    assert ck2.audio_feature_dim == 8
    assert ck2.fps == 30.0


def test_latent_normalization_stats_roundtrip(tmp_path: Path):
    ck = _build_tiny_bundle()
    ck.latent_mean = torch.arange(4, dtype=torch.float32) * 0.1
    ck.latent_std = torch.arange(4, dtype=torch.float32) * 0.1 + 1.0
    path = tmp_path / "b.pt"
    save_checkpoint(ck, path)
    ck2 = load_checkpoint(path, map_location="cpu")
    torch.testing.assert_close(ck2.latent_mean, ck.latent_mean)
    torch.testing.assert_close(ck2.latent_std, ck.latent_std)


def test_missing_latent_stats_load_as_none(tmp_path: Path):
    ck = _build_tiny_bundle()  # no mean/std set
    path = tmp_path / "b.pt"
    save_checkpoint(ck, path)
    ck2 = load_checkpoint(path, map_location="cpu")
    assert ck2.latent_mean is None
    assert ck2.latent_std is None


def test_loaded_vae_produces_same_output(tmp_path: Path):
    ck = _build_tiny_bundle()
    x = torch.randn(1, ck.vae_cfg.n_points, 6)
    ck.vae.eval()
    y_orig = ck.vae.decode(ck.vae.encode(x)[0])

    path = tmp_path / "b.pt"
    save_checkpoint(ck, path)
    ck2 = load_checkpoint(path, map_location="cpu")
    ck2.vae.eval()
    y_loaded = ck2.vae.decode(ck2.vae.encode(x)[0])

    torch.testing.assert_close(y_orig, y_loaded, atol=1e-6, rtol=1e-6)
