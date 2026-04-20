"""Tests for Audio-to-Latent sequencer."""
from __future__ import annotations

import torch

from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig


def test_sequencer_forward_shape():
    cfg = SequencerConfig(
        feature_dim=146, latent_dim=32, hidden=128, n_layers=2, n_heads=4,
    )
    m = AudioToLatentSequencer(cfg).eval()
    x = torch.randn(2, 60, 146)
    y = m(x)
    assert y.shape == (2, 60, 32)


def test_sequencer_is_causal():
    """Output at time t must not depend on inputs at time > t."""
    cfg = SequencerConfig(feature_dim=8, latent_dim=4, hidden=16, n_layers=2, n_heads=2)
    m = AudioToLatentSequencer(cfg).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 10, 8)
    y1 = m(x)
    # Perturb only the last half of the input
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])
    y2 = m(x2)
    # Outputs before the perturbation must remain unchanged
    torch.testing.assert_close(y1[:, :5], y2[:, :5], atol=1e-5, rtol=1e-5)
    # Outputs after the perturbation must change
    assert not torch.allclose(y1[:, 5:], y2[:, 5:], atol=1e-5)


def test_sequencer_param_count_in_target_range():
    cfg = SequencerConfig()  # defaults: ~2M params
    m = AudioToLatentSequencer(cfg)
    n = sum(p.numel() for p in m.parameters())
    # Spec says ~2M; allow 0.5M - 5M window
    assert 500_000 < n < 5_000_000
