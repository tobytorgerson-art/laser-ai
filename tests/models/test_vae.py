"""Tests for Frame VAE."""
from __future__ import annotations

import torch

from laser_ai.models.vae import FrameVAE, FrameVAEConfig


def test_vae_encode_returns_mu_and_logvar_of_correct_shape():
    cfg = FrameVAEConfig(n_points=64, latent_dim=16, hidden=32)
    vae = FrameVAE(cfg).eval()
    x = torch.randn(4, 64, 6)
    mu, logvar = vae.encode(x)
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_vae_decode_returns_frame_shape():
    cfg = FrameVAEConfig(n_points=64, latent_dim=16, hidden=32)
    vae = FrameVAE(cfg).eval()
    z = torch.randn(4, 16)
    out = vae.decode(z)
    assert out.shape == (4, 64, 6)


def test_vae_forward_returns_recon_mu_logvar():
    cfg = FrameVAEConfig(n_points=64, latent_dim=16, hidden=32)
    vae = FrameVAE(cfg).eval()
    x = torch.randn(2, 64, 6)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)


def test_vae_decode_outputs_stay_bounded():
    """xy in [-1, 1] via tanh, rgb in [0, 1] via sigmoid, is_travel in [0, 1]."""
    cfg = FrameVAEConfig(n_points=32, latent_dim=8, hidden=16)
    vae = FrameVAE(cfg).eval()
    out = vae.decode(torch.randn(8, 8) * 10.0)
    assert out[..., :2].abs().max() <= 1.0 + 1e-5
    assert out[..., 2:5].min() >= 0.0
    assert out[..., 2:5].max() <= 1.0 + 1e-5
    assert out[..., 5].min() >= 0.0
    assert out[..., 5].max() <= 1.0 + 1e-5


def test_vae_reparameterize_is_stochastic_in_train_deterministic_in_eval():
    cfg = FrameVAEConfig(n_points=32, latent_dim=8, hidden=16)
    vae = FrameVAE(cfg)
    mu = torch.zeros(2, 8)
    logvar = torch.zeros(2, 8)

    vae.train()
    z1 = vae.reparameterize(mu, logvar)
    z2 = vae.reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2)

    vae.eval()
    z3 = vae.reparameterize(mu, logvar)
    z4 = vae.reparameterize(mu, logvar)
    torch.testing.assert_close(z3, z4)


def test_vae_param_count_is_small():
    cfg = FrameVAEConfig()  # defaults: n_points=512, latent_dim=64, hidden=128
    vae = FrameVAE(cfg)
    n = sum(p.numel() for p in vae.parameters())
    # Should be in the ~1–3M param range per spec
    assert 500_000 < n < 5_000_000
