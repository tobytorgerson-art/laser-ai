"""Tests for VAE loss components."""
from __future__ import annotations

import torch

from laser_ai.models.losses import LossWeights, chamfer_distance, vae_loss


def test_chamfer_of_identical_sets_is_zero():
    x = torch.tensor([[[-1.0, 0.0], [0.0, 0.5], [1.0, 0.5]]])
    assert chamfer_distance(x, x).item() < 1e-6


def test_chamfer_is_positive_for_different_sets():
    a = torch.tensor([[[0.0, 0.0]]])
    b = torch.tensor([[[1.0, 0.0]]])
    assert chamfer_distance(a, b).item() > 0.9  # square-distance-style


def test_chamfer_supports_batches():
    a = torch.randn(3, 8, 2)
    b = torch.randn(3, 8, 2)
    out = chamfer_distance(a, b)
    assert out.shape == ()
    assert out.item() > 0


def test_vae_loss_is_positive_with_random_inputs():
    B, N = 2, 32
    recon = torch.randn(B, N, 6).clamp(-1, 1)
    target = torch.randn(B, N, 6).clamp(-1, 1)
    recon = (recon + 1) / 2  # pretend some in [0,1]
    target = (target + 1) / 2
    recon[..., :2] = recon[..., :2] * 2 - 1
    target[..., :2] = target[..., :2] * 2 - 1
    mu = torch.zeros(B, 8)
    logvar = torch.zeros(B, 8)
    total, parts = vae_loss(recon, target, mu, logvar, LossWeights())
    assert total.item() > 0
    assert set(parts.keys()) == {"chamfer", "rgb", "travel", "kl"}


def test_vae_loss_reduces_when_recon_matches_target():
    B, N = 2, 16
    target = torch.rand(B, N, 6)
    target[..., :2] = target[..., :2] * 2 - 1
    mu = torch.zeros(B, 8)
    logvar = torch.zeros(B, 8)
    total_exact, _ = vae_loss(target.clone(), target, mu, logvar, LossWeights())
    total_random, _ = vae_loss(torch.randn_like(target), target, mu, logvar, LossWeights())
    assert total_exact.item() < total_random.item()


def test_kl_weight_zero_drops_kl_term():
    B, N = 1, 8
    target = torch.rand(B, N, 6)
    target[..., :2] = target[..., :2] * 2 - 1
    mu = torch.randn(B, 4) * 3
    logvar = torch.randn(B, 4) * 3
    _, parts_on = vae_loss(target, target, mu, logvar, LossWeights(kl=1.0))
    _, parts_off = vae_loss(target, target, mu, logvar, LossWeights(kl=0.0))
    assert parts_on["kl"] > parts_off["kl"]
    assert parts_off["kl"] == 0.0
