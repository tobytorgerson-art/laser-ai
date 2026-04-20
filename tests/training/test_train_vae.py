"""Smoke tests for the VAE training loop."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from laser_ai.models.vae import FrameVAEConfig
from laser_ai.training.train_vae import VAETrainConfig, train_vae


class _SyntheticFrameDS(Dataset):
    """A tiny hand-built dataset: random (N, 6) tensors."""

    def __init__(self, count: int, n_points: int) -> None:
        torch.manual_seed(0)
        self._data = torch.rand(count, n_points, 6) * 2 - 1
        self._data[..., 2:5] = (self._data[..., 2:5] + 1) / 2
        self._data[..., 5] = (self._data[..., 5] > 0).float()

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


def test_train_vae_runs_and_returns_model():
    ds = _SyntheticFrameDS(count=16, n_points=32)
    vae_cfg = FrameVAEConfig(n_points=32, latent_dim=8, hidden=16)
    train_cfg = VAETrainConfig(epochs=1, batch_size=4, lr=1e-3)
    model, history = train_vae(ds, vae_cfg=vae_cfg, train_cfg=train_cfg)
    assert model is not None
    assert len(history) >= 1


def test_train_vae_loss_decreases_over_a_few_epochs():
    ds = _SyntheticFrameDS(count=32, n_points=32)
    vae_cfg = FrameVAEConfig(n_points=32, latent_dim=8, hidden=16)
    train_cfg = VAETrainConfig(epochs=5, batch_size=8, lr=2e-3)
    _, history = train_vae(ds, vae_cfg=vae_cfg, train_cfg=train_cfg)
    assert history[-1]["total"] < history[0]["total"]


def test_train_vae_handles_empty_dataset_gracefully():
    import pytest
    ds = _SyntheticFrameDS(count=0, n_points=32)
    with pytest.raises(ValueError, match="empty"):
        train_vae(ds, vae_cfg=FrameVAEConfig(n_points=32, latent_dim=8, hidden=16),
                  train_cfg=VAETrainConfig(epochs=1, batch_size=4))
