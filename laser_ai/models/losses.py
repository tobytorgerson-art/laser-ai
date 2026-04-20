"""Loss functions for Frame VAE training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class LossWeights:
    chamfer: float = 1.0
    rgb: float = 0.3
    travel: float = 0.1
    kl: float = 0.01


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Chamfer L2-squared distance between two point sets.

    a, b: (B, N, 2). Returns scalar mean over the batch.
    """
    # Pairwise squared distances: (B, N, N)
    d = torch.cdist(a, b, p=2) ** 2
    min_ab = d.min(dim=2).values  # closest b for each a (B, N)
    min_ba = d.min(dim=1).values  # closest a for each b (B, N)
    return 0.5 * (min_ab.mean() + min_ba.mean())


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined VAE loss. recon and target are (B, N, 6)."""
    # xy: Chamfer
    ch = chamfer_distance(recon[..., :2], target[..., :2])
    # rgb: MSE
    rgb = F.mse_loss(recon[..., 2:5], target[..., 2:5])
    # is_travel: BCE
    tr = F.binary_cross_entropy(
        recon[..., 5].clamp(1e-6, 1 - 1e-6),
        target[..., 5],
    )
    # KL divergence to N(0, I)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = (
        weights.chamfer * ch
        + weights.rgb * rgb
        + weights.travel * tr
        + weights.kl * kl
    )
    parts = {
        "chamfer": float(ch.item()),
        "rgb": float(rgb.item()),
        "travel": float(tr.item()),
        "kl": float(kl.item()) * weights.kl,  # report contribution after weighting
    }
    return total, parts
