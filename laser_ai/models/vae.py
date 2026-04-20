"""Frame VAE: encode (N, 6) laser frames into a latent vector and back.

Encoder: 1D conv stack over the point axis → mean + logvar of a `latent_dim`
Gaussian. Decoder: latent → 1D conv-transpose → (N, 6) reconstruction with
tanh/sigmoid heads to keep outputs in valid ranges.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class FrameVAEConfig:
    n_points: int = 512
    latent_dim: int = 64
    hidden: int = 128
    n_channels_in: int = 6  # (x, y, r, g, b, is_travel)
    kernel: int = 5


class FrameVAE(nn.Module):
    def __init__(self, cfg: FrameVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FrameVAEConfig()
        c = self.cfg.n_channels_in
        h = self.cfg.hidden
        N = self.cfg.n_points
        k = self.cfg.kernel
        p = k // 2

        # Encoder: (B, C, N) -> (B, h*2, N/8) via three stride-2 convs
        self.enc = nn.Sequential(
            nn.Conv1d(c, h, k, stride=2, padding=p),          # N -> N/2
            nn.GELU(),
            nn.Conv1d(h, h * 2, k, stride=2, padding=p),       # N/2 -> N/4
            nn.GELU(),
            nn.Conv1d(h * 2, h * 2, k, stride=2, padding=p),   # N/4 -> N/8
            nn.GELU(),
        )
        flat = h * 2 * (N // 8)
        self.to_mu = nn.Linear(flat, self.cfg.latent_dim)
        self.to_logvar = nn.Linear(flat, self.cfg.latent_dim)

        # Decoder: latent -> (B, h*2, N/8) -> upscale to (B, C, N)
        self.from_latent = nn.Linear(self.cfg.latent_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(h * 2, h * 2, k, stride=2, padding=p, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(h * 2, h, k, stride=2, padding=p, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(h, h, k, stride=2, padding=p, output_padding=1),
            nn.GELU(),
        )
        # Separate heads so we can apply different activations per column
        self.head_xy = nn.Conv1d(h, 2, 1)
        self.head_rgb = nn.Conv1d(h, 3, 1)
        self.head_travel = nn.Conv1d(h, 1, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, C)
        h = self.enc(x.transpose(1, 2))           # (B, h*2, N/8)
        flat = h.flatten(1)
        return self.to_mu(flat), self.to_logvar(flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.from_latent(z).view(B, self.cfg.hidden * 2, self.cfg.n_points // 8)
        h = self.dec(h)
        xy = torch.tanh(self.head_xy(h))              # (B, 2, N)
        rgb = torch.sigmoid(self.head_rgb(h))         # (B, 3, N)
        travel = torch.sigmoid(self.head_travel(h))   # (B, 1, N)
        out = torch.cat([xy, rgb, travel], dim=1)     # (B, 6, N)
        return out.transpose(1, 2)                    # (B, N, 6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
