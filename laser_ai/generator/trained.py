"""Generator backed by a trained LaserAICheckpoint."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from laser_ai.generator.base import Generator
from laser_ai.models.checkpoint import LaserAICheckpoint


class TrainedGenerator(Generator):
    def __init__(self, checkpoint: LaserAICheckpoint, device: str = "cpu") -> None:
        self.ck = checkpoint
        self.device = torch.device(device)
        self.ck.vae.to(self.device).eval()
        self.ck.sequencer.to(self.device).eval()

    @torch.no_grad()
    def generate(self, features: np.ndarray, *, n_points: int = 512) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != self.ck.audio_feature_dim:
            raise ValueError(
                f"features feature_dim mismatch: expected (T, {self.ck.audio_feature_dim}), "
                f"got {features.shape}"
            )
        native_n = self.ck.vae_cfg.n_points
        if n_points != native_n:
            warnings.warn(
                f"requested n_points={n_points} but model was trained at {native_n}; "
                f"using {native_n}",
                stacklevel=2,
            )
            n_points = native_n

        feats_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)  # (1, T, F)
        latents = self.ck.sequencer(feats_t)   # (1, T, latent_dim)
        # If the sequencer was trained on standardized latents, un-standardize.
        if self.ck.latent_mean is not None and self.ck.latent_std is not None:
            mean = self.ck.latent_mean.to(self.device)
            std = self.ck.latent_std.to(self.device)
            latents = latents * std + mean
        # Decode each frame's latent
        frames = self.ck.vae.decode(latents[0])  # (T, n_points, 6)
        return frames.cpu().numpy().astype(np.float32)
