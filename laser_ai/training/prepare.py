"""Build the sequencer training dataset from audio+ILDA pairs."""
from __future__ import annotations

import numpy as np
import torch

from laser_ai.audio.features import extract_features
from laser_ai.audio.loader import load_audio
from laser_ai.dataset.pair import AudioLaserPair
from laser_ai.dataset.tensors import show_to_tensor
from laser_ai.ilda.reader import read_ilda
from laser_ai.models.vae import FrameVAE


def _stretch_latents_to_length(latents: torch.Tensor, target_T: int) -> torch.Tensor:
    """Resample a (T, D) latent sequence to (target_T, D) via nearest-neighbor indexing.

    Needed when the ILDA has a different frame count than the audio's feature-frame count.
    """
    T, D = latents.shape
    if T == target_T:
        return latents
    if T == 0:
        return torch.zeros(target_T, D, dtype=latents.dtype)
    # Nearest index lookup: sample[i] = latents[round(i * (T - 1) / (target_T - 1))]
    if target_T == 1:
        return latents[:1]
    idx = torch.linspace(0, T - 1, target_T).round().long()
    return latents[idx]


@torch.no_grad()
def build_sequencer_dataset(
    pairs: list[AudioLaserPair],
    vae: FrameVAE,
    *,
    n_points: int = 512,
    fps: float = 30.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """For each (audio, ilda) pair, produce (audio_features, frame_latents).

    audio_features: (T, FEATURE_DIM) from librosa at `fps`.
    frame_latents: (T, latent_dim) from VAE.encode mean (mu) applied to each ILDA frame.
    Sequences are time-aligned by nearest-neighbor stretching if lengths differ.
    """
    vae.eval()
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for pair in pairs:
        # Audio features
        samples, sr = load_audio(pair.audio_path)
        feats_np = extract_features(samples, sr, fps=fps)  # (T_audio, FEATURE_DIM)
        feats = torch.from_numpy(feats_np).float()

        # ILDA latents
        show = read_ilda(pair.ilda_path)
        frames_np = show_to_tensor(show, n_points=n_points)  # (T_ilda, N, 6)
        frames = torch.from_numpy(frames_np).float()
        if frames.shape[0] == 0:
            continue
        mu, _ = vae.encode(frames)   # (T_ilda, latent_dim)

        # Align the two time axes
        latents = _stretch_latents_to_length(mu.detach(), target_T=feats.shape[0])
        out.append((feats, latents))
    return out
