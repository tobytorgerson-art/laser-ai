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


def _stretch_to_length(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Resample any tensor with leading time dim to length `target_T` via nearest-neighbor.

    Works for (T, D) latents, (T, N, 6) frames, and (T, F) features.
    """
    T = x.shape[0]
    if T == target_T:
        return x
    if T == 0:
        return torch.zeros((target_T,) + tuple(x.shape[1:]), dtype=x.dtype)
    if target_T == 1:
        return x[:1]
    idx = torch.linspace(0, T - 1, target_T).round().long()
    return x[idx]


# Backward-compatible alias for callers that import the old name
_stretch_latents_to_length = _stretch_to_length


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
    device = next(vae.parameters()).device
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for pair in pairs:
        # Audio features
        samples, sr = load_audio(pair.audio_path)
        feats_np = extract_features(samples, sr, fps=fps)  # (T_audio, FEATURE_DIM)
        feats = torch.from_numpy(feats_np).float()

        # ILDA latents
        show = read_ilda(pair.ilda_path)
        frames_np = show_to_tensor(show, n_points=n_points)  # (T_ilda, N, 6)
        frames = torch.from_numpy(frames_np).float().to(device)
        if frames.shape[0] == 0:
            continue
        mu, _ = vae.encode(frames)   # (T_ilda, latent_dim), on device
        # Move latents back to CPU for storage / downstream training collation
        mu = mu.detach().cpu()

        # Align the two time axes
        latents = _stretch_to_length(mu, target_T=feats.shape[0])
        out.append((feats, latents))
    return out


def build_sequencer_dataset_e2e(
    pairs: list[AudioLaserPair],
    *,
    n_points: int = 512,
    fps: float = 30.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """For each (audio, ilda) pair, produce (audio_features, ground_truth_frames).

    Used by end-to-end fine-tuning where the loss is chamfer/rgb/travel on frames
    decoded by the VAE, rather than MSE in latent space. Frames are
    nearest-neighbor stretched to match the audio feature timeline.
    """
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for pair in pairs:
        samples, sr = load_audio(pair.audio_path)
        feats_np = extract_features(samples, sr, fps=fps)  # (T_audio, FEATURE_DIM)
        feats = torch.from_numpy(feats_np).float()

        show = read_ilda(pair.ilda_path)
        frames_np = show_to_tensor(show, n_points=n_points)  # (T_ilda, N, 6)
        frames = torch.from_numpy(frames_np).float()
        if frames.shape[0] == 0:
            continue

        frames = _stretch_to_length(frames, target_T=feats.shape[0])
        out.append((feats, frames))
    return out
