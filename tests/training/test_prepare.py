"""Tests for latent extraction pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from laser_ai.audio.features import FEATURE_DIM
from laser_ai.dataset.pair import AudioLaserPair
from laser_ai.models.vae import FrameVAE, FrameVAEConfig
from laser_ai.training.prepare import build_sequencer_dataset


FIXTURE_ILDA = Path(__file__).parent.parent / "fixtures" / "tiny_show.ilda"


def _write_tiny_wav(path: Path, duration_s: float = 1.0) -> None:
    import soundfile as sf
    sr = 44100
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    samples = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    sf.write(path, samples, sr, subtype="PCM_16")


def test_build_sequencer_dataset_returns_paired_tensors(tmp_path: Path):
    wav = tmp_path / "song.wav"
    _write_tiny_wav(wav)
    pair = AudioLaserPair(audio_path=wav, ilda_path=FIXTURE_ILDA)

    vae_cfg = FrameVAEConfig(n_points=64, latent_dim=8, hidden=16)
    vae = FrameVAE(vae_cfg).eval()

    samples = build_sequencer_dataset([pair], vae=vae, n_points=64, fps=30.0)

    assert len(samples) == 1
    feats, latents = samples[0]
    assert feats.shape[1] == FEATURE_DIM
    assert latents.shape[1] == 8
    # Feature and latent time axes must align
    assert feats.shape[0] == latents.shape[0]
    assert feats.dtype == torch.float32
    assert latents.dtype == torch.float32
