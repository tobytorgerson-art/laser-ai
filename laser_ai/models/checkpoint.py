"""Packaged save/load for the laser-ai model bundle."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
from laser_ai.models.vae import FrameVAE, FrameVAEConfig


_FORMAT_VERSION = 2
_SUPPORTED_VERSIONS = {1, 2}


@dataclass(slots=True)
class LaserAICheckpoint:
    vae: FrameVAE
    vae_cfg: FrameVAEConfig
    sequencer: AudioToLatentSequencer
    seq_cfg: SequencerConfig
    audio_feature_dim: int
    fps: float
    # Per-dim normalization stats applied to VAE latents during sequencer training.
    # When present, the sequencer outputs in normalized space and inference must
    # un-standardize (latent = pred * std + mean) before decoding. v1 checkpoints
    # carry None and the inference path treats them as identity.
    latent_mean: torch.Tensor | None = None
    latent_std: torch.Tensor | None = None


def save_checkpoint(ck: LaserAICheckpoint, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "vae_cfg": asdict(ck.vae_cfg),
        "seq_cfg": asdict(ck.seq_cfg),
        "vae_state": ck.vae.state_dict(),
        "seq_state": ck.sequencer.state_dict(),
        "audio_feature_dim": ck.audio_feature_dim,
        "fps": ck.fps,
        "latent_mean": ck.latent_mean,
        "latent_std": ck.latent_std,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> LaserAICheckpoint:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"checkpoint format version {version} not in supported {_SUPPORTED_VERSIONS}"
        )
    vcfg_dict = payload["vae_cfg"]
    scfg_dict = payload["seq_cfg"]
    vae_cfg = FrameVAEConfig(**vcfg_dict)
    seq_cfg = SequencerConfig(**scfg_dict)

    vae = FrameVAE(vae_cfg)
    vae.load_state_dict(payload["vae_state"])
    sequencer = AudioToLatentSequencer(seq_cfg)
    sequencer.load_state_dict(payload["seq_state"])

    return LaserAICheckpoint(
        vae=vae, vae_cfg=vae_cfg,
        sequencer=sequencer, seq_cfg=seq_cfg,
        audio_feature_dim=int(payload["audio_feature_dim"]),
        fps=float(payload["fps"]),
        latent_mean=payload.get("latent_mean"),
        latent_std=payload.get("latent_std"),
    )
