"""Packaged save/load for the laser-ai model bundle."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
from laser_ai.models.vae import FrameVAE, FrameVAEConfig


_FORMAT_VERSION = 1


@dataclass(slots=True)
class LaserAICheckpoint:
    vae: FrameVAE
    vae_cfg: FrameVAEConfig
    sequencer: AudioToLatentSequencer
    seq_cfg: SequencerConfig
    audio_feature_dim: int
    fps: float


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
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> LaserAICheckpoint:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {payload.get('format_version')} "
            f"!= expected {_FORMAT_VERSION}"
        )
    # Coerce scale_range list back to tuple if needed
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
    )
