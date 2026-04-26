"""End-to-end Colab training orchestrator.

Called from the Colab notebook; assumes laser-ai is already pip-installed.
Input: a `bundle.zip` produced by `laser-ai prepare-bundle`.
Output: a `model.pt` checkpoint with trained VAE + Sequencer.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path


def run(
    bundle_zip: str,
    out_checkpoint: str,
    *,
    n_points: int = 512,
    latent_dim: int = 64,
    hidden: int = 128,
    vae_epochs: int = 30,
    vae_batch_size: int = 64,
    vae_lr: float = 1e-3,
    augment_mult: int = 8,
    seq_epochs: int = 40,
    seq_batch_size: int = 4,
    seq_lr: float = 5e-4,
    device: str = "auto",
) -> str:
    """Unpack bundle, train VAE then Sequencer, save checkpoint. Returns output path."""
    from laser_ai.augment.frame import AugmentConfig
    from laser_ai.audio.features import FEATURE_DIM
    from laser_ai.dataset.discovery import discover_pairs
    from laser_ai.dataset.torch_dataset import FrameDataset
    from laser_ai.models.checkpoint import LaserAICheckpoint, save_checkpoint
    from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
    from laser_ai.models.vae import FrameVAEConfig
    from laser_ai.training.prepare import build_sequencer_dataset
    from laser_ai.training.train_sequencer import SequencerTrainConfig, train_sequencer
    from laser_ai.training.train_vae import VAETrainConfig, train_vae

    # Unpack bundle into /content/data (or local equivalent)
    bundle_zip_path = Path(bundle_zip).resolve()
    work = Path("work").resolve()
    pairs_dir = work / "pairs"
    work.mkdir(exist_ok=True)
    pairs_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle_zip_path) as zf:
        zf.extractall(work)
        with zf.open("index.json") as f:
            index = json.load(f)
    print(f"[colab] unpacked {len(index['pairs'])} pair(s) to {work}")

    # 1. Train VAE
    ilda_paths = sorted(
        p for p in pairs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".ild", ".ilda"}
    )
    print(f"[colab] training VAE on {len(ilda_paths)} ILDA file(s)")
    ds = FrameDataset(
        ilda_paths, n_points=n_points,
        augment_mult=augment_mult, augment_cfg=AugmentConfig(),
    )
    vae_cfg = FrameVAEConfig(n_points=n_points, latent_dim=latent_dim, hidden=hidden)
    vae_train_cfg = VAETrainConfig(
        epochs=vae_epochs, batch_size=vae_batch_size, lr=vae_lr, device=device,
    )

    def _log_vae(epoch, entry):
        if epoch % 1 == 0:
            print(f"  [vae] epoch {epoch}: total={entry['total']:.4f}  "
                  f"chamfer={entry['chamfer']:.4f}  kl={entry['kl']:.4f}")

    vae, _ = train_vae(ds, vae_cfg=vae_cfg, train_cfg=vae_train_cfg,
                        progress_callback=_log_vae)

    # 2. Build sequencer dataset via VAE latents
    result = discover_pairs(pairs_dir)
    print(f"[colab] extracting latents from {len(result.pairs)} pair(s)")
    seq_pairs = build_sequencer_dataset(
        result.pairs, vae=vae, n_points=n_points, fps=30.0,
    )

    # 3. Train Sequencer
    # Use max_len=16384 (~9 min at 30 fps) so longer songs fit without truncation.
    print(f"[colab] training sequencer")
    seq_cfg = SequencerConfig(feature_dim=FEATURE_DIM, latent_dim=latent_dim, max_len=16384)
    seq_train_cfg = SequencerTrainConfig(
        epochs=seq_epochs, batch_size=seq_batch_size, lr=seq_lr, device=device,
    )

    def _log_seq(epoch, entry):
        print(f"  [seq] epoch {epoch}: mse={entry['mse']:.6f}")

    sequencer, _ = train_sequencer(
        seq_pairs, seq_cfg=seq_cfg, train_cfg=seq_train_cfg,
        progress_callback=_log_seq,
    )

    # 4. Save bundle
    ck = LaserAICheckpoint(
        vae=vae, vae_cfg=vae_cfg,
        sequencer=sequencer, seq_cfg=seq_cfg,
        audio_feature_dim=FEATURE_DIM, fps=30.0,
    )
    save_checkpoint(ck, out_checkpoint)
    print(f"[colab] saved {out_checkpoint}")
    return out_checkpoint
