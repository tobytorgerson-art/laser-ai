"""Train the AudioToLatentSequencer on (features, latents) pairs.

Training strategy: instead of feeding whole songs (15 long sequences -> ~4
gradient steps per epoch -> chronic underfit), we slice each song into many
fixed-length windows and randomize the positional offset. This:
  * Multiplies effective sample count by ~T/window per song.
  * Eliminates padding waste; every batch is dense.
  * Spreads gradient through every position of the learned pos_emb so the
    model generalizes to song lengths it has never seen as one chunk.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from laser_ai.models.losses import chamfer_distance
from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
from laser_ai.models.vae import FrameVAE


@dataclass(slots=True)
class SequencerTrainConfig:
    epochs: int = 200
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-5
    device: str = "auto"
    window: int = 512                # train on 512-frame chunks (~17s @ 30 fps)
    samples_per_epoch: int = 256     # number of random windows drawn per epoch


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sample_window(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    window: int,
    max_pos_offset: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pick a random pair, slice a `window`-length chunk, return with random pos offset.

    If a pair is shorter than `window`, the whole pair is returned and pos_offset is 0
    (the caller batches it back with same-length windows so this can't happen unless
    the entire dataset is short).
    """
    pair_idx = rng.randrange(len(pairs))
    feats, lats = pairs[pair_idx]
    T = feats.shape[0]
    if T <= window:
        return feats, lats, 0
    start = rng.randrange(T - window + 1)
    pos_offset = rng.randrange(max_pos_offset + 1) if max_pos_offset > 0 else 0
    return feats[start:start + window], lats[start:start + window], pos_offset


def _draw_batch(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    window: int,
    batch_size: int,
    max_pos_offset: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Draw `batch_size` windows that all share one pos_offset (single forward call)."""
    pos_offset = rng.randrange(max_pos_offset + 1) if max_pos_offset > 0 else 0
    feat_chunks = []
    lat_chunks = []
    for _ in range(batch_size):
        f, l, _ = _sample_window(pairs, window, max_pos_offset=0, rng=rng)
        # If a pair is shorter than `window`, pad to `window` so the batch stacks.
        if f.shape[0] < window:
            pad_T = window - f.shape[0]
            f = F.pad(f, (0, 0, 0, pad_T))
            l = F.pad(l, (0, 0, 0, pad_T))
        feat_chunks.append(f)
        lat_chunks.append(l)
    feats = torch.stack(feat_chunks, dim=0)
    lats = torch.stack(lat_chunks, dim=0)
    return feats, lats, pos_offset


def train_sequencer(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    seq_cfg: SequencerConfig | None = None,
    train_cfg: SequencerTrainConfig | None = None,
    progress_callback=None,
) -> tuple[AudioToLatentSequencer, list[dict], torch.Tensor, torch.Tensor]:
    """Train sequencer on per-dim z-scored latents.

    Returns ``(model, history, latent_mean, latent_std)``. The model predicts in
    standardized space; inference must apply ``pred * latent_std + latent_mean``
    before decoding. Standardizing per-dim balances gradient across latent
    dimensions and prevents the dominant-variance dims from making
    mean-regression an MSE-optimal solution.
    """
    if len(pairs) == 0:
        raise ValueError("pairs list is empty")
    seq_cfg = seq_cfg or SequencerConfig()
    train_cfg = train_cfg or SequencerTrainConfig()
    device = _resolve_device(train_cfg.device)

    model = AudioToLatentSequencer(seq_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)
    history: list[dict] = []
    rng = random.Random(0)
    window = min(train_cfg.window, seq_cfg.max_len)
    max_pos_offset = max(0, seq_cfg.max_len - window)

    # Per-dim normalization stats across all training latents.
    all_lats = torch.cat([l for _, l in pairs], dim=0)
    latent_mean = all_lats.mean(dim=0)                 # (D,)
    latent_std = all_lats.std(dim=0).clamp_min(1e-6)   # (D,) avoid /0
    mean_dev = latent_mean.to(device)
    std_dev = latent_std.to(device)

    n_steps_per_epoch = max(1, train_cfg.samples_per_epoch // train_cfg.batch_size)

    for epoch in range(train_cfg.epochs):
        model.train()
        total = 0.0
        pred_var_sum = 0.0
        for _ in range(n_steps_per_epoch):
            feats, lats, pos_offset = _draw_batch(
                pairs, window=window,
                batch_size=train_cfg.batch_size,
                max_pos_offset=max_pos_offset,
                rng=rng,
            )
            feats = feats.to(device); lats = lats.to(device)
            lats_norm = (lats - mean_dev) / std_dev
            pred = model(feats, pos_offset=pos_offset)
            loss = F.mse_loss(pred, lats_norm)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            pred_var_sum += float(pred.detach().std(dim=1).mean().item())

        entry = {
            "epoch": epoch,
            "mse": total / n_steps_per_epoch,
            "pred_std": pred_var_sum / n_steps_per_epoch,
            "target_std": 1.0,  # targets are unit-std per dim by construction
        }
        history.append(entry)
        if progress_callback is not None:
            progress_callback(epoch, entry)

    return model, history, latent_mean.cpu(), latent_std.cpu()


@dataclass(slots=True)
class SequencerE2ETrainConfig:
    """Config for the end-to-end fine-tuning pass."""
    epochs: int = 50
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "auto"
    window: int = 256                # smaller than MSE window — decoding is expensive
    samples_per_epoch: int = 64
    chamfer_weight: float = 1.0
    rgb_weight: float = 0.3
    travel_weight: float = 0.1


def _draw_batch_frames(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    window: int,
    batch_size: int,
    max_pos_offset: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Like _draw_batch but for (features, frames) pairs where frames is (T, N, 6)."""
    pos_offset = rng.randrange(max_pos_offset + 1) if max_pos_offset > 0 else 0
    feat_chunks: list[torch.Tensor] = []
    frame_chunks: list[torch.Tensor] = []
    for _ in range(batch_size):
        idx = rng.randrange(len(pairs))
        feats, frames = pairs[idx]
        T = feats.shape[0]
        if T <= window:
            f, fr = feats, frames
            if f.shape[0] < window:
                pad_T = window - f.shape[0]
                f = F.pad(f, (0, 0, 0, pad_T))                    # (T,F)
                fr = F.pad(fr, (0, 0, 0, 0, 0, pad_T))            # (T,N,6)
        else:
            start = rng.randrange(T - window + 1)
            f = feats[start:start + window]
            fr = frames[start:start + window]
        feat_chunks.append(f)
        frame_chunks.append(fr)
    return torch.stack(feat_chunks), torch.stack(frame_chunks), pos_offset


def train_sequencer_e2e(
    pairs_features_frames: list[tuple[torch.Tensor, torch.Tensor]],
    sequencer: AudioToLatentSequencer,
    vae: FrameVAE,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    train_cfg: SequencerE2ETrainConfig | None = None,
    progress_callback=None,
) -> tuple[AudioToLatentSequencer, list[dict]]:
    """End-to-end fine-tune: backprop chamfer/rgb/travel through VAE.decode.

    The MSE-pre-trained sequencer reaches the right latent statistics but its
    outputs may sit off the VAE's manifold (decode produces noise). This pass
    freezes the VAE and trains the sequencer with a reconstruction loss on
    decoded frames, forcing latents to land where the decoder can use them.
    """
    if len(pairs_features_frames) == 0:
        raise ValueError("pairs list is empty")
    train_cfg = train_cfg or SequencerE2ETrainConfig()
    device = _resolve_device(train_cfg.device)

    sequencer = sequencer.to(device)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    mean_dev = latent_mean.to(device)
    std_dev = latent_std.to(device)

    opt = torch.optim.AdamW(sequencer.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)
    history: list[dict] = []
    rng = random.Random(0)
    window = min(train_cfg.window, sequencer.cfg.max_len)
    max_pos_offset = max(0, sequencer.cfg.max_len - window)
    n_steps_per_epoch = max(1, train_cfg.samples_per_epoch // train_cfg.batch_size)
    n_points = vae.cfg.n_points

    for epoch in range(train_cfg.epochs):
        sequencer.train()
        ch_total = 0.0
        rgb_total = 0.0
        tr_total = 0.0
        for _ in range(n_steps_per_epoch):
            feats, frames, pos_offset = _draw_batch_frames(
                pairs_features_frames, window=window,
                batch_size=train_cfg.batch_size,
                max_pos_offset=max_pos_offset, rng=rng,
            )
            feats = feats.to(device); frames = frames.to(device)
            B, T_w, _ = feats.shape

            pred_norm = sequencer(feats, pos_offset=pos_offset)        # (B, T, D)
            pred_lats = pred_norm * std_dev + mean_dev                  # un-standardize

            decoded = vae.decode(pred_lats.reshape(B * T_w, -1))        # (B*T, N, 6)
            decoded = decoded.reshape(B, T_w, n_points, 6)

            decoded_xy = decoded[..., :2].reshape(B * T_w, n_points, 2)
            target_xy = frames[..., :2].reshape(B * T_w, n_points, 2)
            ch = chamfer_distance(decoded_xy, target_xy)
            rgb = F.mse_loss(decoded[..., 2:5], frames[..., 2:5])
            tr = F.binary_cross_entropy(
                decoded[..., 5].clamp(1e-6, 1 - 1e-6),
                frames[..., 5],
            )
            loss = (train_cfg.chamfer_weight * ch
                    + train_cfg.rgb_weight * rgb
                    + train_cfg.travel_weight * tr)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sequencer.parameters(), 1.0)
            opt.step()

            ch_total += float(ch.item())
            rgb_total += float(rgb.item())
            tr_total += float(tr.item())

        entry = {
            "epoch": epoch,
            "chamfer": ch_total / n_steps_per_epoch,
            "rgb": rgb_total / n_steps_per_epoch,
            "travel": tr_total / n_steps_per_epoch,
        }
        history.append(entry)
        if progress_callback is not None:
            progress_callback(epoch, entry)

    return sequencer, history
