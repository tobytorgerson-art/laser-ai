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

from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig


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
) -> tuple[AudioToLatentSequencer, list[dict]]:
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

    # Compute target latent stats so we can report "is the model still mean-regressing?"
    all_lats = torch.cat([l for _, l in pairs], dim=0)
    target_std = float(all_lats.std(dim=0).mean().item())

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
            pred = model(feats, pos_offset=pos_offset)
            loss = F.mse_loss(pred, lats)

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
            "target_std": target_std,
        }
        history.append(entry)
        if progress_callback is not None:
            progress_callback(epoch, entry)

    return model, history
