"""Train the AudioToLatentSequencer on (features, latents) pairs."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig


@dataclass(slots=True)
class SequencerTrainConfig:
    epochs: int = 20
    batch_size: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-5
    device: str = "auto"


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length (features, latents) tuples."""
    max_T = max(f.shape[0] for f, _ in batch)
    feat_dim = batch[0][0].shape[1]
    latent_dim = batch[0][1].shape[1]
    feats = torch.zeros(len(batch), max_T, feat_dim)
    lats = torch.zeros(len(batch), max_T, latent_dim)
    mask = torch.zeros(len(batch), max_T, dtype=torch.bool)
    for i, (f, l) in enumerate(batch):
        T = f.shape[0]
        feats[i, :T] = f
        lats[i, :T] = l
        mask[i, :T] = True
    return feats, lats, mask


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

    for epoch in range(train_cfg.epochs):
        model.train()
        # simple epoch: iterate in order, mini-batches of `batch_size`
        order = torch.randperm(len(pairs)).tolist()
        total = 0.0
        n = 0
        for start in range(0, len(order), train_cfg.batch_size):
            batch = [pairs[i] for i in order[start:start + train_cfg.batch_size]]
            feats, lats, mask = _collate(batch)
            feats = feats.to(device); lats = lats.to(device); mask = mask.to(device)
            pred = model(feats)
            # Masked MSE — ignore padding positions
            diff = (pred - lats) ** 2
            diff = diff * mask.unsqueeze(-1)
            loss = diff.sum() / (mask.sum() * lats.shape[-1] + 1e-8)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            n += 1

        entry = {"epoch": epoch, "mse": total / max(1, n)}
        history.append(entry)
        if progress_callback is not None:
            progress_callback(epoch, entry)

    return model, history
