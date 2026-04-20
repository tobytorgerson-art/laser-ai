"""Train the Frame VAE."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from laser_ai.models.losses import LossWeights, vae_loss
from laser_ai.models.vae import FrameVAE, FrameVAEConfig


@dataclass(slots=True)
class VAETrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"           # "auto" | "cpu" | "cuda"
    loss_weights: LossWeights = field(default_factory=LossWeights)
    log_every: int = 1             # per epoch


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_vae(
    dataset: Dataset,
    vae_cfg: FrameVAEConfig | None = None,
    train_cfg: VAETrainConfig | None = None,
    progress_callback=None,   # optional fn(epoch, history_entry) for external UIs
) -> tuple[FrameVAE, list[dict]]:
    """Run VAE training and return (model, per-epoch history)."""
    if len(dataset) == 0:
        raise ValueError("dataset is empty; cannot train")

    vae_cfg = vae_cfg or FrameVAEConfig()
    train_cfg = train_cfg or VAETrainConfig()
    device = _resolve_device(train_cfg.device)

    model = FrameVAE(vae_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)
    loader = DataLoader(
        dataset, batch_size=train_cfg.batch_size, shuffle=True,
        drop_last=False, num_workers=0,
    )

    history: list[dict] = []
    for epoch in range(train_cfg.epochs):
        model.train()
        totals = {"total": 0.0, "chamfer": 0.0, "rgb": 0.0, "travel": 0.0, "kl": 0.0}
        n_batches = 0
        for x in loader:
            x = x.to(device).float()
            recon, mu, logvar = model(x)
            loss, parts = vae_loss(recon, x, mu, logvar, train_cfg.loss_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            totals["total"] += float(loss.item())
            for k, v in parts.items():
                totals[k] += float(v)
            n_batches += 1

        entry = {k: v / max(1, n_batches) for k, v in totals.items()}
        entry["epoch"] = epoch
        history.append(entry)
        if progress_callback is not None:
            progress_callback(epoch, entry)

    return model, history
