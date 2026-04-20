"""Lazy PyTorch dataset over ILDA files with optional augmentation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from laser_ai.augment.frame import AugmentConfig, augment_frame
from laser_ai.dataset.tensors import show_to_tensor
from laser_ai.ilda.reader import read_ilda


class FrameDataset(Dataset):
    """Yields individual `(n_points, 6)` float32 frame tensors.

    Args:
        ilda_paths: list of .ild/.ilda files to draw frames from.
        n_points: fixed point count per frame (default 512).
        augment_mult: virtual dataset multiplier — each frame yields this many
            augmented copies per epoch. `augment_mult=1` with no `augment_cfg`
            produces the original frames once.
        augment_cfg: if provided, augmentations are applied per-sample.
        seed: base seed for the per-sample RNG (item index is mixed in).
    """

    def __init__(
        self,
        ilda_paths: list[str | Path],
        n_points: int = 512,
        augment_mult: int = 1,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.paths = [Path(p) for p in ilda_paths]
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(f"ILDA file not found: {p}")
        self.n_points = n_points
        self.augment_mult = max(1, augment_mult)
        self.augment_cfg = augment_cfg
        self.seed = seed

        # Precompute tensor blocks per file (cheap for foundation-sized data)
        self._blocks: list[np.ndarray] = []
        for p in self.paths:
            show = read_ilda(p)
            self._blocks.append(show_to_tensor(show, n_points=n_points))
        self._cum = np.cumsum([0] + [len(b) for b in self._blocks]).astype(np.int64)

    @property
    def _base_frames(self) -> int:
        return int(self._cum[-1])

    def __len__(self) -> int:
        return self._base_frames * self.augment_mult

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        base_idx = idx % self._base_frames
        aug_idx = idx // self._base_frames
        block_idx = int(np.searchsorted(self._cum, base_idx, side="right") - 1)
        offset = base_idx - int(self._cum[block_idx])
        arr = self._blocks[block_idx][offset].copy()

        if self.augment_cfg is not None:
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
            arr = augment_frame(arr, self.augment_cfg, rng)

        return torch.from_numpy(arr)
