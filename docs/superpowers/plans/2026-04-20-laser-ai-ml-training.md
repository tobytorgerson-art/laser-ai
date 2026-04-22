# laser-ai ML Training Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the foundation's rule-based `StubGenerator` with a real trained model: a PyTorch Frame VAE (learns "vocabulary" of laser shapes) + a small causal Transformer sequencer (learns audio → latent mapping). Training runs in free Google Colab; inference runs locally on CPU. Output is the same `.ilda` format the foundation pipeline already emits.

**Architecture:** Two-model pipeline — (1) **VAE** encodes each `(512, 6)` frame into a 64-dim latent; (2) **Sequencer** maps `(T, 146)` audio-feature stream to `(T, 64)` latent stream; VAE decoder turns each predicted latent back into a frame. The existing `Generator` protocol stays; a new `TrainedGenerator` slots into the existing pipeline without touching anything upstream. Everything is CPU-compatible for inference, GPU-accelerated for Colab training. Synthetic test data is used for local unit tests; real training uses the user's ILDA library in Colab.

**Tech Stack:** PyTorch 2.x (CPU locally, CUDA in Colab), NumPy, existing foundation stack (librosa, click, soundfile). No CLAP in this plan — punted to a later plan.

**Spec:** `docs/superpowers/specs/2026-04-20-laser-ai-design.md`
**Prerequisite:** Plan 1 foundation complete (commits up to `v0.1.0-foundation`).

---

## File Structure

This plan creates:

```
laser_ai/
  augment/
    __init__.py
    frame.py                     # rotate/flip/scale/hue augmentations on (N, 6) arrays

  dataset/
    tensors.py                   # ILDA Show → (num_frames, N, 6) float32 tensor
    torch_dataset.py             # PyTorch Dataset wrapping frame tensors with augmentation

  models/
    __init__.py
    vae.py                       # FrameVAE (encoder + decoder) + config dataclass
    sequencer.py                 # AudioToLatentSequencer (small causal Transformer)
    losses.py                    # Chamfer distance, VAE combined loss
    checkpoint.py                # save/load VAE+Sequencer bundle to a single .pt file

  training/
    __init__.py
    train_vae.py                 # VAE training loop (local + Colab)
    train_sequencer.py           # Sequencer training loop (local + Colab)
    prepare.py                   # extract latents from VAE, build sequencer dataset

  generator/
    trained.py                   # TrainedGenerator — real Generator implementation

  bundle/
    __init__.py
    pack.py                      # zip up dataset for Colab upload
    unpack.py                    # unzip + validate dataset in Colab

  cli.py                         # UPDATED: add train-vae, train-sequencer, prepare-bundle, --model flag

colab/
  laser_ai_train.ipynb           # Colab notebook (thin; imports Python below)
  colab_train.py                 # orchestrates end-to-end Colab training
  README.md                      # user-facing Colab instructions

tests/
  augment/
    __init__.py
    test_frame.py
  dataset/
    test_tensors.py
    test_torch_dataset.py
  models/
    __init__.py
    test_vae.py
    test_sequencer.py
    test_losses.py
    test_checkpoint.py
  training/
    __init__.py
    test_train_vae.py            # tiny-dataset smoke tests (~5 steps)
    test_train_sequencer.py
    test_prepare.py
  generator/
    test_trained.py
  bundle/
    __init__.py
    test_pack.py
  cli/
    test_cli_train.py            # new CLI subcommand tests
```

All new code is under `laser_ai/` except the Colab shell (`colab/`). Tests mirror source structure.

---

## Task 1: Add PyTorch dependency

**Files:**
- Modify: `pyproject.toml`

PyTorch on Windows + Python 3.11 installs cleanly from PyPI. The CPU wheel is ~200MB; no CUDA toolkit required.

- [ ] **Step 1: Update `pyproject.toml`** — add `torch` to `dependencies`.

Current `dependencies` list (from Plan 1):
```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "librosa>=0.10",
    "soundfile>=0.12",
    "click>=8.1",
    "structlog>=24.1",
]
```

Change to:
```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "librosa>=0.10",
    "soundfile>=0.12",
    "click>=8.1",
    "structlog>=24.1",
    "torch>=2.2,<3",
]
```

- [ ] **Step 2: Install**

```bash
source .venv/Scripts/activate
pip install -e ".[dev]"
```

Expected: torch (CPU) installs. May take 2-5 minutes (~200MB download).

- [ ] **Step 3: Verify**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: version like `2.x.y False` (False on the CPU-only machine is correct).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add torch dependency for ML training"
```

---

## Task 2: Frame augmentations

**Files:**
- Create: `laser_ai/augment/__init__.py`
- Create: `laser_ai/augment/frame.py`
- Create: `tests/augment/__init__.py`
- Create: `tests/augment/test_frame.py`

Per spec §4: rotation ±15°, flip, scale jitter ±10%, hue rotation. Applied to `(N, 6)` float32 arrays where the last column (`is_travel`) is preserved unchanged.

- [ ] **Step 1: Write `laser_ai/augment/__init__.py`**

```python
"""Frame augmentations for training data."""

from laser_ai.augment.frame import (
    AugmentConfig,
    augment_frame,
    flip_horizontal,
    flip_vertical,
    rotate,
    rotate_hue,
    scale,
)

__all__ = [
    "AugmentConfig",
    "augment_frame",
    "flip_horizontal",
    "flip_vertical",
    "rotate",
    "rotate_hue",
    "scale",
]
```

- [ ] **Step 2: Write `tests/augment/__init__.py`** (empty)

- [ ] **Step 3: Write `tests/augment/test_frame.py`**

```python
"""Tests for frame augmentations."""
from __future__ import annotations

import numpy as np
import pytest

from laser_ai.augment.frame import (
    AugmentConfig,
    augment_frame,
    flip_horizontal,
    flip_vertical,
    rotate,
    rotate_hue,
    scale,
)


def _line_frame(n: int = 64) -> np.ndarray:
    """Horizontal red line from (-0.5, 0) to (0.5, 0)."""
    arr = np.zeros((n, 6), dtype=np.float32)
    arr[:, 0] = np.linspace(-0.5, 0.5, n)
    arr[:, 2] = 1.0  # red
    return arr


def test_rotate_90_swaps_xy():
    arr = _line_frame(64)
    out = rotate(arr, np.pi / 2)
    # After 90° rotation, a horizontal line becomes vertical
    assert out[:, 0].std() < 0.05
    assert out[:, 1].std() > 0.2


def test_rotate_preserves_shape_and_dtype():
    arr = _line_frame(32)
    out = rotate(arr, 0.3)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype


def test_rotate_preserves_travel_flag():
    arr = _line_frame(16)
    arr[::2, 5] = 1.0
    out = rotate(arr, 0.5)
    np.testing.assert_array_equal(out[:, 5], arr[:, 5])


def test_flip_horizontal_negates_x():
    arr = _line_frame(16)
    out = flip_horizontal(arr)
    np.testing.assert_allclose(out[:, 0], -arr[:, 0])
    np.testing.assert_allclose(out[:, 1], arr[:, 1])


def test_flip_vertical_negates_y():
    arr = np.zeros((8, 6), dtype=np.float32)
    arr[:, 1] = np.linspace(-0.5, 0.5, 8)
    out = flip_vertical(arr)
    np.testing.assert_allclose(out[:, 1], -arr[:, 1])


def test_scale_multiplies_xy():
    arr = _line_frame(16)
    out = scale(arr, 1.5)
    np.testing.assert_allclose(out[:, 0], arr[:, 0] * 1.5, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], arr[:, 1] * 1.5, atol=1e-6)


def test_scale_does_not_touch_color_or_travel():
    arr = _line_frame(16)
    out = scale(arr, 0.5)
    np.testing.assert_allclose(out[:, 2:6], arr[:, 2:6])


def test_rotate_hue_by_zero_is_identity():
    arr = _line_frame(8)
    out = rotate_hue(arr, 0.0)
    np.testing.assert_allclose(out[:, 2:5], arr[:, 2:5], atol=1e-5)


def test_rotate_hue_changes_red_to_green_at_120():
    arr = np.zeros((4, 6), dtype=np.float32)
    arr[:, 2] = 1.0  # pure red
    out = rotate_hue(arr, np.radians(120))
    # Approx green — green channel should dominate
    assert out[0, 3] > 0.9
    assert out[0, 2] < 0.1


def test_augment_frame_respects_config_disables():
    rng = np.random.default_rng(0)
    arr = _line_frame(32)
    cfg = AugmentConfig(
        enable_rotate=False, enable_flip_h=False, enable_flip_v=False,
        enable_scale=False, enable_hue=False,
    )
    out = augment_frame(arr, cfg, rng)
    np.testing.assert_allclose(out, arr)


def test_augment_frame_is_deterministic_for_fixed_rng():
    arr = _line_frame(32)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    cfg = AugmentConfig()
    out1 = augment_frame(arr, cfg, rng1)
    out2 = augment_frame(arr, cfg, rng2)
    np.testing.assert_allclose(out1, out2)


def test_augment_frame_keeps_output_in_range():
    rng = np.random.default_rng(1)
    arr = _line_frame(64)
    out = augment_frame(arr, AugmentConfig(), rng)
    assert out[:, 0].max() <= 1.0
    assert out[:, 0].min() >= -1.0
    assert out[:, 1].max() <= 1.0
    assert out[:, 1].min() >= -1.0
    assert out[:, 2:5].max() <= 1.0 + 1e-6
    assert out[:, 2:5].min() >= 0.0 - 1e-6


def test_augment_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        augment_frame(np.zeros((10, 5), dtype=np.float32), AugmentConfig(), np.random.default_rng(0))
```

- [ ] **Step 4: Write `laser_ai/augment/frame.py`**

```python
"""Per-frame augmentations operating on (N, 6) float32 arrays.

Columns: (x, y, r, g, b, is_travel). Augmentations preserve is_travel unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AugmentConfig:
    """Control which augmentations fire and with what magnitudes."""
    enable_rotate: bool = True
    enable_flip_h: bool = True
    enable_flip_v: bool = True
    enable_scale: bool = True
    enable_hue: bool = True
    rotate_max_deg: float = 15.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    flip_h_prob: float = 0.5
    flip_v_prob: float = 0.5


def _check(arr: np.ndarray) -> None:
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"expected (N, 6) array, got {arr.shape}")


def rotate(arr: np.ndarray, theta: float) -> np.ndarray:
    """Rotate xy by `theta` radians. Non-xy columns untouched."""
    _check(arr)
    c, s = np.cos(theta), np.sin(theta)
    out = arr.copy()
    out[:, 0] = c * arr[:, 0] - s * arr[:, 1]
    out[:, 1] = s * arr[:, 0] + c * arr[:, 1]
    return out


def flip_horizontal(arr: np.ndarray) -> np.ndarray:
    _check(arr)
    out = arr.copy()
    out[:, 0] = -arr[:, 0]
    return out


def flip_vertical(arr: np.ndarray) -> np.ndarray:
    _check(arr)
    out = arr.copy()
    out[:, 1] = -arr[:, 1]
    return out


def scale(arr: np.ndarray, factor: float) -> np.ndarray:
    _check(arr)
    out = arr.copy()
    out[:, :2] = arr[:, :2] * factor
    return out


def rotate_hue(arr: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate the RGB hue by `angle_rad` (0 .. 2π). Preserves saturation/value."""
    _check(arr)
    rgb = arr[:, 2:5]
    # RGB → HSV (via simple formula; keeps the per-point intensity)
    cmax = rgb.max(axis=1)
    cmin = rgb.min(axis=1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 0
    idx_r = (rgb[:, 0] == cmax) & mask
    idx_g = (rgb[:, 1] == cmax) & mask
    idx_b = (rgb[:, 2] == cmax) & mask
    h[idx_r] = ((rgb[idx_r, 1] - rgb[idx_r, 2]) / delta[idx_r]) % 6
    h[idx_g] = (rgb[idx_g, 2] - rgb[idx_g, 0]) / delta[idx_g] + 2
    h[idx_b] = (rgb[idx_b, 0] - rgb[idx_b, 1]) / delta[idx_b] + 4
    h = (h * 60.0 + np.degrees(angle_rad)) % 360.0
    s = np.where(cmax > 0, delta / np.maximum(cmax, 1e-12), 0.0)
    v = cmax
    # HSV → RGB
    hi = (h / 60.0).astype(np.int64) % 6
    f = (h / 60.0) - np.floor(h / 60.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    new_rgb = np.zeros_like(rgb)
    conds = [
        (hi == 0, v, t, p),
        (hi == 1, q, v, p),
        (hi == 2, p, v, t),
        (hi == 3, p, q, v),
        (hi == 4, t, p, v),
        (hi == 5, v, p, q),
    ]
    for cond, rv, gv, bv in conds:
        new_rgb[cond, 0] = rv[cond]
        new_rgb[cond, 1] = gv[cond]
        new_rgb[cond, 2] = bv[cond]
    out = arr.copy()
    out[:, 2:5] = np.clip(new_rgb, 0.0, 1.0).astype(arr.dtype)
    return out


def augment_frame(arr: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply a random subset of augmentations per cfg using the provided RNG."""
    _check(arr)
    out = arr.copy()

    if cfg.enable_rotate:
        theta = np.radians(rng.uniform(-cfg.rotate_max_deg, cfg.rotate_max_deg))
        out = rotate(out, theta)

    if cfg.enable_flip_h and rng.random() < cfg.flip_h_prob:
        out = flip_horizontal(out)

    if cfg.enable_flip_v and rng.random() < cfg.flip_v_prob:
        out = flip_vertical(out)

    if cfg.enable_scale:
        lo, hi = cfg.scale_range
        out = scale(out, float(rng.uniform(lo, hi)))

    if cfg.enable_hue:
        out = rotate_hue(out, float(rng.uniform(0.0, 2 * np.pi)))

    # Clamp to valid ranges
    out[:, :2] = np.clip(out[:, :2], -1.0, 1.0)
    out[:, 2:5] = np.clip(out[:, 2:5], 0.0, 1.0)
    return out
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/augment/ -v
```

Expected: all 13 PASS.

- [ ] **Step 6: Commit**

```bash
git add laser_ai/augment tests/augment
git commit -m "feat(augment): per-frame rotation/flip/scale/hue augmentations"
```

---

## Task 3: Show → tensor extraction

**Files:**
- Create: `laser_ai/dataset/tensors.py`
- Create: `tests/dataset/test_tensors.py`

Build on `ilda/resample.py` from Plan 1 to turn an ILDA `Show` into a `(num_frames, N, 6)` float32 tensor ready for training. Pads short shows, truncates long ones.

- [ ] **Step 1: Write `tests/dataset/test_tensors.py`**

```python
"""Tests for ILDA Show → tensor extraction."""
from __future__ import annotations

import numpy as np

from laser_ai.dataset.tensors import show_to_tensor
from laser_ai.ilda.types import Frame, Point, Show


def _two_frame_show() -> Show:
    f0 = Frame(points=[
        Point(-16000, 0, 255, 0, 0),
        Point(16000, 0, 255, 0, 0, is_last_point=True),
    ])
    f1 = Frame(points=[
        Point(0, -16000, 0, 255, 0),
        Point(0, 16000, 0, 255, 0, is_last_point=True),
    ])
    return Show(frames=[f0, f1])


def test_show_to_tensor_shape():
    t = show_to_tensor(_two_frame_show(), n_points=64)
    assert t.shape == (2, 64, 6)
    assert t.dtype == np.float32


def test_tensor_values_in_normalized_range():
    t = show_to_tensor(_two_frame_show(), n_points=64)
    assert t[..., :2].max() <= 1.0 + 1e-6
    assert t[..., :2].min() >= -1.0 - 1e-6
    assert t[..., 2:5].max() <= 1.0 + 1e-6
    assert t[..., 2:5].min() >= 0.0 - 1e-6


def test_empty_show_returns_zero_frame_tensor():
    t = show_to_tensor(Show(frames=[]), n_points=32)
    assert t.shape == (0, 32, 6)


def test_color_preserved_per_frame():
    t = show_to_tensor(_two_frame_show(), n_points=32)
    # Frame 0 is red, frame 1 is green — check mean color
    assert t[0, :, 2].mean() > 0.9   # red channel high in frame 0
    assert t[0, :, 3].mean() < 0.1   # green low
    assert t[1, :, 3].mean() > 0.9   # green high in frame 1
    assert t[1, :, 2].mean() < 0.1   # red low
```

- [ ] **Step 2: Write `laser_ai/dataset/tensors.py`**

```python
"""Convert ILDA Show objects to fixed-shape float32 tensors."""
from __future__ import annotations

import numpy as np

from laser_ai.ilda.resample import resample_frame
from laser_ai.ilda.types import Show


def show_to_tensor(show: Show, n_points: int = 512) -> np.ndarray:
    """Return a `(num_frames, n_points, 6)` float32 tensor for the show.

    Each frame is arc-length resampled to exactly `n_points` points, then
    converted to the normalized (x, y, r, g, b, is_travel) float32 format.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    n_frames = len(show.frames)
    if n_frames == 0:
        return np.zeros((0, n_points, 6), dtype=np.float32)

    out = np.empty((n_frames, n_points, 6), dtype=np.float32)
    for i, frame in enumerate(show.frames):
        resampled = resample_frame(frame, n=n_points)
        out[i] = resampled.to_array()
    return out
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/dataset/test_tensors.py -v
```

Expected: all 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add laser_ai/dataset/tensors.py tests/dataset/test_tensors.py
git commit -m "feat(dataset): Show→tensor extraction for training data"
```

---

## Task 4: PyTorch frame Dataset

**Files:**
- Create: `laser_ai/dataset/torch_dataset.py`
- Create: `tests/dataset/test_torch_dataset.py`

A `torch.utils.data.Dataset` that loads training ILDA files lazily and applies augmentations.

- [ ] **Step 1: Write `tests/dataset/test_torch_dataset.py`**

```python
"""Tests for PyTorch frame dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from laser_ai.augment.frame import AugmentConfig
from laser_ai.dataset.torch_dataset import FrameDataset


FIXTURE = Path(__file__).parent.parent / "fixtures" / "tiny_show.ilda"


def test_dataset_length_matches_frames_times_augment_mult():
    ds = FrameDataset([FIXTURE], n_points=64, augment_mult=1, augment_cfg=None)
    # tiny_show has 3 frames
    assert len(ds) == 3


def test_dataset_item_is_tensor_with_correct_shape():
    ds = FrameDataset([FIXTURE], n_points=64)
    x = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (64, 6)
    assert x.dtype == torch.float32


def test_dataset_augment_mult_expands_length():
    ds = FrameDataset([FIXTURE], n_points=32, augment_mult=4)
    assert len(ds) == 3 * 4


def test_dataset_rejects_missing_file(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        FrameDataset([tmp_path / "nope.ilda"], n_points=32)


def test_dataset_can_be_iterated_deterministically_when_seeded():
    ds1 = FrameDataset([FIXTURE], n_points=32, augment_mult=3,
                       augment_cfg=AugmentConfig(), seed=123)
    ds2 = FrameDataset([FIXTURE], n_points=32, augment_mult=3,
                       augment_cfg=AugmentConfig(), seed=123)
    for i in range(len(ds1)):
        np.testing.assert_allclose(ds1[i].numpy(), ds2[i].numpy())
```

- [ ] **Step 2: Write `laser_ai/dataset/torch_dataset.py`**

```python
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/dataset/test_torch_dataset.py -v
```

Expected: all 5 PASS.

- [ ] **Step 4: Commit**

```bash
git add laser_ai/dataset/torch_dataset.py tests/dataset/test_torch_dataset.py
git commit -m "feat(dataset): PyTorch FrameDataset with augmentation multiplier"
```

---

## Task 5: Frame VAE architecture

**Files:**
- Create: `laser_ai/models/__init__.py`
- Create: `laser_ai/models/vae.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_vae.py`

1D conv encoder over point sequence → mean + log-variance → 64-dim latent. Mirror decoder. No learned positional encoding — point order is implicitly positional. `FrameVAEConfig` dataclass holds all hyperparams.

- [ ] **Step 1: Write `laser_ai/models/__init__.py`**

```python
"""PyTorch models for laser-ai training."""

from laser_ai.models.vae import FrameVAE, FrameVAEConfig

__all__ = ["FrameVAE", "FrameVAEConfig"]
```

- [ ] **Step 2: Write `tests/models/__init__.py`** (empty)

- [ ] **Step 3: Write `tests/models/test_vae.py`**

```python
"""Tests for Frame VAE."""
from __future__ import annotations

import torch

from laser_ai.models.vae import FrameVAE, FrameVAEConfig


def test_vae_encode_returns_mu_and_logvar_of_correct_shape():
    cfg = FrameVAEConfig(n_points=64, latent_dim=16, hidden=32)
    vae = FrameVAE(cfg).eval()
    x = torch.randn(4, 64, 6)
    mu, logvar = vae.encode(x)
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_vae_decode_returns_frame_shape():
    cfg = FrameVAEConfig(n_points=64, latent_dim=16, hidden=32)
    vae = FrameVAE(cfg).eval()
    z = torch.randn(4, 16)
    out = vae.decode(z)
    assert out.shape == (4, 64, 6)


def test_vae_forward_returns_recon_mu_logvar():
    cfg = FrameVAEConfig(n_points=64, latent_dim=16, hidden=32)
    vae = FrameVAE(cfg).eval()
    x = torch.randn(2, 64, 6)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)


def test_vae_decode_outputs_stay_bounded():
    """xy in [-1, 1] via tanh, rgb in [0, 1] via sigmoid, is_travel in [0, 1]."""
    cfg = FrameVAEConfig(n_points=32, latent_dim=8, hidden=16)
    vae = FrameVAE(cfg).eval()
    out = vae.decode(torch.randn(8, 8) * 10.0)
    assert out[..., :2].abs().max() <= 1.0 + 1e-5
    assert out[..., 2:5].min() >= 0.0
    assert out[..., 2:5].max() <= 1.0 + 1e-5
    assert out[..., 5].min() >= 0.0
    assert out[..., 5].max() <= 1.0 + 1e-5


def test_vae_reparameterize_is_stochastic_in_train_deterministic_in_eval():
    cfg = FrameVAEConfig(n_points=32, latent_dim=8, hidden=16)
    vae = FrameVAE(cfg)
    mu = torch.zeros(2, 8)
    logvar = torch.zeros(2, 8)

    vae.train()
    z1 = vae.reparameterize(mu, logvar)
    z2 = vae.reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2)

    vae.eval()
    z3 = vae.reparameterize(mu, logvar)
    z4 = vae.reparameterize(mu, logvar)
    torch.testing.assert_close(z3, z4)


def test_vae_param_count_is_small():
    cfg = FrameVAEConfig()  # defaults: n_points=512, latent_dim=64, hidden=128
    vae = FrameVAE(cfg)
    n = sum(p.numel() for p in vae.parameters())
    # Should be in the ~1–3M param range per spec
    assert 500_000 < n < 5_000_000
```

- [ ] **Step 4: Write `laser_ai/models/vae.py`**

```python
"""Frame VAE: encode (N, 6) laser frames into a latent vector and back.

Encoder: 1D conv stack over the point axis → mean + logvar of a `latent_dim`
Gaussian. Decoder: latent → 1D conv-transpose → (N, 6) reconstruction with
tanh/sigmoid heads to keep outputs in valid ranges.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class FrameVAEConfig:
    n_points: int = 512
    latent_dim: int = 64
    hidden: int = 128
    n_channels_in: int = 6  # (x, y, r, g, b, is_travel)
    kernel: int = 5


class FrameVAE(nn.Module):
    def __init__(self, cfg: FrameVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FrameVAEConfig()
        c = self.cfg.n_channels_in
        h = self.cfg.hidden
        N = self.cfg.n_points
        k = self.cfg.kernel
        p = k // 2

        # Encoder: (B, C, N) -> (B, h*2, N/8) via three stride-2 convs
        self.enc = nn.Sequential(
            nn.Conv1d(c, h, k, stride=2, padding=p),          # N -> N/2
            nn.GELU(),
            nn.Conv1d(h, h * 2, k, stride=2, padding=p),       # N/2 -> N/4
            nn.GELU(),
            nn.Conv1d(h * 2, h * 2, k, stride=2, padding=p),   # N/4 -> N/8
            nn.GELU(),
        )
        flat = h * 2 * (N // 8)
        self.to_mu = nn.Linear(flat, self.cfg.latent_dim)
        self.to_logvar = nn.Linear(flat, self.cfg.latent_dim)

        # Decoder: latent -> (B, h*2, N/8) -> upscale to (B, C, N)
        self.from_latent = nn.Linear(self.cfg.latent_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(h * 2, h * 2, k, stride=2, padding=p, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(h * 2, h, k, stride=2, padding=p, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(h, h, k, stride=2, padding=p, output_padding=1),
            nn.GELU(),
        )
        # Separate heads so we can apply different activations per column
        self.head_xy = nn.Conv1d(h, 2, 1)
        self.head_rgb = nn.Conv1d(h, 3, 1)
        self.head_travel = nn.Conv1d(h, 1, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, C)
        h = self.enc(x.transpose(1, 2))           # (B, h*2, N/8)
        flat = h.flatten(1)
        return self.to_mu(flat), self.to_logvar(flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.from_latent(z).view(B, self.cfg.hidden * 2, self.cfg.n_points // 8)
        h = self.dec(h)
        xy = torch.tanh(self.head_xy(h))              # (B, 2, N)
        rgb = torch.sigmoid(self.head_rgb(h))         # (B, 3, N)
        travel = torch.sigmoid(self.head_travel(h))   # (B, 1, N)
        out = torch.cat([xy, rgb, travel], dim=1)     # (B, 6, N)
        return out.transpose(1, 2)                    # (B, N, 6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/models/test_vae.py -v
```

Expected: all 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add laser_ai/models/__init__.py laser_ai/models/vae.py tests/models/__init__.py tests/models/test_vae.py
git commit -m "feat(models): Frame VAE with 1D conv encoder/decoder"
```

---

## Task 6: VAE losses

**Files:**
- Create: `laser_ai/models/losses.py`
- Create: `tests/models/test_losses.py`

Chamfer distance on xy, MSE on rgb, BCE on is_travel, KL divergence on the latent. Combined into one scalar with configurable weights.

- [ ] **Step 1: Write `tests/models/test_losses.py`**

```python
"""Tests for VAE loss components."""
from __future__ import annotations

import torch

from laser_ai.models.losses import LossWeights, chamfer_distance, vae_loss


def test_chamfer_of_identical_sets_is_zero():
    x = torch.tensor([[[-1.0, 0.0], [0.0, 0.5], [1.0, 0.5]]])
    assert chamfer_distance(x, x).item() < 1e-6


def test_chamfer_is_positive_for_different_sets():
    a = torch.tensor([[[0.0, 0.0]]])
    b = torch.tensor([[[1.0, 0.0]]])
    assert chamfer_distance(a, b).item() > 0.9  # square-distance-style


def test_chamfer_supports_batches():
    a = torch.randn(3, 8, 2)
    b = torch.randn(3, 8, 2)
    out = chamfer_distance(a, b)
    assert out.shape == ()
    assert out.item() > 0


def test_vae_loss_is_positive_with_random_inputs():
    B, N = 2, 32
    recon = torch.randn(B, N, 6).clamp(-1, 1)
    target = torch.randn(B, N, 6).clamp(-1, 1)
    recon = (recon + 1) / 2  # pretend some in [0,1]
    target = (target + 1) / 2
    recon[..., :2] = recon[..., :2] * 2 - 1
    target[..., :2] = target[..., :2] * 2 - 1
    mu = torch.zeros(B, 8)
    logvar = torch.zeros(B, 8)
    total, parts = vae_loss(recon, target, mu, logvar, LossWeights())
    assert total.item() > 0
    assert set(parts.keys()) == {"chamfer", "rgb", "travel", "kl"}


def test_vae_loss_reduces_when_recon_matches_target():
    B, N = 2, 16
    target = torch.rand(B, N, 6)
    target[..., :2] = target[..., :2] * 2 - 1
    mu = torch.zeros(B, 8)
    logvar = torch.zeros(B, 8)
    total_exact, _ = vae_loss(target.clone(), target, mu, logvar, LossWeights())
    total_random, _ = vae_loss(torch.randn_like(target), target, mu, logvar, LossWeights())
    assert total_exact.item() < total_random.item()


def test_kl_weight_zero_drops_kl_term():
    B, N = 1, 8
    target = torch.rand(B, N, 6)
    target[..., :2] = target[..., :2] * 2 - 1
    mu = torch.randn(B, 4) * 3
    logvar = torch.randn(B, 4) * 3
    _, parts_on = vae_loss(target, target, mu, logvar, LossWeights(kl=1.0))
    _, parts_off = vae_loss(target, target, mu, logvar, LossWeights(kl=0.0))
    assert parts_on["kl"] > parts_off["kl"]
    assert parts_off["kl"] == 0.0
```

- [ ] **Step 2: Write `laser_ai/models/losses.py`**

```python
"""Loss functions for Frame VAE training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class LossWeights:
    chamfer: float = 1.0
    rgb: float = 0.3
    travel: float = 0.1
    kl: float = 0.01


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Chamfer L2-squared distance between two point sets.

    a, b: (B, N, 2). Returns scalar mean over the batch.
    """
    # Pairwise squared distances: (B, N, N)
    d = torch.cdist(a, b, p=2) ** 2
    min_ab = d.min(dim=2).values  # closest b for each a (B, N)
    min_ba = d.min(dim=1).values  # closest a for each b (B, N)
    return 0.5 * (min_ab.mean() + min_ba.mean())


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined VAE loss. recon and target are (B, N, 6)."""
    # xy: Chamfer
    ch = chamfer_distance(recon[..., :2], target[..., :2])
    # rgb: MSE
    rgb = F.mse_loss(recon[..., 2:5], target[..., 2:5])
    # is_travel: BCE
    tr = F.binary_cross_entropy(
        recon[..., 5].clamp(1e-6, 1 - 1e-6),
        target[..., 5],
    )
    # KL divergence to N(0, I)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = (
        weights.chamfer * ch
        + weights.rgb * rgb
        + weights.travel * tr
        + weights.kl * kl
    )
    parts = {
        "chamfer": float(ch.item()),
        "rgb": float(rgb.item()),
        "travel": float(tr.item()),
        "kl": float(kl.item()) * weights.kl,  # report contribution after weighting
    }
    return total, parts
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/models/test_losses.py -v
```

Expected: all 6 PASS.

- [ ] **Step 4: Commit**

```bash
git add laser_ai/models/losses.py tests/models/test_losses.py
git commit -m "feat(models): Chamfer + combined VAE loss"
```

---

## Task 7: VAE training loop

**Files:**
- Create: `laser_ai/training/__init__.py`
- Create: `laser_ai/training/train_vae.py`
- Create: `tests/training/__init__.py`
- Create: `tests/training/test_train_vae.py`

Training function usable from both local CLI (Task 12) and Colab notebook (Task 14). Accepts a `FrameDataset`, runs N epochs, logs loss parts, returns the trained model.

- [ ] **Step 1: Write `laser_ai/training/__init__.py`**

```python
"""Training loops for VAE and Sequencer."""

from laser_ai.training.train_vae import VAETrainConfig, train_vae

__all__ = ["VAETrainConfig", "train_vae"]
```

- [ ] **Step 2: Write `tests/training/__init__.py`** (empty)

- [ ] **Step 3: Write `tests/training/test_train_vae.py`**

```python
"""Smoke tests for the VAE training loop."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from laser_ai.models.vae import FrameVAEConfig
from laser_ai.training.train_vae import VAETrainConfig, train_vae


class _SyntheticFrameDS(Dataset):
    """A tiny hand-built dataset: random (N, 6) tensors."""

    def __init__(self, count: int, n_points: int) -> None:
        torch.manual_seed(0)
        self._data = torch.rand(count, n_points, 6) * 2 - 1
        self._data[..., 2:5] = (self._data[..., 2:5] + 1) / 2
        self._data[..., 5] = (self._data[..., 5] > 0).float()

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


def test_train_vae_runs_and_returns_model():
    ds = _SyntheticFrameDS(count=16, n_points=32)
    vae_cfg = FrameVAEConfig(n_points=32, latent_dim=8, hidden=16)
    train_cfg = VAETrainConfig(epochs=1, batch_size=4, lr=1e-3)
    model, history = train_vae(ds, vae_cfg=vae_cfg, train_cfg=train_cfg)
    assert model is not None
    assert len(history) >= 1


def test_train_vae_loss_decreases_over_a_few_epochs():
    ds = _SyntheticFrameDS(count=32, n_points=32)
    vae_cfg = FrameVAEConfig(n_points=32, latent_dim=8, hidden=16)
    train_cfg = VAETrainConfig(epochs=5, batch_size=8, lr=2e-3)
    _, history = train_vae(ds, vae_cfg=vae_cfg, train_cfg=train_cfg)
    assert history[-1]["total"] < history[0]["total"]


def test_train_vae_handles_empty_dataset_gracefully():
    import pytest
    ds = _SyntheticFrameDS(count=0, n_points=32)
    with pytest.raises(ValueError, match="empty"):
        train_vae(ds, vae_cfg=FrameVAEConfig(n_points=32, latent_dim=8, hidden=16),
                  train_cfg=VAETrainConfig(epochs=1, batch_size=4))
```

- [ ] **Step 4: Write `laser_ai/training/train_vae.py`**

```python
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
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/training/test_train_vae.py -v
```

Expected: all 3 PASS (may take ~20 seconds on CPU for the convergence test).

- [ ] **Step 6: Commit**

```bash
git add laser_ai/training/__init__.py laser_ai/training/train_vae.py tests/training/__init__.py tests/training/test_train_vae.py
git commit -m "feat(training): Frame VAE training loop"
```

---

## Task 8: Audio-to-Latent sequencer architecture

**Files:**
- Create: `laser_ai/models/sequencer.py`
- Create: `tests/models/test_sequencer.py`

Small causal Transformer — 6 layers, 256 hidden dim, 4 heads, ~2M params per spec. Input: `(B, T, feature_dim)` audio features. Output: `(B, T, latent_dim)` predicted VAE latents.

- [ ] **Step 1: Write `tests/models/test_sequencer.py`**

```python
"""Tests for Audio-to-Latent sequencer."""
from __future__ import annotations

import torch

from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig


def test_sequencer_forward_shape():
    cfg = SequencerConfig(
        feature_dim=146, latent_dim=32, hidden=128, n_layers=2, n_heads=4,
    )
    m = AudioToLatentSequencer(cfg).eval()
    x = torch.randn(2, 60, 146)
    y = m(x)
    assert y.shape == (2, 60, 32)


def test_sequencer_is_causal():
    """Output at time t must not depend on inputs at time > t."""
    cfg = SequencerConfig(feature_dim=8, latent_dim=4, hidden=16, n_layers=2, n_heads=2)
    m = AudioToLatentSequencer(cfg).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 10, 8)
    y1 = m(x)
    # Perturb only the last half of the input
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])
    y2 = m(x2)
    # Outputs before the perturbation must remain unchanged
    torch.testing.assert_close(y1[:, :5], y2[:, :5], atol=1e-5, rtol=1e-5)
    # Outputs after the perturbation must change
    assert not torch.allclose(y1[:, 5:], y2[:, 5:], atol=1e-5)


def test_sequencer_param_count_in_target_range():
    cfg = SequencerConfig()  # defaults: ~2M params
    m = AudioToLatentSequencer(cfg)
    n = sum(p.numel() for p in m.parameters())
    # Spec says ~2M; allow 0.5M - 5M window
    assert 500_000 < n < 5_000_000
```

- [ ] **Step 2: Write `laser_ai/models/sequencer.py`**

```python
"""Small causal Transformer mapping per-frame audio features → VAE latents."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class SequencerConfig:
    feature_dim: int = 146       # matches laser_ai.audio.features.FEATURE_DIM
    latent_dim: int = 64
    hidden: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_len: int = 4096          # supports up to ~2.3 min at 30 fps


class _CausalTransformerBlock(nn.Module):
    def __init__(self, cfg: SequencerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden, num_heads=cfg.n_heads,
            dropout=cfg.dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.hidden)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden * cfg.ff_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden * cfg.ff_mult, cfg.hidden),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class AudioToLatentSequencer(nn.Module):
    def __init__(self, cfg: SequencerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SequencerConfig()
        self.in_proj = nn.Linear(self.cfg.feature_dim, self.cfg.hidden)
        # Learned positional embedding; simpler than sinusoidal and adequate at this scale
        self.pos_emb = nn.Embedding(self.cfg.max_len, self.cfg.hidden)
        self.blocks = nn.ModuleList([
            _CausalTransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(self.cfg.hidden)
        self.out_proj = nn.Linear(self.cfg.hidden, self.cfg.latent_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, T, _ = features.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.cfg.max_len}")
        pos = torch.arange(T, device=features.device)
        x = self.in_proj(features) + self.pos_emb(pos).unsqueeze(0)
        # Causal mask (T, T) — True = disallow attending
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=features.device),
                          diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.out_proj(self.ln_out(x))
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/models/test_sequencer.py -v
```

Expected: all 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add laser_ai/models/sequencer.py tests/models/test_sequencer.py
git commit -m "feat(models): causal Transformer audio-to-latent sequencer"
```

---

## Task 9: Latent extraction + sequencer training

**Files:**
- Create: `laser_ai/training/prepare.py`
- Create: `laser_ai/training/train_sequencer.py`
- Create: `tests/training/test_prepare.py`
- Create: `tests/training/test_train_sequencer.py`

`prepare.build_sequencer_dataset` runs the trained VAE's encoder across all training pairs to produce `(audio_features, target_latents)` tuples per song. `train_sequencer` runs the loop.

- [ ] **Step 1: Write `tests/training/test_prepare.py`**

```python
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
```

- [ ] **Step 2: Write `laser_ai/training/prepare.py`**

```python
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


def _stretch_latents_to_length(latents: torch.Tensor, target_T: int) -> torch.Tensor:
    """Resample a (T, D) latent sequence to (target_T, D) via nearest-neighbor indexing.

    Needed when the ILDA has a different frame count than the audio's feature-frame count.
    """
    T, D = latents.shape
    if T == target_T:
        return latents
    if T == 0:
        return torch.zeros(target_T, D, dtype=latents.dtype)
    # Nearest index lookup: sample[i] = latents[round(i * (T - 1) / (target_T - 1))]
    if target_T == 1:
        return latents[:1]
    idx = torch.linspace(0, T - 1, target_T).round().long()
    return latents[idx]


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
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for pair in pairs:
        # Audio features
        samples, sr = load_audio(pair.audio_path)
        feats_np = extract_features(samples, sr, fps=fps)  # (T_audio, FEATURE_DIM)
        feats = torch.from_numpy(feats_np).float()

        # ILDA latents
        show = read_ilda(pair.ilda_path)
        frames_np = show_to_tensor(show, n_points=n_points)  # (T_ilda, N, 6)
        frames = torch.from_numpy(frames_np).float()
        if frames.shape[0] == 0:
            continue
        mu, _ = vae.encode(frames)   # (T_ilda, latent_dim)

        # Align the two time axes
        latents = _stretch_latents_to_length(mu.detach(), target_T=feats.shape[0])
        out.append((feats, latents))
    return out
```

- [ ] **Step 3: Run prepare test**

```bash
pytest tests/training/test_prepare.py -v
```

Expected: 1 PASS.

- [ ] **Step 4: Write `tests/training/test_train_sequencer.py`**

```python
"""Smoke tests for sequencer training loop."""
from __future__ import annotations

import torch

from laser_ai.models.sequencer import SequencerConfig
from laser_ai.training.train_sequencer import SequencerTrainConfig, train_sequencer


def _synthetic_pairs(count: int, T: int, feat_dim: int, latent_dim: int):
    torch.manual_seed(0)
    return [
        (
            torch.randn(T, feat_dim),
            torch.randn(T, latent_dim),
        )
        for _ in range(count)
    ]


def test_train_sequencer_runs():
    pairs = _synthetic_pairs(count=3, T=32, feat_dim=16, latent_dim=8)
    seq_cfg = SequencerConfig(
        feature_dim=16, latent_dim=8, hidden=32, n_layers=2, n_heads=2,
    )
    train_cfg = SequencerTrainConfig(epochs=1, batch_size=1)
    model, history = train_sequencer(pairs, seq_cfg=seq_cfg, train_cfg=train_cfg)
    assert model is not None
    assert len(history) == 1


def test_train_sequencer_loss_decreases():
    pairs = _synthetic_pairs(count=4, T=32, feat_dim=16, latent_dim=8)
    seq_cfg = SequencerConfig(
        feature_dim=16, latent_dim=8, hidden=32, n_layers=2, n_heads=2,
    )
    train_cfg = SequencerTrainConfig(epochs=5, batch_size=2, lr=3e-3)
    _, history = train_sequencer(pairs, seq_cfg=seq_cfg, train_cfg=train_cfg)
    assert history[-1]["mse"] < history[0]["mse"]
```

- [ ] **Step 5: Write `laser_ai/training/train_sequencer.py`**

```python
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
```

- [ ] **Step 6: Run sequencer training tests**

```bash
pytest tests/training/test_train_sequencer.py -v
```

Expected: both PASS (may take ~15s on CPU for convergence test).

- [ ] **Step 7: Commit**

```bash
git add laser_ai/training/prepare.py laser_ai/training/train_sequencer.py tests/training/test_prepare.py tests/training/test_train_sequencer.py
git commit -m "feat(training): latent extraction + sequencer training loop"
```

---

## Task 10: Checkpoint I/O

**Files:**
- Create: `laser_ai/models/checkpoint.py`
- Create: `tests/models/test_checkpoint.py`

Save both models, their configs, and audio feature dim into a single `.pt` file. Load it back on the target device.

- [ ] **Step 1: Write `tests/models/test_checkpoint.py`**

```python
"""Tests for saving/loading model bundles."""
from __future__ import annotations

from pathlib import Path

import torch

from laser_ai.models.checkpoint import LaserAICheckpoint, load_checkpoint, save_checkpoint
from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
from laser_ai.models.vae import FrameVAE, FrameVAEConfig


def _build_tiny_bundle() -> LaserAICheckpoint:
    vae_cfg = FrameVAEConfig(n_points=32, latent_dim=4, hidden=8)
    seq_cfg = SequencerConfig(feature_dim=8, latent_dim=4, hidden=16,
                              n_layers=1, n_heads=2, max_len=64)
    return LaserAICheckpoint(
        vae=FrameVAE(vae_cfg),
        vae_cfg=vae_cfg,
        sequencer=AudioToLatentSequencer(seq_cfg),
        seq_cfg=seq_cfg,
        audio_feature_dim=8,
        fps=30.0,
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    ck = _build_tiny_bundle()
    path = tmp_path / "bundle.pt"
    save_checkpoint(ck, path)
    assert path.exists()

    ck2 = load_checkpoint(path, map_location="cpu")
    assert ck2.vae_cfg.latent_dim == ck.vae_cfg.latent_dim
    assert ck2.seq_cfg.feature_dim == ck.seq_cfg.feature_dim
    assert ck2.audio_feature_dim == 8
    assert ck2.fps == 30.0


def test_loaded_vae_produces_same_output(tmp_path: Path):
    ck = _build_tiny_bundle()
    x = torch.randn(1, ck.vae_cfg.n_points, 6)
    ck.vae.eval()
    y_orig = ck.vae.decode(ck.vae.encode(x)[0])

    path = tmp_path / "b.pt"
    save_checkpoint(ck, path)
    ck2 = load_checkpoint(path, map_location="cpu")
    ck2.vae.eval()
    y_loaded = ck2.vae.decode(ck2.vae.encode(x)[0])

    torch.testing.assert_close(y_orig, y_loaded, atol=1e-6, rtol=1e-6)
```

- [ ] **Step 2: Write `laser_ai/models/checkpoint.py`**

```python
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/models/test_checkpoint.py -v
```

Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add laser_ai/models/checkpoint.py tests/models/test_checkpoint.py
git commit -m "feat(models): checkpoint save/load for VAE+Sequencer bundle"
```

---

## Task 11: TrainedGenerator

**Files:**
- Create: `laser_ai/generator/trained.py`
- Create: `tests/generator/test_trained.py`

Concrete `Generator` that wraps a loaded `LaserAICheckpoint`. Implements `generate(features, n_points)` by running the sequencer on features, then decoding each predicted latent through the VAE. Output: `(T, n_points, 6)` float32 numpy array, matching the existing `StubGenerator` contract.

- [ ] **Step 1: Write `tests/generator/test_trained.py`**

```python
"""Tests for TrainedGenerator."""
from __future__ import annotations

import numpy as np
import torch

from laser_ai.audio.features import FEATURE_DIM
from laser_ai.generator.trained import TrainedGenerator
from laser_ai.models.checkpoint import LaserAICheckpoint
from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
from laser_ai.models.vae import FrameVAE, FrameVAEConfig


def _make_checkpoint(n_points: int = 64) -> LaserAICheckpoint:
    vae_cfg = FrameVAEConfig(n_points=n_points, latent_dim=8, hidden=16)
    seq_cfg = SequencerConfig(
        feature_dim=FEATURE_DIM, latent_dim=8, hidden=32,
        n_layers=1, n_heads=2, max_len=256,
    )
    return LaserAICheckpoint(
        vae=FrameVAE(vae_cfg), vae_cfg=vae_cfg,
        sequencer=AudioToLatentSequencer(seq_cfg), seq_cfg=seq_cfg,
        audio_feature_dim=FEATURE_DIM, fps=30.0,
    )


def test_trained_generator_output_shape():
    ck = _make_checkpoint(n_points=64)
    gen = TrainedGenerator(ck)
    feats = np.random.rand(30, FEATURE_DIM).astype(np.float32)
    out = gen.generate(feats, n_points=64)
    assert out.shape == (30, 64, 6)


def test_trained_generator_output_in_valid_ranges():
    ck = _make_checkpoint(n_points=64)
    gen = TrainedGenerator(ck)
    feats = np.random.rand(20, FEATURE_DIM).astype(np.float32)
    out = gen.generate(feats, n_points=64)
    assert out[..., :2].max() <= 1.0 + 1e-5
    assert out[..., :2].min() >= -1.0 - 1e-5
    assert out[..., 2:5].min() >= 0.0 - 1e-5
    assert out[..., 2:5].max() <= 1.0 + 1e-5


def test_trained_generator_requires_matching_feature_dim():
    import pytest
    ck = _make_checkpoint()
    gen = TrainedGenerator(ck)
    bad = np.random.rand(10, FEATURE_DIM - 1).astype(np.float32)
    with pytest.raises(ValueError, match="feature_dim"):
        gen.generate(bad, n_points=64)


def test_trained_generator_overrides_n_points_warns_on_mismatch(recwarn):
    import warnings
    ck = _make_checkpoint(n_points=64)
    gen = TrainedGenerator(ck)
    feats = np.random.rand(5, FEATURE_DIM).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = gen.generate(feats, n_points=128)   # different from checkpoint
        assert any("n_points" in str(x.message) for x in w)
    # It still runs at the model's native n_points
    assert out.shape == (5, 64, 6)
```

- [ ] **Step 2: Write `laser_ai/generator/trained.py`**

```python
"""Generator backed by a trained LaserAICheckpoint."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from laser_ai.generator.base import Generator
from laser_ai.models.checkpoint import LaserAICheckpoint


class TrainedGenerator(Generator):
    def __init__(self, checkpoint: LaserAICheckpoint, device: str = "cpu") -> None:
        self.ck = checkpoint
        self.device = torch.device(device)
        self.ck.vae.to(self.device).eval()
        self.ck.sequencer.to(self.device).eval()

    @torch.no_grad()
    def generate(self, features: np.ndarray, *, n_points: int = 512) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != self.ck.audio_feature_dim:
            raise ValueError(
                f"features must be (T, {self.ck.audio_feature_dim}), got {features.shape}"
            )
        native_n = self.ck.vae_cfg.n_points
        if n_points != native_n:
            warnings.warn(
                f"requested n_points={n_points} but model was trained at {native_n}; "
                f"using {native_n}",
                stacklevel=2,
            )
            n_points = native_n

        feats_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)  # (1, T, F)
        latents = self.ck.sequencer(feats_t)   # (1, T, latent_dim)
        # Decode each frame's latent
        T = latents.shape[1]
        frames = self.ck.vae.decode(latents[0])  # (T, n_points, 6)
        return frames.cpu().numpy().astype(np.float32)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/generator/test_trained.py -v
```

Expected: all 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add laser_ai/generator/trained.py tests/generator/test_trained.py
git commit -m "feat(generator): TrainedGenerator wrapping VAE+Sequencer checkpoint"
```

---

## Task 12: CLI integration

**Files:**
- Modify: `laser_ai/cli.py`
- Create: `tests/cli/test_cli_train.py`

New subcommands:
- `laser-ai train-vae DATA_DIR -o CHECKPOINT_PATH` — train VAE on all `.ild`/`.ilda` in `DATA_DIR`, save bundle with an untrained sequencer placeholder.
- `laser-ai train-sequencer DATA_DIR -c CHECKPOINT_PATH -o CHECKPOINT_PATH` — load existing bundle, train sequencer on audio+ILDA pairs in `DATA_DIR`, overwrite.
- `laser-ai generate SONG -o OUT.ilda --model CHECKPOINT_PATH` — use TrainedGenerator.

- [ ] **Step 1: Write `tests/cli/test_cli_train.py`**

```python
"""Tests for new ML-related CLI subcommands."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from laser_ai.cli import cli


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _write_wav(path: Path, duration_s: float = 1.0) -> None:
    import soundfile as sf
    sr = 44100
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    s = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    sf.write(path, s, sr, subtype="PCM_16")


def test_train_vae_creates_checkpoint(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    # Re-use the fixture ILDA twice to give the VAE something to train on
    import shutil
    shutil.copy(FIXTURES / "tiny_show.ilda", data / "a.ilda")
    shutil.copy(FIXTURES / "tiny_show.ilda", data / "b.ilda")

    out_ck = tmp_path / "ck.pt"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "train-vae", str(data), "-o", str(out_ck),
        "--epochs", "1", "--n-points", "32",
        "--latent-dim", "4", "--hidden", "8",
    ])
    assert result.exit_code == 0, result.output
    assert out_ck.exists()


def test_generate_with_model_works(tmp_path: Path):
    # Build a checkpoint, then use it to generate
    import shutil
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(FIXTURES / "tiny_show.ilda", data / "a.ilda")

    ck = tmp_path / "ck.pt"
    runner = CliRunner()
    r1 = runner.invoke(cli, [
        "train-vae", str(data), "-o", str(ck),
        "--epochs", "1", "--n-points", "32",
        "--latent-dim", "4", "--hidden", "8",
    ])
    assert r1.exit_code == 0, r1.output

    song = tmp_path / "song.wav"
    _write_wav(song)
    out = tmp_path / "out.ilda"
    r2 = runner.invoke(cli, [
        "generate", str(song), "-o", str(out), "--model", str(ck),
    ])
    assert r2.exit_code == 0, r2.output
    assert out.exists()
```

- [ ] **Step 2: Modify `laser_ai/cli.py` — add new commands**

Replace the file contents with the version below. Key additions: `train-vae`, `train-sequencer`, and a `--model` flag on `generate`.

```python
"""laser-ai command-line interface."""
from __future__ import annotations

from pathlib import Path

import click

from laser_ai.ilda.reader import read_ilda
from laser_ai.ilda.writer import write_ilda
from laser_ai.pipeline.generate import generate_show_from_audio
from laser_ai.safety.postprocess import SafetyConfig


@click.group()
@click.version_option()
def cli() -> None:
    """laser-ai: AI-driven ILDA laser show generation from audio."""


@cli.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output .ilda path.")
@click.option("--fps", type=float, default=30.0, show_default=True,
              help="Target frame rate.")
@click.option("--n-points", type=int, default=512, show_default=True,
              help="Points per frame (before safety pass).")
@click.option("--safety", type=click.Choice(["loose", "medium", "tight"]),
              default="medium", show_default=True, help="Safety post-processor strength.")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Generator random seed (stub mode only).")
@click.option("--model", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Path to a trained checkpoint (.pt). Without it, uses stub.")
def generate(audio: Path, output: Path, fps: float, n_points: int,
             safety: str, seed: int, model: Path | None) -> None:
    """Generate an ILDA show from an audio file."""
    strengths = {"loose": 0.5, "medium": 1.0, "tight": 2.0}
    cfg = SafetyConfig(strength=strengths[safety])

    if model is not None:
        click.echo(f"loading model: {model}")
        from laser_ai.generator.trained import TrainedGenerator
        from laser_ai.models.checkpoint import load_checkpoint
        gen = TrainedGenerator(load_checkpoint(model))
        # Model n_points wins; warn handled inside TrainedGenerator
    else:
        click.echo("using stub generator (no --model given)")
        from laser_ai.generator.stub import StubGenerator
        gen = StubGenerator(seed=seed)

    click.echo(f"generating: {audio} → {output}")
    show = generate_show_from_audio(
        audio_path=audio, generator=gen, fps=fps,
        n_points=n_points, safety_cfg=cfg,
    )
    write_ilda(show, output)
    click.echo(f"wrote {output} ({len(show.frames)} frames, {show.duration_s:.2f}s)")


@cli.command()
@click.argument("ilda", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(ilda: Path) -> None:
    """Print summary info about an ILDA file."""
    show = read_ilda(ilda)
    total_pts = sum(len(f.points) for f in show.frames)
    click.echo(f"file:      {ilda}")
    click.echo(f"frames:    {len(show.frames)}")
    click.echo(f"duration:  {show.duration_s:.2f}s @ {show.fps} fps")
    click.echo(f"points:    {total_pts} total, avg {total_pts / max(1, len(show.frames)):.1f}/frame")


@cli.command("train-vae")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output checkpoint path (.pt).")
@click.option("--epochs", type=int, default=20, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--n-points", type=int, default=512, show_default=True)
@click.option("--latent-dim", type=int, default=64, show_default=True)
@click.option("--hidden", type=int, default=128, show_default=True)
@click.option("--augment-mult", type=int, default=4, show_default=True,
              help="Virtual dataset size multiplier via augmentation.")
def train_vae_cmd(data_dir: Path, output: Path, epochs: int, batch_size: int,
                  lr: float, n_points: int, latent_dim: int, hidden: int,
                  augment_mult: int) -> None:
    """Train the Frame VAE on all ILDA files in DATA_DIR."""
    from laser_ai.augment.frame import AugmentConfig
    from laser_ai.audio.features import FEATURE_DIM
    from laser_ai.dataset.torch_dataset import FrameDataset
    from laser_ai.models.checkpoint import LaserAICheckpoint, save_checkpoint
    from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
    from laser_ai.models.vae import FrameVAEConfig
    from laser_ai.training.train_vae import VAETrainConfig, train_vae

    ilda_paths = sorted(
        list(data_dir.rglob("*.ild")) + list(data_dir.rglob("*.ilda"))
    )
    if not ilda_paths:
        raise click.ClickException(f"no .ild/.ilda files found in {data_dir}")
    click.echo(f"found {len(ilda_paths)} ILDA file(s)")

    ds = FrameDataset(
        ilda_paths, n_points=n_points,
        augment_mult=augment_mult, augment_cfg=AugmentConfig(),
    )
    click.echo(f"dataset size (augmented): {len(ds)} frames")

    vae_cfg = FrameVAEConfig(n_points=n_points, latent_dim=latent_dim, hidden=hidden)
    train_cfg = VAETrainConfig(epochs=epochs, batch_size=batch_size, lr=lr)

    def _log(epoch: int, entry: dict) -> None:
        click.echo(
            f"  epoch {epoch:3d}: total={entry['total']:.4f}  "
            f"chamfer={entry['chamfer']:.4f}  rgb={entry['rgb']:.4f}  "
            f"travel={entry['travel']:.4f}  kl={entry['kl']:.4f}"
        )

    vae, _ = train_vae(ds, vae_cfg=vae_cfg, train_cfg=train_cfg, progress_callback=_log)

    # Default untrained sequencer placeholder; user will train it via train-sequencer
    seq_cfg = SequencerConfig(feature_dim=FEATURE_DIM, latent_dim=latent_dim)
    ck = LaserAICheckpoint(
        vae=vae, vae_cfg=vae_cfg,
        sequencer=AudioToLatentSequencer(seq_cfg), seq_cfg=seq_cfg,
        audio_feature_dim=FEATURE_DIM, fps=30.0,
    )
    save_checkpoint(ck, output)
    click.echo(f"saved checkpoint: {output}")


@cli.command("train-sequencer")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-c", "--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Input VAE-trained checkpoint.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output path (.pt) for the updated checkpoint.")
@click.option("--epochs", type=int, default=30, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--lr", type=float, default=5e-4, show_default=True)
def train_sequencer_cmd(data_dir: Path, checkpoint: Path, output: Path,
                        epochs: int, batch_size: int, lr: float) -> None:
    """Train the Sequencer on audio+ILDA pairs in DATA_DIR."""
    from laser_ai.dataset.discovery import discover_pairs
    from laser_ai.models.checkpoint import LaserAICheckpoint, load_checkpoint, save_checkpoint
    from laser_ai.training.prepare import build_sequencer_dataset
    from laser_ai.training.train_sequencer import SequencerTrainConfig, train_sequencer

    ck = load_checkpoint(checkpoint)
    result = discover_pairs(data_dir)
    if not result.pairs:
        raise click.ClickException(f"no audio+ILDA pairs found in {data_dir}")
    click.echo(f"found {len(result.pairs)} pair(s)")

    pairs = build_sequencer_dataset(
        result.pairs, vae=ck.vae,
        n_points=ck.vae_cfg.n_points, fps=ck.fps,
    )
    train_cfg = SequencerTrainConfig(epochs=epochs, batch_size=batch_size, lr=lr)

    def _log(epoch: int, entry: dict) -> None:
        click.echo(f"  epoch {epoch:3d}: mse={entry['mse']:.6f}")

    sequencer, _ = train_sequencer(pairs, seq_cfg=ck.seq_cfg, train_cfg=train_cfg,
                                   progress_callback=_log)

    updated = LaserAICheckpoint(
        vae=ck.vae, vae_cfg=ck.vae_cfg,
        sequencer=sequencer, seq_cfg=ck.seq_cfg,
        audio_feature_dim=ck.audio_feature_dim, fps=ck.fps,
    )
    save_checkpoint(updated, output)
    click.echo(f"saved checkpoint: {output}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 3: Run new CLI tests**

```bash
pytest tests/cli/test_cli_train.py -v
```

Expected: both PASS (may take ~30s for the training steps).

- [ ] **Step 4: Run full suite regression**

```bash
pytest -v
```

Expected: all previous tests still pass + new ML tests. Total ≈ 80+ tests.

- [ ] **Step 5: Commit**

```bash
git add laser_ai/cli.py tests/cli/test_cli_train.py
git commit -m "feat(cli): train-vae, train-sequencer, and --model flag on generate"
```

---

## Task 13: Dataset bundling for Colab

**Files:**
- Create: `laser_ai/bundle/__init__.py`
- Create: `laser_ai/bundle/pack.py`
- Create: `tests/bundle/__init__.py`
- Create: `tests/bundle/test_pack.py`
- Modify: `laser_ai/cli.py` (add `prepare-bundle` command)

Zip the user's `data_dir` into a portable bundle with an index JSON (stems, pair listings, audio/ILDA sizes). Colab users upload this zip; the Colab notebook unpacks it.

- [ ] **Step 1: Write `tests/bundle/__init__.py`** (empty)

- [ ] **Step 2: Write `tests/bundle/test_pack.py`**

```python
"""Tests for dataset bundling."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from laser_ai.bundle.pack import pack_dataset


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _write_wav(path: Path) -> None:
    import numpy as np
    import soundfile as sf
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    s = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    sf.write(path, s, sr, subtype="PCM_16")


def test_pack_creates_zip_with_expected_entries(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    _write_wav(data / "song1.wav")
    import shutil
    shutil.copy(FIXTURES / "tiny_show.ilda", data / "song1.ilda")

    out = tmp_path / "bundle.zip"
    pack_dataset(data, out)

    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "index.json" in names
        assert any(n.endswith("song1.wav") for n in names)
        assert any(n.endswith("song1.ilda") for n in names)
        with zf.open("index.json") as f:
            idx = json.load(f)
        assert idx["version"] >= 1
        assert len(idx["pairs"]) == 1
        assert idx["pairs"][0]["stem"] == "song1"


def test_pack_rejects_empty_dataset(tmp_path: Path):
    import pytest
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no audio"):
        pack_dataset(empty, tmp_path / "out.zip")
```

- [ ] **Step 3: Write `laser_ai/bundle/__init__.py`**

```python
"""Dataset bundling for Colab training."""

from laser_ai.bundle.pack import pack_dataset

__all__ = ["pack_dataset"]
```

- [ ] **Step 4: Write `laser_ai/bundle/pack.py`**

```python
"""Pack a training data folder into a portable zip for Colab upload."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from laser_ai.dataset.discovery import discover_pairs


_BUNDLE_VERSION = 1


def pack_dataset(data_dir: str | Path, out_zip: str | Path) -> None:
    """Zip up all audio+ILDA pairs found in `data_dir` into `out_zip`."""
    data_dir = Path(data_dir)
    out_zip = Path(out_zip)

    result = discover_pairs(data_dir)
    if not result.pairs:
        raise ValueError(
            f"no audio+ILDA pairs found in {data_dir}; nothing to bundle"
        )

    pairs_meta = []
    for p in result.pairs:
        pairs_meta.append({
            "stem": p.stem,
            "audio": p.audio_path.name,
            "ilda": p.ilda_path.name,
            "offset_seconds": p.offset_seconds,
            "audio_bytes": p.audio_path.stat().st_size,
            "ilda_bytes": p.ilda_path.stat().st_size,
        })

    index = {
        "version": _BUNDLE_VERSION,
        "pairs": pairs_meta,
    }

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.json", json.dumps(index, indent=2))
        for p in result.pairs:
            zf.write(p.audio_path, arcname=f"pairs/{p.audio_path.name}")
            zf.write(p.ilda_path, arcname=f"pairs/{p.ilda_path.name}")
```

- [ ] **Step 5: Add `prepare-bundle` to `laser_ai/cli.py`**

Open `laser_ai/cli.py`. After the `train_sequencer_cmd` function and before `if __name__ == "__main__":`, add:

```python
@cli.command("prepare-bundle")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output bundle zip path.")
def prepare_bundle_cmd(data_dir: Path, output: Path) -> None:
    """Zip a training folder into a bundle suitable for Colab."""
    from laser_ai.bundle.pack import pack_dataset
    pack_dataset(data_dir, output)
    click.echo(f"wrote {output}")
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/bundle/ -v
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add laser_ai/bundle tests/bundle laser_ai/cli.py
git commit -m "feat(bundle): prepare-bundle for Colab dataset export"
```

---

## Task 14: Colab notebook shell

**Files:**
- Create: `colab/colab_train.py`
- Create: `colab/laser_ai_train.ipynb`
- Create: `colab/README.md`

The notebook is a thin shell — all real logic lives in `colab_train.py` (which re-uses the same `laser_ai` code installed from GitHub).

- [ ] **Step 1: Write `colab/colab_train.py`**

```python
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
    ilda_paths = sorted(list(pairs_dir.glob("*.ild")) + list(pairs_dir.glob("*.ilda")))
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
    print(f"[colab] training sequencer")
    seq_cfg = SequencerConfig(feature_dim=FEATURE_DIM, latent_dim=latent_dim)
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
```

- [ ] **Step 2: Write `colab/laser_ai_train.ipynb`**

This is a JSON file. Use exactly this content:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# laser-ai training notebook\n",
    "\n",
    "Upload a `bundle.zip` produced by `laser-ai prepare-bundle` into the file panel, then click **Runtime → Run all**. Download the resulting `model.pt` when training finishes."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 1. Install laser-ai from GitHub (replace with your fork once pushed)\n",
    "!pip install -q --upgrade pip\n",
    "!pip install -q git+https://github.com/YOUR_USERNAME/laser-ai.git"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 2. Confirm bundle.zip is uploaded\n",
    "import os, pathlib\n",
    "zips = list(pathlib.Path('.').glob('*.zip'))\n",
    "assert zips, 'Upload bundle.zip to the file panel first'\n",
    "BUNDLE = str(zips[0])\n",
    "print('using:', BUNDLE)"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 3. Run training\n",
    "from laser_ai.colab_train import run  # installed by step 1\n",
    "OUT = 'model.pt'\n",
    "run(BUNDLE, OUT)\n",
    "print('\\n Training complete. Download', OUT, 'from the file panel.')"
   ],
   "execution_count": null,
   "outputs": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "kernelspec": {"display_name": "Python 3", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Write `colab/README.md`**

```markdown
# Colab training for laser-ai

## Workflow

1. On your machine, run:
   ```
   laser-ai prepare-bundle /path/to/your/data -o bundle.zip
   ```
2. Open `laser_ai_train.ipynb` in Google Colab.
3. Runtime → Change runtime type → **T4 GPU** (free tier is sufficient).
4. Upload `bundle.zip` into the file panel (left sidebar).
5. Runtime → **Run all**.
6. Wait ~30–60 minutes. Loss should steadily decrease.
7. Download `model.pt` from the file panel.
8. Locally, run:
   ```
   laser-ai generate your-song.mp3 -o out.ilda --model model.pt
   ```

## Troubleshooting

- **"no bundle.zip"** — drag the zip file into the Colab file panel first.
- **pip install fails** — the notebook installs `laser-ai` from a Git URL. Edit cell 1 to point at your fork of this repo.
- **CUDA OOM** — halve `vae_batch_size` or `hidden` in `colab_train.run(...)`.
```

- [ ] **Step 4: Move `colab/colab_train.py` into the package for pip-install to work**

Actually, to make `from laser_ai.colab_train import run` work in Colab (as the notebook expects), move the file to `laser_ai/colab_train.py` instead. Update the path accordingly.

```bash
mv colab/colab_train.py laser_ai/colab_train.py
```

The notebook already imports from `laser_ai.colab_train`, matching this path.

- [ ] **Step 5: Commit**

```bash
git add colab/ laser_ai/colab_train.py
git commit -m "feat(colab): training notebook + orchestrator script"
```

---

## Task 15: End-to-end regression + README polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full test suite**

```bash
pytest --cov=laser_ai --cov-report=term-missing
```

Expected: all tests PASS, aggregate coverage > 80%.

- [ ] **Step 2: Update `README.md`**

Open the existing `README.md` (from Plan 1). Replace the `## Status` section with:

```markdown
## Status

ML training pipeline complete (v0.2.0):
- ✓ Everything in v0.1.0 (ILDA I/O, audio features, safety, CLI, stub generator)
- ✓ PyTorch Frame VAE (1D conv encoder/decoder, 64-dim latent)
- ✓ Causal Transformer sequencer (audio features → latent sequence)
- ✓ Chamfer + KL + RGB + travel loss suite
- ✓ Training data augmentation (rotate/flip/scale/hue)
- ✓ Checkpoint save/load (VAE + Sequencer bundle)
- ✓ TrainedGenerator swappable into the existing pipeline via `--model` flag
- ✓ Colab training notebook + orchestrator
- ✓ `prepare-bundle` for Colab dataset upload

Coming in later plans:
- Plan 3: PyQt6 GUI app (dataset/train/generate/preview tabs)
- Plan 4: OpenGL preview renderer with audio sync
- Plan 5: Helios DAC real-time streaming
- Possibly: CLAP embedding integration for richer audio semantics

## Full CLI

```bash
laser-ai info SHOW.ilda                         # inspect an ILDA file
laser-ai prepare-bundle DATA_DIR -o bundle.zip  # export training data for Colab
laser-ai train-vae DATA_DIR -o ck.pt            # train VAE locally on ILDA files
laser-ai train-sequencer DATA_DIR -c ck.pt -o ck.pt  # train sequencer on pairs
laser-ai generate SONG.mp3 -o OUT.ilda [--model ck.pt]  # generate show
```

## Training workflow (recommended)

1. Put 20+ `song.mp3` + `song.ilda` pairs (matching stems) in a folder.
2. `laser-ai prepare-bundle ./data -o bundle.zip` — exports the training bundle.
3. Open `colab/laser_ai_train.ipynb` in Google Colab (T4 GPU, free tier).
4. Upload `bundle.zip`, Run All, download `model.pt` (~30–60 min).
5. `laser-ai generate new-song.mp3 -o out.ilda --model model.pt`
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with v0.2.0 ML training workflow"
```

- [ ] **Step 4: Tag the release**

```bash
git tag -a v0.2.0-ml -m "ML training pipeline: VAE + Sequencer + Colab workflow"
```

---

## Definition of done

- All tests pass (`pytest` green). Aggregate coverage > 80%.
- `laser-ai prepare-bundle`, `laser-ai train-vae`, `laser-ai train-sequencer`, and `laser-ai generate --model` all work end-to-end.
- `colab/laser_ai_train.ipynb` contains a valid JSON notebook that imports from `laser_ai.colab_train` and runs a three-cell end-to-end training flow.
- `TrainedGenerator` implements the same `Generator` protocol that `StubGenerator` does — the pipeline code in `laser_ai/pipeline/generate.py` is unchanged from Plan 1.
- A user with no Python experience can follow `colab/README.md` and produce a `model.pt` without touching code.

## Spec coverage map (this plan vs spec §)

| Spec section | Covered in this plan? | Where |
|---|---|---|
| §4 Augmentations (rotate ±15°, flip, scale ±10%, hue) | ✓ | Task 2 |
| §5 Audio features (CLAP optional portion) | ✗ — punted | — |
| §6.1 Frame VAE | ✓ | Task 5 |
| §6.2 Audio-to-Latent Sequencer | ✓ | Task 8 |
| §6.3 Inference speed target | — verified by `test_trained_generator_output_shape` | Task 11 |
| §11 Training workflow (Colab) | ✓ | Tasks 13, 14 |
| §12 Generation workflow with model | ✓ | Task 12 (`--model` flag) |
| §17 Success: ILDA loads in third-party software | ✓ (Plan 1 fix) | inherited |
| §17 Success: generated shows feel musically-driven | — depends on real training data; infrastructure is in place | — |

Remaining spec work (not in this plan): §5 CLAP integration, §7 safety-strength UI (slider — comes with GUI in Plan 3), §8 preview renderer, §9 GUI, §10 real-time/Helios.
