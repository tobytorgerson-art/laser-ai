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
