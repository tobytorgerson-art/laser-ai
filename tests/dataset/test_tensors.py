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
