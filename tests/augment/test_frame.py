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
