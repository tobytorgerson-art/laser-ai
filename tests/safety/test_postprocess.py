"""Tests for the safety post-processor."""
from __future__ import annotations

import numpy as np

from laser_ai.safety.postprocess import SafetyConfig, apply_safety


def test_output_never_exceeds_max_points():
    rng = np.random.default_rng(0)
    arr = rng.random((2000, 6), dtype=np.float32) * 2 - 1  # noisy random
    arr[:, 5] = 0
    out = apply_safety(arr, SafetyConfig(max_points=1200))
    assert len(out) <= 1200


def test_output_coords_are_clamped():
    arr = np.zeros((100, 6), dtype=np.float32)
    arr[:, 0] = 5.0  # out of range
    arr[:, 1] = -5.0
    out = apply_safety(arr, SafetyConfig())
    assert np.all(out[:, 0] <= 1.0) and np.all(out[:, 0] >= -1.0)
    assert np.all(out[:, 1] <= 1.0) and np.all(out[:, 1] >= -1.0)


def test_output_colors_are_clipped():
    arr = np.zeros((100, 6), dtype=np.float32)
    arr[:, 2] = 2.0  # out of range
    arr[:, 3] = -0.5
    out = apply_safety(arr, SafetyConfig())
    assert np.all(out[:, 2:5] <= 1.0) and np.all(out[:, 2:5] >= 0.0)


def test_large_jumps_are_blanked_or_interpolated():
    # Two far-apart visible points
    arr = np.array([
        [-1.0, -1.0, 1.0, 0, 0, 0.0],
        [ 1.0,  1.0, 1.0, 0, 0, 0.0],
    ], dtype=np.float32)
    out = apply_safety(arr, SafetyConfig(max_step=0.1))
    # Output should have more points now (interpolation) and some blanked
    assert len(out) > 2
    assert out[:, 5].sum() > 0


def test_dwell_points_injected_at_endpoint():
    arr = np.array([
        [-1.0, 0.0, 1.0, 0, 0, 0.0],
        [ 0.0, 0.0, 1.0, 0, 0, 0.0],
        [ 1.0, 0.0, 1.0, 0, 0, 0.0],
    ], dtype=np.float32)
    out = apply_safety(arr, SafetyConfig(dwell_count=4, max_step=10.0))
    # Final repeated point should appear >= dwell_count times at the end
    last_xy = out[-1, :2]
    tail_same = 0
    for i in range(len(out) - 1, -1, -1):
        if np.allclose(out[i, :2], last_xy, atol=1e-4):
            tail_same += 1
        else:
            break
    assert tail_same >= 4


def test_safety_is_idempotent_ish():
    arr = np.zeros((100, 6), dtype=np.float32)
    arr[:, 0] = np.linspace(-0.5, 0.5, 100)
    arr[:, 2] = 1.0  # red
    once = apply_safety(arr, SafetyConfig())
    twice = apply_safety(once, SafetyConfig())
    # After two passes with no out-of-range input, length should be stable
    assert abs(len(once) - len(twice)) < 10


def test_empty_input_produces_empty_safe_frame():
    out = apply_safety(np.zeros((0, 6), dtype=np.float32), SafetyConfig())
    # Empty frame passes through (pipeline handles frame-level empty cases)
    assert out.shape[1] == 6
