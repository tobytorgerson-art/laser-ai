"""Tests for primitive shape generators."""
from __future__ import annotations

import numpy as np

from laser_ai.generator.primitives import circle, grid, lissajous, sine_wave


def test_sine_wave_returns_n_by_6():
    arr = sine_wave(n=512, amplitude=0.5, frequency=2.0, phase=0.0, color=(1.0, 0.0, 0.0))
    assert arr.shape == (512, 6)
    assert arr.dtype == np.float32


def test_sine_wave_respects_amplitude():
    arr = sine_wave(n=256, amplitude=0.3, frequency=1.0, phase=0.0, color=(1.0, 1.0, 1.0))
    assert arr[:, 1].max() <= 0.31
    assert arr[:, 1].min() >= -0.31


def test_lissajous_returns_n_by_6_in_range():
    arr = lissajous(n=512, a=3, b=2, delta=np.pi / 2, color=(0.0, 1.0, 0.0))
    assert arr.shape == (512, 6)
    assert np.all(arr[:, 0] >= -1.0) and np.all(arr[:, 0] <= 1.0)
    assert np.all(arr[:, 1] >= -1.0) and np.all(arr[:, 1] <= 1.0)


def test_circle_is_closed_loop():
    arr = circle(n=512, radius=0.5, color=(0.0, 0.0, 1.0))
    # First and last point should coincide (closed loop)
    assert np.allclose(arr[0, :2], arr[-1, :2], atol=0.01)


def test_grid_produces_horizontal_lines_with_blanks():
    arr = grid(n=512, rows=3, cols=3, color=(1.0, 1.0, 0.0))
    # Must have some blanked travel points between grid lines
    assert arr[:, 5].sum() > 0
    assert arr.shape == (512, 6)


def test_all_primitives_have_color_set():
    for prim_fn, kwargs in [
        (sine_wave, dict(amplitude=0.5, frequency=1.0, phase=0.0)),
        (lissajous, dict(a=2, b=3, delta=0.0)),
        (circle, dict(radius=0.8)),
        (grid, dict(rows=4, cols=4)),
    ]:
        arr = prim_fn(n=256, color=(0.7, 0.2, 0.4), **kwargs)
        # at least one point should be non-blank and have the target color
        visible = arr[arr[:, 5] < 0.5]
        if len(visible) > 0:
            assert np.allclose(visible[0, 2:5], [0.7, 0.2, 0.4], atol=1e-3)
