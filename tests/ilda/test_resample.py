"""Tests for arc-length frame resampling."""
from __future__ import annotations

import numpy as np

from laser_ai.ilda.resample import resample_frame
from laser_ai.ilda.types import Frame, Point


def _line_frame(n: int = 10) -> Frame:
    """A simple horizontal line from (-10000, 0) to (10000, 0)."""
    return Frame(points=[
        Point(x=-10000 + int(20000 * i / (n - 1)), y=0, r=255, g=255, b=255,
              is_last_point=(i == n - 1))
        for i in range(n)
    ])


def test_resample_returns_exact_n_points():
    frame = _line_frame(10)
    out = resample_frame(frame, n=512)
    assert len(out.points) == 512


def test_resample_preserves_start_and_end_position():
    frame = _line_frame(10)
    out = resample_frame(frame, n=512)
    # Start close to (-10000, 0)
    assert abs(out.points[0].x - (-10000)) < 100
    assert abs(out.points[0].y) < 100
    # End close to (10000, 0)
    assert abs(out.points[-1].x - 10000) < 100
    assert abs(out.points[-1].y) < 100


def test_resample_preserves_color():
    frame = _line_frame(10)
    out = resample_frame(frame, n=512)
    for p in out.points:
        assert (p.r, p.g, p.b) == (255, 255, 255)


def test_resample_flags_last_point():
    frame = _line_frame(10)
    out = resample_frame(frame, n=512)
    assert out.points[-1].is_last_point
    assert not any(p.is_last_point for p in out.points[:-1])


def test_resample_handles_single_point_frame():
    frame = Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)])
    out = resample_frame(frame, n=64)
    assert len(out.points) == 64
    for p in out.points:
        assert p.x == 0 and p.y == 0


def test_resample_handles_empty_frame():
    out = resample_frame(Frame(points=[]), n=64)
    assert len(out.points) == 64
    for p in out.points:
        assert p.is_blank


def test_resample_arc_length_spacing_is_uniform():
    """Points along a straight line should be roughly evenly spaced after resampling."""
    frame = _line_frame(10)
    out = resample_frame(frame, n=100)
    xs = np.array([p.x for p in out.points], dtype=np.float64)
    diffs = np.diff(xs)
    # All spacings should be within 5% of the mean
    assert np.std(diffs) / np.mean(diffs) < 0.05
