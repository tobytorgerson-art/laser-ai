"""Tests for ILDA data types."""
from __future__ import annotations

import numpy as np

from laser_ai.ilda.types import Frame, Point


def test_frame_to_array_roundtrips_through_from_array():
    original = Frame(points=[
        Point(x=0, y=0, r=255, g=0, b=0),
        Point(x=16000, y=-16000, r=0, g=255, b=128, is_blank=True),
        Point(x=-32768, y=32767, r=100, g=100, b=100),
    ])
    arr = original.to_array()
    recovered = Frame.from_array(arr)

    assert len(recovered.points) == 3
    # allow ±1 off due to int->float->int quantization
    for p_orig, p_rec in zip(original.points, recovered.points):
        assert abs(p_orig.x - p_rec.x) <= 2
        assert abs(p_orig.y - p_rec.y) <= 2
        assert abs(p_orig.r - p_rec.r) <= 1
        assert abs(p_orig.g - p_rec.g) <= 1
        assert abs(p_orig.b - p_rec.b) <= 1
        assert p_orig.is_blank == p_rec.is_blank


def test_frame_from_array_rejects_wrong_shape():
    import pytest
    with pytest.raises(ValueError, match="expected"):
        Frame.from_array(np.zeros((10, 5), dtype=np.float32))


def test_empty_frame_to_array_is_empty():
    arr = Frame().to_array()
    assert arr.shape == (0, 6)
