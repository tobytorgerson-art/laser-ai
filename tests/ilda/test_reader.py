"""Tests for the ILDA reader."""
from __future__ import annotations

from pathlib import Path

import pytest

from laser_ai.ilda.reader import read_ilda


FIXTURE = Path(__file__).parent.parent / "fixtures" / "tiny_show.ilda"


def test_read_tiny_fixture_returns_three_frames():
    show = read_ilda(FIXTURE)
    assert len(show.frames) == 3


def test_read_tiny_fixture_first_frame_is_red_square():
    show = read_ilda(FIXTURE)
    frame = show.frames[0]
    assert len(frame.points) == 5
    # All points are red
    for p in frame.points:
        assert p.r == 255
        assert p.g == 0
        assert p.b == 0
    # Last point flagged
    assert frame.points[-1].is_last_point
    # First 4 points not flagged as last
    for p in frame.points[:4]:
        assert not p.is_last_point


def test_read_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_ilda(Path("nonexistent.ilda"))


def test_read_rejects_non_ilda_bytes(tmp_path):
    p = tmp_path / "bad.ilda"
    p.write_bytes(b"not an ilda file at all")
    with pytest.raises(ValueError, match="ILDA magic"):
        read_ilda(p)
