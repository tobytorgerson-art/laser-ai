"""Tests for the ILDA writer."""
from __future__ import annotations

import struct
from pathlib import Path

from laser_ai.ilda.types import Frame, Point, Show
from laser_ai.ilda.writer import write_ilda


def test_write_produces_ilda_magic(tmp_path: Path):
    show = Show(frames=[Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)])])
    out = tmp_path / "test.ilda"
    write_ilda(show, out)
    data = out.read_bytes()
    assert data[:4] == b"ILDA"


def test_write_records_format_4(tmp_path: Path):
    show = Show(frames=[Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)])])
    out = tmp_path / "test.ilda"
    write_ilda(show, out)
    data = out.read_bytes()
    (fmt,) = struct.unpack(">I", data[4:8])
    assert fmt == 4


def test_write_ends_with_zero_record_section(tmp_path: Path):
    show = Show(frames=[Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)])])
    out = tmp_path / "test.ilda"
    write_ilda(show, out)
    data = out.read_bytes()
    # Last header should be an ILDA section with records=0. The writer
    # emits magic(4) + fmt(4) + name(8) + company(8) + 5*u16(10) = 34 bytes,
    # matching the fixture format, so look at the trailing 34 bytes.
    tail = data[-34:]
    assert tail[:4] == b"ILDA"
    records = struct.unpack(">H", tail[24:26])[0]
    assert records == 0
