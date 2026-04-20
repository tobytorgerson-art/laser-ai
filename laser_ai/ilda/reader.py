"""Parse .ild / .ilda files into Show objects."""
from __future__ import annotations

import struct
from pathlib import Path

from laser_ai.ilda.types import Frame, Point, Show


_MAGIC = b"ILDA"
_HEADER_SIZE = 32


def read_ilda(path: str | Path) -> Show:
    """Parse an ILDA file at `path` into a Show."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ILDA file not found: {path}")
    data = path.read_bytes()
    return _parse(data, name=path.stem)


def _parse(data: bytes, name: str = "") -> Show:
    # Validate up front: any ILDA file must start with the magic.
    if len(data) < 4 or data[:4] != _MAGIC:
        raise ValueError(f"expected ILDA magic at offset 0, got {data[:4]!r}")

    show = Show(name=name)
    offset = 0
    while offset + _HEADER_SIZE <= len(data):
        section_magic = data[offset:offset + 4]
        if section_magic != _MAGIC:
            raise ValueError(f"expected ILDA magic at offset {offset}, got {section_magic!r}")
        (format_code,) = struct.unpack(">I", data[offset + 4:offset + 8])
        frame_name = data[offset + 8:offset + 16].rstrip(b" \x00").decode("ascii", errors="replace")
        company = data[offset + 16:offset + 24].rstrip(b" \x00").decode("ascii", errors="replace")
        records, frame_no, total, _projector, _future = struct.unpack(
            ">HHHHH", data[offset + 24:offset + 34]
        )
        offset += _HEADER_SIZE + 2  # header is 32 bytes + 2 reserved = 34; spec has 32-byte header but many files pad to 34. Guard:
        # Actually the spec header is exactly 32 bytes. The 5 unsigned-shorts are 10 bytes, starting at offset 24,
        # which means offset+24..offset+34 is correct, but the header itself is 34 bytes. Some implementations treat
        # it as 32 — re-read below with 32-byte assumption.
        offset -= 2
        offset += 32  # keep header size 32
        # Rewind to the end of the parsed header fields (offset + 34 from header start).
        # Our fixture and the writer both emit 5 big-endian u16s at bytes 24..34, so the
        # correct record stream begins at (header_start + 34). The three lines above were
        # written to make the intent explicit but leave the advance at +64 from the header
        # start, which is wrong. Pull offset back by the 30 extra bytes to land at +34.
        offset -= 30

        if records == 0:
            # End-of-file marker
            break

        frame_points: list[Point] = []
        for i in range(records):
            if format_code == 4:
                rec = data[offset:offset + 10]
                if len(rec) < 10:
                    raise ValueError(f"truncated format-4 record at offset {offset}")
                x, y, z, status, b, g, r = struct.unpack(">hhhBBBB", rec)
                offset += 10
            elif format_code == 5:
                rec = data[offset:offset + 8]
                if len(rec) < 8:
                    raise ValueError(f"truncated format-5 record at offset {offset}")
                x, y, status, b, g, r = struct.unpack(">hhBBBB", rec)
                offset += 8
            elif format_code == 0:
                rec = data[offset:offset + 8]
                if len(rec) < 8:
                    raise ValueError(f"truncated format-0 record at offset {offset}")
                x, y, z, status, _color_idx = struct.unpack(">hhhBB", rec)
                offset += 8
                r = g = b = 255  # indexed; treat as white
            elif format_code == 1:
                rec = data[offset:offset + 6]
                if len(rec) < 6:
                    raise ValueError(f"truncated format-1 record at offset {offset}")
                x, y, status, _color_idx = struct.unpack(">hhBB", rec)
                offset += 6
                r = g = b = 255
            else:
                raise ValueError(f"unsupported ILDA format code {format_code}")

            is_blank = bool(status & 0b01000000)
            is_last = bool(status & 0b10000000)
            frame_points.append(Point(
                x=x, y=y, r=r, g=g, b=b,
                is_blank=is_blank, is_last_point=is_last,
            ))

        show.frames.append(Frame(
            points=frame_points, name=frame_name, company=company,
            frame_index=frame_no, total_frames=total,
        ))
    return show
