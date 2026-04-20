"""Write Show objects as ILDA Format 4 files."""
from __future__ import annotations

import struct
from pathlib import Path

from laser_ai.ilda.types import Show


_MAGIC = b"ILDA"
_FORMAT = 4


def write_ilda(show: Show, path: str | Path) -> None:
    """Write a Show as an ILDA Format 4 (3D RGB) file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(show.frames)
    buf = bytearray()
    company = (show.company or "laser-ai").encode("ascii", errors="replace")[:8].ljust(8, b" ")

    for idx, frame in enumerate(show.frames):
        name = (frame.name or f"f{idx:04d}").encode("ascii", errors="replace")[:8].ljust(8, b" ")
        records = len(frame.points)

        # 32-byte header
        buf += _MAGIC
        buf += struct.pack(">I", _FORMAT)
        buf += name
        buf += company
        buf += struct.pack(">HHHHH", records, idx, total, 0, 0)

        for i, p in enumerate(frame.points):
            status = 0
            if p.is_blank:
                status |= 0b01000000
            if i == records - 1 or p.is_last_point:
                status |= 0b10000000
            buf += struct.pack(
                ">hhhBBBB",
                int(p.x), int(p.y), 0,  # z always 0 for 2D shows
                status,
                int(p.b), int(p.g), int(p.r),  # ILDA spec: B, G, R order
            )

    # End-of-file section
    buf += _MAGIC
    buf += struct.pack(">I", _FORMAT)
    buf += b" " * 8
    buf += company
    buf += struct.pack(">HHHHH", 0, total, total, 0, 0)

    path.write_bytes(bytes(buf))
