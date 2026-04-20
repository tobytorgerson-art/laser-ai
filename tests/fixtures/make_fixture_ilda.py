"""Generate a deterministic tiny ILDA fixture for tests.

Run: python tests/fixtures/make_fixture_ilda.py
"""
from __future__ import annotations

import struct
from pathlib import Path


def write_tiny_ilda(path: Path) -> None:
    """Three frames, each with a small square pattern, ILDA format 4."""
    frames = []
    total_frames = 3

    def square_points(size: int, color: tuple[int, int, int]) -> list[tuple]:
        r, g, b = color
        # 4 corners + close, 5 points, last one flagged last-point
        pts = [
            (-size, -size, 0, False, r, g, b),
            ( size, -size, 0, False, r, g, b),
            ( size,  size, 0, False, r, g, b),
            (-size,  size, 0, False, r, g, b),
            (-size, -size, 0, True,  r, g, b),  # last
        ]
        return pts

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    sizes = [8000, 16000, 24000]

    for idx in range(total_frames):
        pts = square_points(sizes[idx], colors[idx])
        # ILDA format 4 header (32 bytes)
        header = b"ILDA" + struct.pack(">I", 4)  # magic + format
        header += b"FRAME   ".ljust(8, b" ")[:8]    # name (8 bytes)
        header += b"LASERAI ".ljust(8, b" ")[:8]   # company (8 bytes)
        header += struct.pack(">HHHBB",
                              len(pts),            # records
                              idx,                 # frame no
                              total_frames,        # total
                              0,                   # projector
                              0)                   # future
        frames.append(header)
        # records: 10 bytes per point for format 4 (x, y, z int16 + status + B, G, R)
        for (x, y, z, is_last, r, g, b) in pts:
            status = 0
            if is_last:
                status |= 0b10000000
            frames.append(struct.pack(">hhhBBBB", x, y, z, status, b, g, r))

    # End-of-file section (0 records)
    eof_header = b"ILDA" + struct.pack(">I", 4)
    eof_header += b"        " + b"        "
    eof_header += struct.pack(">HHHBB", 0, total_frames, total_frames, 0, 0)
    frames.append(eof_header)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for chunk in frames:
            f.write(chunk)


if __name__ == "__main__":
    here = Path(__file__).parent
    write_tiny_ilda(here / "tiny_show.ilda")
    print(f"wrote {here / 'tiny_show.ilda'}")
