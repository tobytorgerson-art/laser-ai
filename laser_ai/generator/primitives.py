"""Parametric primitive shapes. Each returns an (N, 6) float32 array.

Format: (x, y, r, g, b, is_blank) with x,y in [-1, 1] and r,g,b in [0, 1].
"""
from __future__ import annotations

import numpy as np


def sine_wave(
    *,
    n: int,
    amplitude: float,
    frequency: float,
    phase: float,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Horizontal sine across x=[-1, 1]."""
    t = np.linspace(-1.0, 1.0, n)
    x = t
    y = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return _assemble(x, y, color, blank=np.zeros(n, dtype=np.float32))


def lissajous(
    *,
    n: int,
    a: float,
    b: float,
    delta: float,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Classic lissajous: x=sin(a*t+delta), y=sin(b*t), t in [0, 2π]."""
    t = np.linspace(0.0, 2 * np.pi, n)
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    return _assemble(x, y, color, blank=np.zeros(n, dtype=np.float32))


def circle(
    *,
    n: int,
    radius: float,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Closed circle at origin."""
    t = np.linspace(0.0, 2 * np.pi, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    # Close the loop exactly
    x[-1] = x[0]
    y[-1] = y[0]
    return _assemble(x, y, color, blank=np.zeros(n, dtype=np.float32))


def grid(
    *,
    n: int,
    rows: int,
    cols: int,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Horizontal + vertical grid lines with blanked transits between strokes."""
    # Build a polyline of rows horizontal lines then cols vertical lines,
    # with blanked moves between them.
    segs_x: list[np.ndarray] = []
    segs_y: list[np.ndarray] = []
    is_blank_chunks: list[np.ndarray] = []

    def add(xs: np.ndarray, ys: np.ndarray, blanks: np.ndarray) -> None:
        segs_x.append(xs)
        segs_y.append(ys)
        is_blank_chunks.append(blanks)

    row_ys = np.linspace(-0.9, 0.9, rows)
    col_xs = np.linspace(-0.9, 0.9, cols)

    # Budget: split n evenly across strokes. Each stroke = k visible pts + 2 blank transit pts.
    total_strokes = rows + cols
    per_stroke = max(8, n // (total_strokes + 1))  # +1 for end padding

    for yv in row_ys:
        xs = np.linspace(-0.9, 0.9, per_stroke)
        ys = np.full(per_stroke, yv, dtype=np.float32)
        add(xs, ys, np.zeros(per_stroke, dtype=np.float32))
        # blank transit (2 pts)
        add(xs[-1:].copy(), ys[-1:].copy(), np.array([1.0], dtype=np.float32))

    for xv in col_xs:
        ys = np.linspace(-0.9, 0.9, per_stroke)
        xs = np.full(per_stroke, xv, dtype=np.float32)
        add(xs, ys, np.zeros(per_stroke, dtype=np.float32))
        add(xs[-1:].copy(), ys[-1:].copy(), np.array([1.0], dtype=np.float32))

    x = np.concatenate(segs_x)
    y = np.concatenate(segs_y)
    blank = np.concatenate(is_blank_chunks)

    # Pad or truncate to exactly n
    if len(x) > n:
        x = x[:n]; y = y[:n]; blank = blank[:n]
    elif len(x) < n:
        pad = n - len(x)
        x = np.concatenate([x, np.full(pad, x[-1])])
        y = np.concatenate([y, np.full(pad, y[-1])])
        blank = np.concatenate([blank, np.ones(pad, dtype=np.float32)])

    return _assemble(x, y, color, blank=blank)


def _assemble(
    x: np.ndarray,
    y: np.ndarray,
    color: tuple[float, float, float],
    blank: np.ndarray,
) -> np.ndarray:
    n = len(x)
    arr = np.zeros((n, 6), dtype=np.float32)
    arr[:, 0] = np.clip(x, -1.0, 1.0)
    arr[:, 1] = np.clip(y, -1.0, 1.0)
    arr[:, 2] = color[0]
    arr[:, 3] = color[1]
    arr[:, 4] = color[2]
    arr[:, 5] = blank
    # Blanked points rendered as zero-brightness too, but color kept for future style info
    arr[blank >= 0.5, 2:5] = 0.0
    return arr
