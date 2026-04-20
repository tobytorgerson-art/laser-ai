"""Deterministic DAC-safety post-processor.

Turns whatever the generator emits into a frame that is safe to send to a laser DAC:
- Velocity-limited
- Dwell-padded on endpoints
- Coord-clamped
- Color-clipped
- Point-rate capped
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SafetyConfig:
    """Runtime-tunable safety parameters."""
    max_points: int = 1200           # ~36k pps at 30 fps
    max_step: float = 0.08            # max normalized distance between consecutive points
    dwell_count: int = 4              # repeated points at endpoints
    coord_margin: float = 0.02        # shrink effective XY range by this much
    strength: float = 1.0             # 0.0 = loose, 1.0 = medium, 2.0 = tight


def apply_safety(arr: np.ndarray, cfg: SafetyConfig = SafetyConfig()) -> np.ndarray:
    """Apply all safety rules to a single (N, 6) frame; return (M, 6)."""
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"expected (N, 6) array, got {arr.shape}")
    if arr.shape[0] == 0:
        return arr.copy()

    a = arr.copy()
    # 1. Clip colors
    a[:, 2:5] = np.clip(a[:, 2:5], 0.0, 1.0)
    # 2. Clamp coords
    limit = 1.0 - cfg.coord_margin
    a[:, :2] = np.clip(a[:, :2], -limit, limit)
    # Discretize is_blank to {0, 1}
    a[:, 5] = (a[:, 5] >= 0.5).astype(a.dtype)

    # 3. Velocity limit / blanking insertion (step-scaled by strength)
    effective_max_step = cfg.max_step / max(0.1, cfg.strength)
    a = _velocity_limit(a, max_step=effective_max_step)

    # 4. Endpoint dwell
    a = _add_endpoint_dwell(a, count=cfg.dwell_count)

    # 5. Point-rate cap (downsample)
    if len(a) > cfg.max_points:
        a = _downsample_arc_length(a, target=cfg.max_points)

    return a


def _velocity_limit(a: np.ndarray, *, max_step: float) -> np.ndarray:
    """For consecutive points farther apart than max_step, insert intermediates.

    If the jump is between two visible points, inserted points remain visible
    (interpolation). If either endpoint is blank, inserted points are blank transit.
    """
    if len(a) < 2:
        return a

    out: list[np.ndarray] = [a[0:1]]
    for i in range(1, len(a)):
        prev = a[i - 1]
        cur = a[i]
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        dist = float(np.hypot(dx, dy))
        if dist > max_step:
            steps = int(np.ceil(dist / max_step))
            # interpolate (excluding prev, including cur)
            lerp = np.linspace(0.0, 1.0, steps + 1)[1:, None]
            xy = (1 - lerp) * prev[:2] + lerp * cur[:2]
            # blank transit if either endpoint was blank OR if visual line would cross too far
            is_transit = (prev[5] >= 0.5) or (cur[5] >= 0.5) or (dist > 1.0)
            rgb = (prev[2:5] + cur[2:5]) / 2.0  # rough blend
            blank_flag = 1.0 if is_transit else 0.0
            color_fill = np.zeros(3, dtype=a.dtype) if is_transit else rgb
            block = np.zeros((steps, 6), dtype=a.dtype)
            block[:, :2] = xy
            block[:, 2:5] = color_fill
            block[:, 5] = blank_flag
            # Ensure the final inserted point equals cur exactly
            block[-1] = cur
            out.append(block)
        else:
            out.append(cur[None, :])
    return np.concatenate(out, axis=0)


def _add_endpoint_dwell(a: np.ndarray, *, count: int) -> np.ndarray:
    """Append `count` repeated copies of the last point (and prepend for the first)."""
    if len(a) == 0 or count <= 0:
        return a
    head = np.tile(a[0:1], (count - 1, 1))
    tail = np.tile(a[-1:], (count - 1, 1))
    return np.concatenate([head, a, tail], axis=0)


def _downsample_arc_length(a: np.ndarray, *, target: int) -> np.ndarray:
    """Resample to exactly `target` points preserving path shape."""
    if len(a) <= target:
        return a
    xs = a[:, 0]; ys = a[:, 1]
    seg = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = cum[-1]
    if total <= 0:
        return a[:target]
    t = np.linspace(0.0, total, target)
    idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(a) - 2)
    out = np.zeros((target, 6), dtype=a.dtype)
    for i, ti in enumerate(t):
        j = int(idx[i])
        seg_len = cum[j + 1] - cum[j]
        frac = 0.0 if seg_len <= 0 else (ti - cum[j]) / seg_len
        out[i, :2] = (1 - frac) * a[j, :2] + frac * a[j + 1, :2]
        src = a[j] if frac < 0.5 else a[j + 1]
        out[i, 2:6] = src[2:6]
    return out
