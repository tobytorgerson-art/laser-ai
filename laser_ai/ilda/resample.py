"""Even-arc-length resampling of ILDA frames to a fixed point count."""
from __future__ import annotations

import numpy as np

from laser_ai.ilda.types import Frame, Point


def resample_frame(frame: Frame, n: int = 512) -> Frame:
    """Resample `frame` to exactly `n` points via uniform arc-length spacing.

    - Empty frames become `n` blanked points at origin.
    - Single-point frames become `n` copies of that point.
    - Blanking/color of each resampled point is taken from the nearest original point.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    src = frame.points
    if len(src) == 0:
        return Frame(points=[
            Point(x=0, y=0, r=0, g=0, b=0, is_blank=True, is_last_point=(i == n - 1))
            for i in range(n)
        ], name=frame.name, company=frame.company,
           frame_index=frame.frame_index, total_frames=frame.total_frames)

    if len(src) == 1:
        p = src[0]
        return Frame(points=[
            Point(x=p.x, y=p.y, r=p.r, g=p.g, b=p.b, is_blank=p.is_blank,
                  is_last_point=(i == n - 1))
            for i in range(n)
        ], name=frame.name, company=frame.company,
           frame_index=frame.frame_index, total_frames=frame.total_frames)

    xs = np.array([p.x for p in src], dtype=np.float64)
    ys = np.array([p.y for p in src], dtype=np.float64)
    seg_lens = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = cum[-1]

    if total_len <= 0.0:
        # All points at same location
        p = src[0]
        return Frame(points=[
            Point(x=p.x, y=p.y, r=p.r, g=p.g, b=p.b, is_blank=p.is_blank,
                  is_last_point=(i == n - 1))
            for i in range(n)
        ], name=frame.name, company=frame.company,
           frame_index=frame.frame_index, total_frames=frame.total_frames)

    targets = np.linspace(0.0, total_len, n)
    indices = np.searchsorted(cum, targets, side="right") - 1
    indices = np.clip(indices, 0, len(src) - 2)

    resampled_pts: list[Point] = []
    for i, t in enumerate(targets):
        idx = int(indices[i])
        seg_start = cum[idx]
        seg_len = cum[idx + 1] - cum[idx]
        frac = 0.0 if seg_len <= 0.0 else (t - seg_start) / seg_len
        frac = float(np.clip(frac, 0.0, 1.0))

        a = src[idx]
        b = src[idx + 1]
        x = a.x + frac * (b.x - a.x)
        y = a.y + frac * (b.y - a.y)

        # Take attributes from whichever endpoint we're closer to
        src_pt = a if frac < 0.5 else b
        resampled_pts.append(Point(
            x=int(round(x)),
            y=int(round(y)),
            r=src_pt.r, g=src_pt.g, b=src_pt.b,
            is_blank=src_pt.is_blank,
            is_last_point=(i == n - 1),
        ))

    return Frame(points=resampled_pts, name=frame.name, company=frame.company,
                 frame_index=frame.frame_index, total_frames=frame.total_frames)
