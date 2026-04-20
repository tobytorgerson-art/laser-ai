"""Convert ILDA Show objects to fixed-shape float32 tensors."""
from __future__ import annotations

import numpy as np

from laser_ai.ilda.resample import resample_frame
from laser_ai.ilda.types import Show


def show_to_tensor(show: Show, n_points: int = 512) -> np.ndarray:
    """Return a `(num_frames, n_points, 6)` float32 tensor for the show.

    Each frame is arc-length resampled to exactly `n_points` points, then
    converted to the normalized (x, y, r, g, b, is_travel) float32 format.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    n_frames = len(show.frames)
    if n_frames == 0:
        return np.zeros((0, n_points, 6), dtype=np.float32)

    out = np.empty((n_frames, n_points, 6), dtype=np.float32)
    for i, frame in enumerate(show.frames):
        resampled = resample_frame(frame, n=n_points)
        out[i] = resampled.to_array()
    return out
