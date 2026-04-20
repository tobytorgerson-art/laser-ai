"""Generator protocol — contract for anything that maps audio features to laser frames."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Generator(Protocol):
    """A Generator turns a (T, FEATURE_DIM) feature stream into a (T, N, 6) frame stream."""

    def generate(self, features: np.ndarray, *, n_points: int = 512) -> np.ndarray:
        """Return (T, n_points, 6) frames for T feature rows."""
        ...
