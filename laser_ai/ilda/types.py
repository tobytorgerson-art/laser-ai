"""Core ILDA data types."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Point:
    """A single laser point.

    x, y are signed int16 in the ILDA coordinate space [-32768, 32767].
    r, g, b are uint8 [0, 255].
    is_blank means the laser is off during this point (travel move).
    is_last_point marks the final point of a frame.
    """
    x: int
    y: int
    r: int = 0
    g: int = 0
    b: int = 0
    is_blank: bool = False
    is_last_point: bool = False


@dataclass(slots=True)
class Frame:
    """A single ILDA frame: an ordered list of Points."""
    points: list[Point] = field(default_factory=list)
    name: str = ""
    company: str = ""
    frame_index: int = 0
    total_frames: int = 1

    def to_array(self) -> np.ndarray:
        """Return an (N, 6) float32 array: (x, y, r, g, b, is_blank).

        Coordinates are normalized to [-1, 1]; colors to [0, 1]; is_blank to {0, 1}.
        """
        if not self.points:
            return np.zeros((0, 6), dtype=np.float32)
        arr = np.zeros((len(self.points), 6), dtype=np.float32)
        for i, p in enumerate(self.points):
            arr[i, 0] = p.x / 32768.0
            arr[i, 1] = p.y / 32768.0
            arr[i, 2] = p.r / 255.0
            arr[i, 3] = p.g / 255.0
            arr[i, 4] = p.b / 255.0
            arr[i, 5] = 1.0 if p.is_blank else 0.0
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray, name: str = "") -> "Frame":
        """Build a Frame from an (N, 6) float32 array in the format of to_array()."""
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"expected (N, 6) array, got {arr.shape}")
        pts: list[Point] = []
        n = arr.shape[0]
        for i in range(n):
            x = int(np.clip(arr[i, 0], -1.0, 1.0) * 32767)
            y = int(np.clip(arr[i, 1], -1.0, 1.0) * 32767)
            r = int(np.clip(arr[i, 2], 0.0, 1.0) * 255)
            g = int(np.clip(arr[i, 3], 0.0, 1.0) * 255)
            b = int(np.clip(arr[i, 4], 0.0, 1.0) * 255)
            is_blank = bool(arr[i, 5] >= 0.5)
            is_last = i == n - 1
            pts.append(Point(x=x, y=y, r=r, g=g, b=b, is_blank=is_blank, is_last_point=is_last))
        return cls(points=pts, name=name)


@dataclass(slots=True)
class Show:
    """A complete laser show: ordered list of Frames + metadata."""
    frames: list[Frame] = field(default_factory=list)
    fps: float = 30.0
    name: str = ""
    company: str = "laser-ai"

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0.0
