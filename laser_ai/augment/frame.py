"""Per-frame augmentations operating on (N, 6) float32 arrays.

Columns: (x, y, r, g, b, is_travel). Augmentations preserve is_travel unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AugmentConfig:
    """Control which augmentations fire and with what magnitudes."""
    enable_rotate: bool = True
    enable_flip_h: bool = True
    enable_flip_v: bool = True
    enable_scale: bool = True
    enable_hue: bool = True
    rotate_max_deg: float = 15.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    flip_h_prob: float = 0.5
    flip_v_prob: float = 0.5


def _check(arr: np.ndarray) -> None:
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"expected (N, 6) array, got {arr.shape}")


def rotate(arr: np.ndarray, theta: float) -> np.ndarray:
    """Rotate xy by `theta` radians. Non-xy columns untouched."""
    _check(arr)
    c, s = np.cos(theta), np.sin(theta)
    out = arr.copy()
    out[:, 0] = c * arr[:, 0] - s * arr[:, 1]
    out[:, 1] = s * arr[:, 0] + c * arr[:, 1]
    return out


def flip_horizontal(arr: np.ndarray) -> np.ndarray:
    _check(arr)
    out = arr.copy()
    out[:, 0] = -arr[:, 0]
    return out


def flip_vertical(arr: np.ndarray) -> np.ndarray:
    _check(arr)
    out = arr.copy()
    out[:, 1] = -arr[:, 1]
    return out


def scale(arr: np.ndarray, factor: float) -> np.ndarray:
    _check(arr)
    out = arr.copy()
    out[:, :2] = arr[:, :2] * factor
    return out


def rotate_hue(arr: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate the RGB hue by `angle_rad` (0 .. 2π). Preserves saturation/value."""
    _check(arr)
    rgb = arr[:, 2:5]
    # RGB → HSV (via simple formula; keeps the per-point intensity)
    cmax = rgb.max(axis=1)
    cmin = rgb.min(axis=1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 0
    idx_r = (rgb[:, 0] == cmax) & mask
    idx_g = (rgb[:, 1] == cmax) & mask & ~idx_r
    idx_b = (rgb[:, 2] == cmax) & mask & ~idx_r & ~idx_g
    h[idx_r] = ((rgb[idx_r, 1] - rgb[idx_r, 2]) / delta[idx_r]) % 6
    h[idx_g] = (rgb[idx_g, 2] - rgb[idx_g, 0]) / delta[idx_g] + 2
    h[idx_b] = (rgb[idx_b, 0] - rgb[idx_b, 1]) / delta[idx_b] + 4
    h = (h * 60.0 + np.degrees(angle_rad)) % 360.0
    s = np.where(cmax > 0, delta / np.maximum(cmax, 1e-12), 0.0)
    v = cmax
    # HSV → RGB
    hi = (h / 60.0).astype(np.int64) % 6
    f = (h / 60.0) - np.floor(h / 60.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    new_rgb = np.zeros_like(rgb)
    conds = [
        (hi == 0, v, t, p),
        (hi == 1, q, v, p),
        (hi == 2, p, v, t),
        (hi == 3, p, q, v),
        (hi == 4, t, p, v),
        (hi == 5, v, p, q),
    ]
    for cond, rv, gv, bv in conds:
        new_rgb[cond, 0] = rv[cond]
        new_rgb[cond, 1] = gv[cond]
        new_rgb[cond, 2] = bv[cond]
    out = arr.copy()
    out[:, 2:5] = np.clip(new_rgb, 0.0, 1.0).astype(arr.dtype)
    return out


def augment_frame(arr: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply a random subset of augmentations per cfg using the provided RNG."""
    _check(arr)
    out = arr.copy()

    if cfg.enable_rotate:
        theta = np.radians(rng.uniform(-cfg.rotate_max_deg, cfg.rotate_max_deg))
        out = rotate(out, theta)

    if cfg.enable_flip_h and rng.random() < cfg.flip_h_prob:
        out = flip_horizontal(out)

    if cfg.enable_flip_v and rng.random() < cfg.flip_v_prob:
        out = flip_vertical(out)

    if cfg.enable_scale:
        lo, hi = cfg.scale_range
        out = scale(out, float(rng.uniform(lo, hi)))

    if cfg.enable_hue:
        out = rotate_hue(out, float(rng.uniform(0.0, 2 * np.pi)))

    # Clamp to valid ranges
    out[:, :2] = np.clip(out[:, :2], -1.0, 1.0)
    out[:, 2:5] = np.clip(out[:, 2:5], 0.0, 1.0)
    return out
