"""Frame augmentations for training data."""

from laser_ai.augment.frame import (
    AugmentConfig,
    augment_frame,
    flip_horizontal,
    flip_vertical,
    rotate,
    rotate_hue,
    scale,
)

__all__ = [
    "AugmentConfig",
    "augment_frame",
    "flip_horizontal",
    "flip_vertical",
    "rotate",
    "rotate_hue",
    "scale",
]
