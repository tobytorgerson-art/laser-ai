"""Rule-based stub generator — maps audio features to primitive selections.

Serves as a placeholder for the trained VAE+Sequencer models (arriving in plan 2).
Both implement the same `Generator` protocol so the pipeline code doesn't change.
"""
from __future__ import annotations

import numpy as np

from laser_ai.audio.features import FEATURE_DIM, N_CHROMA, N_MELS
from laser_ai.generator.primitives import circle, grid, lissajous, sine_wave


# Feature layout: [128 mel | 12 chroma | 3 spectral (cent, rolloff, flat) | rms | onset | beat_phase]
_IDX_CHROMA = N_MELS
_IDX_SPEC_CENTROID = N_MELS + N_CHROMA
_IDX_SPEC_ROLLOFF = _IDX_SPEC_CENTROID + 1
_IDX_SPEC_FLATNESS = _IDX_SPEC_CENTROID + 2
_IDX_RMS = N_MELS + N_CHROMA + 3
_IDX_ONSET = _IDX_RMS + 1
_IDX_BEAT_PHASE = _IDX_ONSET + 1


class StubGenerator:
    """Rule-based placeholder Generator.

    Primitive selection:
    - onset > 0.7         → grid burst
    - flatness > 0.5      → lissajous (noisy/percussive)
    - centroid > 0.5      → sine wave (bright)
    - else                → circle

    Modulation:
    - Scale = 0.3 + 0.6 * rms
    - Rotation (applied via lissajous delta or shape phase) = beat_phase * 2π
    - Hue: argmax chroma → 12 evenly spaced hues
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(self, features: np.ndarray, *, n_points: int = 512) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"features must be (T, {FEATURE_DIM}), got {features.shape}"
            )
        T = features.shape[0]
        out = np.zeros((T, n_points, 6), dtype=np.float32)

        for i in range(T):
            f = features[i]
            onset = float(f[_IDX_ONSET])
            flatness = float(f[_IDX_SPEC_FLATNESS])
            centroid = float(f[_IDX_SPEC_CENTROID])
            rms = float(f[_IDX_RMS])
            beat_phase = float(f[_IDX_BEAT_PHASE])
            chroma = f[_IDX_CHROMA:_IDX_CHROMA + N_CHROMA]

            hue_bin = int(np.argmax(chroma)) if chroma.sum() > 0 else 0
            color = _hue_to_rgb(hue_bin / 12.0)

            scale = 0.3 + 0.6 * np.clip(rms, 0.0, 1.0)
            rot_phase = beat_phase * 2 * np.pi

            if onset > 0.7:
                arr = grid(n=n_points, rows=3, cols=3, color=color)
                arr[:, :2] *= scale
            elif flatness > 0.5:
                arr = lissajous(
                    n=n_points, a=3.0, b=2.0, delta=rot_phase, color=color
                )
                arr[:, :2] *= scale
            elif centroid > 0.5:
                arr = sine_wave(
                    n=n_points,
                    amplitude=0.5 * scale,
                    frequency=2.0 + 2.0 * centroid,
                    phase=rot_phase,
                    color=color,
                )
            else:
                arr = circle(n=n_points, radius=scale, color=color)

            # Apply rotation as a rigid transform for variety
            arr[:, :2] = _rotate(arr[:, :2], rot_phase * 0.5)
            arr[:, :2] = np.clip(arr[:, :2], -1.0, 1.0)

            out[i] = arr

        return out


def _hue_to_rgb(h: float) -> tuple[float, float, float]:
    """HSV with S=V=1 to RGB."""
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    q = 1 - f
    t = f
    i = i % 6
    if i == 0: return (1.0, t, 0.0)
    if i == 1: return (q, 1.0, 0.0)
    if i == 2: return (0.0, 1.0, t)
    if i == 3: return (0.0, q, 1.0)
    if i == 4: return (t, 0.0, 1.0)
    return (1.0, 0.0, q)


def _rotate(xy: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    m = np.array([[c, -s], [s, c]], dtype=xy.dtype)
    return xy @ m.T
