"""End-to-end generation: audio file → Show."""
from __future__ import annotations

from pathlib import Path

from laser_ai.audio.features import extract_features
from laser_ai.audio.loader import load_audio
from laser_ai.generator.base import Generator
from laser_ai.generator.stub import StubGenerator
from laser_ai.ilda.types import Frame, Show
from laser_ai.safety.postprocess import SafetyConfig, apply_safety


def generate_show_from_audio(
    *,
    audio_path: str | Path,
    generator: Generator | None = None,
    fps: float = 30.0,
    n_points: int = 512,
    safety_cfg: SafetyConfig | None = None,
    show_name: str = "",
) -> Show:
    """Generate an ILDA Show from an audio file.

    1. Load + resample audio to 44.1k mono float32
    2. Extract per-frame features at `fps`
    3. Run generator on features → (T, N, 6) frames
    4. Apply safety post-processor per frame
    5. Pack into Show via Frame.from_array
    """
    gen = generator if generator is not None else StubGenerator()
    cfg = safety_cfg if safety_cfg is not None else SafetyConfig()

    samples, sr = load_audio(audio_path)
    feats = extract_features(samples, sr, fps=fps)
    raw = gen.generate(feats, n_points=n_points)  # (T, N, 6)

    frames: list[Frame] = []
    for i in range(raw.shape[0]):
        safe = apply_safety(raw[i], cfg)
        frames.append(Frame.from_array(safe, name=f"f{i:04d}"))

    # Fill metadata
    total = len(frames)
    for i, f in enumerate(frames):
        f.frame_index = i
        f.total_frames = total

    return Show(frames=frames, fps=fps, name=show_name or Path(audio_path).stem)
