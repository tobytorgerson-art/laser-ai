"""End-to-end pipeline tests."""
from __future__ import annotations

from pathlib import Path

from laser_ai.generator.stub import StubGenerator
from laser_ai.pipeline.generate import generate_show_from_audio
from laser_ai.safety.postprocess import SafetyConfig


def test_generate_returns_show_with_correct_fps_and_frame_count(tiny_wav_path: Path):
    show = generate_show_from_audio(
        audio_path=tiny_wav_path,
        generator=StubGenerator(seed=0),
        fps=30.0,
        safety_cfg=SafetyConfig(),
    )
    # 1-sec audio at 30 fps → 30 frames
    assert abs(len(show.frames) - 30) <= 1
    assert show.fps == 30.0


def test_generate_produces_nonempty_frames(tiny_wav_path: Path):
    show = generate_show_from_audio(
        audio_path=tiny_wav_path,
        generator=StubGenerator(seed=0),
    )
    for f in show.frames:
        assert len(f.points) > 0


def test_generate_writes_valid_ilda_file(tiny_wav_path: Path, tmp_path: Path):
    from laser_ai.ilda.reader import read_ilda
    from laser_ai.ilda.writer import write_ilda

    show = generate_show_from_audio(
        audio_path=tiny_wav_path,
        generator=StubGenerator(seed=0),
    )
    out = tmp_path / "generated.ilda"
    write_ilda(show, out)
    recovered = read_ilda(out)
    assert len(recovered.frames) == len(show.frames)
