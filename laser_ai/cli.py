"""laser-ai command-line interface."""
from __future__ import annotations

from pathlib import Path

import click

from laser_ai.ilda.reader import read_ilda
from laser_ai.ilda.writer import write_ilda
from laser_ai.pipeline.generate import generate_show_from_audio
from laser_ai.safety.postprocess import SafetyConfig


@click.group()
@click.version_option()
def cli() -> None:
    """laser-ai: AI-driven ILDA laser show generation from audio."""


@cli.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output .ilda path.")
@click.option("--fps", type=float, default=30.0, show_default=True,
              help="Target frame rate.")
@click.option("--n-points", type=int, default=512, show_default=True,
              help="Points per frame (before safety pass).")
@click.option("--safety", type=click.Choice(["loose", "medium", "tight"]),
              default="medium", show_default=True, help="Safety post-processor strength.")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Generator random seed.")
def generate(audio: Path, output: Path, fps: float, n_points: int,
             safety: str, seed: int) -> None:
    """Generate an ILDA show from an audio file using the stub generator."""
    strengths = {"loose": 0.5, "medium": 1.0, "tight": 2.0}
    cfg = SafetyConfig(strength=strengths[safety])

    click.echo(f"loading audio: {audio}")
    click.echo(f"generating at {fps} fps, {n_points} pts/frame, safety={safety}...")

    from laser_ai.generator.stub import StubGenerator
    show = generate_show_from_audio(
        audio_path=audio,
        generator=StubGenerator(seed=seed),
        fps=fps,
        n_points=n_points,
        safety_cfg=cfg,
    )

    write_ilda(show, output)
    click.echo(f"wrote {output} ({len(show.frames)} frames, {show.duration_s:.2f}s)")


@cli.command()
@click.argument("ilda", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(ilda: Path) -> None:
    """Print summary info about an ILDA file."""
    show = read_ilda(ilda)
    total_pts = sum(len(f.points) for f in show.frames)
    click.echo(f"file:      {ilda}")
    click.echo(f"frames:    {len(show.frames)}")
    click.echo(f"duration:  {show.duration_s:.2f}s @ {show.fps} fps")
    click.echo(f"points:    {total_pts} total, avg {total_pts / max(1, len(show.frames)):.1f}/frame")


if __name__ == "__main__":
    cli()
