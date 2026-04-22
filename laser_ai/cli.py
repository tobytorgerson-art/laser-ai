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
              required=True, help="Output .ild path.")
@click.option("--fps", type=float, default=30.0, show_default=True,
              help="Target frame rate.")
@click.option("--n-points", type=int, default=512, show_default=True,
              help="Points per frame (before safety pass).")
@click.option("--safety", type=click.Choice(["loose", "medium", "tight"]),
              default="medium", show_default=True, help="Safety post-processor strength.")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Generator random seed (stub mode only).")
@click.option("--model", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Path to a trained checkpoint (.pt). Without it, uses stub.")
def generate(audio: Path, output: Path, fps: float, n_points: int,
             safety: str, seed: int, model: Path | None) -> None:
    """Generate an ILDA show from an audio file."""
    strengths = {"loose": 0.5, "medium": 1.0, "tight": 2.0}
    cfg = SafetyConfig(strength=strengths[safety])

    if model is not None:
        click.echo(f"loading model: {model}")
        from laser_ai.generator.trained import TrainedGenerator
        from laser_ai.models.checkpoint import load_checkpoint
        gen = TrainedGenerator(load_checkpoint(model))
        # Model n_points wins; warn handled inside TrainedGenerator
    else:
        click.echo("using stub generator (no --model given)")
        from laser_ai.generator.stub import StubGenerator
        gen = StubGenerator(seed=seed)

    click.echo(f"generating: {audio} -> {output}")
    show = generate_show_from_audio(
        audio_path=audio, generator=gen, fps=fps,
        n_points=n_points, safety_cfg=cfg,
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


@cli.command("train-vae")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output checkpoint path (.pt).")
@click.option("--epochs", type=int, default=20, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--n-points", type=int, default=512, show_default=True)
@click.option("--latent-dim", type=int, default=64, show_default=True)
@click.option("--hidden", type=int, default=128, show_default=True)
@click.option("--augment-mult", type=int, default=4, show_default=True,
              help="Virtual dataset size multiplier via augmentation.")
def train_vae_cmd(data_dir: Path, output: Path, epochs: int, batch_size: int,
                  lr: float, n_points: int, latent_dim: int, hidden: int,
                  augment_mult: int) -> None:
    """Train the Frame VAE on all ILDA files in DATA_DIR."""
    from laser_ai.augment.frame import AugmentConfig
    from laser_ai.audio.features import FEATURE_DIM
    from laser_ai.dataset.torch_dataset import FrameDataset
    from laser_ai.models.checkpoint import LaserAICheckpoint, save_checkpoint
    from laser_ai.models.sequencer import AudioToLatentSequencer, SequencerConfig
    from laser_ai.models.vae import FrameVAEConfig
    from laser_ai.training.train_vae import VAETrainConfig, train_vae

    ilda_paths = sorted(
        list(data_dir.rglob("*.ild")) + list(data_dir.rglob("*.ilda"))
    )
    if not ilda_paths:
        raise click.ClickException(f"no .ild/.ilda files found in {data_dir}")
    click.echo(f"found {len(ilda_paths)} ILDA file(s)")

    ds = FrameDataset(
        ilda_paths, n_points=n_points,
        augment_mult=augment_mult, augment_cfg=AugmentConfig(),
    )
    click.echo(f"dataset size (augmented): {len(ds)} frames")

    vae_cfg = FrameVAEConfig(n_points=n_points, latent_dim=latent_dim, hidden=hidden)
    train_cfg = VAETrainConfig(epochs=epochs, batch_size=batch_size, lr=lr)

    def _log(epoch: int, entry: dict) -> None:
        click.echo(
            f"  epoch {epoch:3d}: total={entry['total']:.4f}  "
            f"chamfer={entry['chamfer']:.4f}  rgb={entry['rgb']:.4f}  "
            f"travel={entry['travel']:.4f}  kl={entry['kl']:.4f}"
        )

    vae, _ = train_vae(ds, vae_cfg=vae_cfg, train_cfg=train_cfg, progress_callback=_log)

    # Default untrained sequencer placeholder; user will train it via train-sequencer
    # Use max_len=16384 (~9 min at 30 fps) so longer songs fit without truncation.
    seq_cfg = SequencerConfig(feature_dim=FEATURE_DIM, latent_dim=latent_dim, max_len=16384)
    ck = LaserAICheckpoint(
        vae=vae, vae_cfg=vae_cfg,
        sequencer=AudioToLatentSequencer(seq_cfg), seq_cfg=seq_cfg,
        audio_feature_dim=FEATURE_DIM, fps=30.0,
    )
    save_checkpoint(ck, output)
    click.echo(f"saved checkpoint: {output}")


@cli.command("train-sequencer")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-c", "--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Input VAE-trained checkpoint.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output path (.pt) for the updated checkpoint.")
@click.option("--epochs", type=int, default=30, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--lr", type=float, default=5e-4, show_default=True)
def train_sequencer_cmd(data_dir: Path, checkpoint: Path, output: Path,
                        epochs: int, batch_size: int, lr: float) -> None:
    """Train the Sequencer on audio+ILDA pairs in DATA_DIR."""
    from laser_ai.dataset.discovery import discover_pairs
    from laser_ai.models.checkpoint import LaserAICheckpoint, load_checkpoint, save_checkpoint
    from laser_ai.training.prepare import build_sequencer_dataset
    from laser_ai.training.train_sequencer import SequencerTrainConfig, train_sequencer

    ck = load_checkpoint(checkpoint)
    result = discover_pairs(data_dir)
    if not result.pairs:
        raise click.ClickException(f"no audio+ILDA pairs found in {data_dir}")
    click.echo(f"found {len(result.pairs)} pair(s)")

    pairs = build_sequencer_dataset(
        result.pairs, vae=ck.vae,
        n_points=ck.vae_cfg.n_points, fps=ck.fps,
    )
    train_cfg = SequencerTrainConfig(epochs=epochs, batch_size=batch_size, lr=lr)

    def _log(epoch: int, entry: dict) -> None:
        click.echo(f"  epoch {epoch:3d}: mse={entry['mse']:.6f}")

    sequencer, _ = train_sequencer(pairs, seq_cfg=ck.seq_cfg, train_cfg=train_cfg,
                                   progress_callback=_log)

    updated = LaserAICheckpoint(
        vae=ck.vae, vae_cfg=ck.vae_cfg,
        sequencer=sequencer, seq_cfg=ck.seq_cfg,
        audio_feature_dim=ck.audio_feature_dim, fps=ck.fps,
    )
    save_checkpoint(updated, output)
    click.echo(f"saved checkpoint: {output}")


@cli.command("prepare-bundle")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output bundle zip path.")
def prepare_bundle_cmd(data_dir: Path, output: Path) -> None:
    """Zip a training folder into a bundle suitable for Colab."""
    from laser_ai.bundle.pack import pack_dataset
    pack_dataset(data_dir, output)
    click.echo(f"wrote {output}")


if __name__ == "__main__":
    cli()
