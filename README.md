# laser-ai

Generate ILDA laser show files from audio with a trained model.

## Status

Foundation complete (v0.1.0):
- ✓ ILDA reader/writer (formats 0/1/4/5 in, format 4 out)
- ✓ Arc-length frame resampling
- ✓ Audio loader (any sample rate → 44.1k mono float32)
- ✓ Per-frame audio features at 30 fps (mel + chroma + spectral + rms + onset + beat phase)
- ✓ Auto-pairing of audio+ILDA files in a folder
- ✓ Parametric primitive library (sine, lissajous, circle, grid)
- ✓ Rule-based stub generator (maps audio features → primitives)
- ✓ DAC-safety post-processor (velocity + dwell + caps)
- ✓ End-to-end pipeline: audio file → Show → ILDA
- ✓ CLI: `laser-ai generate` and `laser-ai info`

Coming in later plans:
- Plan 2: Frame VAE + Audio-to-Latent Sequencer + Colab training notebook (real ML)
- Plan 3: PyQt6 GUI app (dataset/train/generate/preview tabs)
- Plan 4: OpenGL preview renderer with audio sync
- Plan 5: Helios DAC real-time streaming

## Quickstart

### Install

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows git-bash
pip install -e ".[dev]"
```

### Generate a show (foundation mode, stub generator)

```bash
laser-ai generate path/to/song.mp3 -o path/to/output.ilda
```

### Inspect an ILDA file

```bash
laser-ai info path/to/output.ilda
```

### Run tests

```bash
pytest
```
