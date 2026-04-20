# laser-ai

Generate ILDA laser show files from audio with a trained model.

## Status

ML training pipeline complete (v0.2.0):
- ✓ Everything in v0.1.0 (ILDA I/O, audio features, safety, CLI, stub generator)
- ✓ PyTorch Frame VAE (1D conv encoder/decoder, 64-dim latent)
- ✓ Causal Transformer sequencer (audio features → latent sequence)
- ✓ Chamfer + KL + RGB + travel loss suite
- ✓ Training data augmentation (rotate/flip/scale/hue)
- ✓ Checkpoint save/load (VAE + Sequencer bundle)
- ✓ TrainedGenerator swappable into the existing pipeline via `--model` flag
- ✓ Colab training notebook + orchestrator
- ✓ `prepare-bundle` for Colab dataset upload

Coming in later plans:
- Plan 3: PyQt6 GUI app (dataset/train/generate/preview tabs)
- Plan 4: OpenGL preview renderer with audio sync
- Plan 5: Helios DAC real-time streaming
- Possibly: CLAP embedding integration for richer audio semantics

## Full CLI

```bash
laser-ai info SHOW.ilda                         # inspect an ILDA file
laser-ai prepare-bundle DATA_DIR -o bundle.zip  # export training data for Colab
laser-ai train-vae DATA_DIR -o ck.pt            # train VAE locally on ILDA files
laser-ai train-sequencer DATA_DIR -c ck.pt -o ck.pt  # train sequencer on pairs
laser-ai generate SONG.mp3 -o OUT.ilda [--model ck.pt]  # generate show
```

## Training workflow (recommended)

1. Put 20+ `song.mp3` + `song.ilda` pairs (matching stems) in a folder.
2. `laser-ai prepare-bundle ./data -o bundle.zip` — exports the training bundle.
3. Open `colab/laser_ai_train.ipynb` in Google Colab (T4 GPU, free tier).
4. Upload `bundle.zip`, Run All, download `model.pt` (~30–60 min).
5. `laser-ai generate new-song.mp3 -o out.ilda --model model.pt`

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
