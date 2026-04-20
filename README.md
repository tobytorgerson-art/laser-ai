# laser-ai

Generate ILDA laser show files from audio with a trained model.

## Status

Foundation (headless CLI, stub generator). GUI, real models, Colab training, and Helios DAC come in later plans.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows (bash: source .venv/Scripts/activate)
pip install -e ".[dev]"
```

## Generate a show

```bash
laser-ai generate path/to/song.mp3 -o path/to/output.ilda
```

## Run tests

```bash
pytest
```
