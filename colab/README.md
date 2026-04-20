# Colab training for laser-ai

## Workflow

1. On your machine, run:
   ```
   laser-ai prepare-bundle /path/to/your/data -o bundle.zip
   ```
2. Open `laser_ai_train.ipynb` in Google Colab.
3. Runtime → Change runtime type → **T4 GPU** (free tier is sufficient).
4. Upload `bundle.zip` into the file panel (left sidebar).
5. Runtime → **Run all**.
6. Wait ~30–60 minutes. Loss should steadily decrease.
7. Download `model.pt` from the file panel.
8. Locally, run:
   ```
   laser-ai generate your-song.mp3 -o out.ilda --model model.pt
   ```

## Troubleshooting

- **"no bundle.zip"** — drag the zip file into the Colab file panel first.
- **pip install fails** — the notebook installs `laser-ai` from a Git URL. Edit cell 1 to point at your fork of this repo.
- **CUDA OOM** — halve `vae_batch_size` or `hidden` in `colab_train.run(...)`.
