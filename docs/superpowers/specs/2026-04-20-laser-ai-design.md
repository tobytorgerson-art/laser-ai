# laser-ai — Design Spec

**Date:** 2026-04-20
**Working name:** `laser-ai`
**Platform:** Windows 11, Python, local-first
**Target user:** laser-show hobbyist with a Helios DAC, moderate beginner in code

## 1. Goal

Train a model on paired `.ild`/`.ilda` laser-show files and their matching audio (`.mp3`/`.wav`) so it learns what good abstract laser shows look like synced to music. Then use the trained model to generate new `.ilda` files for new songs.

## 2. Scope and constraints

- **Training data:** 20–100 paired ILDA + audio files (modest dataset).
- **Content style:** abstract beam shows / procedural generative art (no graphics/logos).
- **Hardware:**
  - User's machine: Windows 11, CPU-only, no dedicated GPU.
  - Training runs on **free Google Colab (T4 GPU)** — one-off heavy compute only.
  - Inference, preview, and Helios playback run **locally on CPU**.
- **Output:** RGB color ILDA Format 4 or 5, ~25–30k points per second, ILDA-safe and Helios-safe.
- **User interface:** turnkey desktop GUI (beginner-friendly).
- **Operating mode:** offline batch generation first; real-time Helios streaming as phase 2.

## 3. High-level architecture

```
[.mp3/.wav + .ilda pairs]
        ↓  (local preprocessing in the app)
[dataset bundle: frames + audio features + latents]
        ↓  (upload to Colab)
[train Frame VAE]  →  [train Audio→Latent Sequencer]
        ↓  (download)
[packaged model]
        ↓
[new song] → [features] → [Sequencer] → [latents] → [VAE decoder] → [points]
        ↓
[ILDA safety post-processor]
        ↓
[.ilda file + on-screen preview + Helios DAC stream]
```

Three user-visible pieces:
1. **`laser-ai` GUI app** (PyQt6). All user-facing work (data, training launch, generation, preview, playback).
2. **Colab training notebook.** Takes exported dataset bundle, trains both models, returns a packaged model file.
3. **Local inference engine.** PyTorch CPU (with ONNX Runtime as optional faster path).

User's ILDA library and audio never leave the machine except into a Colab session the user controls.

## 4. Laser frame representation

Neural networks need fixed-size tensors; ILDA stores variable-length point sequences. Unify on a fixed schema.

- **Fixed-count resampling.** Every training frame is resampled to exactly **N = 512 points** via even arc-length spacing along the drawn path.
- **Blanking handling.** Blanked/travel points are encoded as zero-brightness points with an `is_travel` flag (not dropped).
- **Per-point vector:** `(x, y, r, g, b, is_travel)` — 6 values, xy normalized to `[-1, 1]`, rgb to `[0, 1]`, `is_travel` binary.
- **Per-frame tensor:** shape `(512, 6)`.
- **Temporal axis:** fixed **30 fps**. A 3-minute song = 5,400 frames.

Rationale for 512 points × 30 fps:
- 512 × 30 = 15,360 pps visible budget; safety post-processor may add up to ~700 points/frame of dwells and blanked transits, total ≤ 1200 points/frame (≈ 36k pps) — within Helios's typical safe operating range and well below the 65k pps hardware ceiling.
- 30 fps matches typical ILDA authoring and is flicker-free on real hardware.
- 512 × 6 is small enough for a CPU-friendly VAE yet large enough to represent complex figures.

**Data augmentation (training only):**
- Rotation ±15°, horizontal/vertical flip, scale jitter ±10%.
- Hue rotation 0–360° on the color channels.
- Time-axis jitter on audio pairing (±100 ms) to teach tolerance for slightly-misaligned pairs.

## 5. Audio representation

Per-frame audio feature vector, sampled at the same 30 fps as the laser frames:

- **Log-mel spectrogram patch:** 128 mel bins × 9 frames of context (±4) — timbre, spectral character.
- **Rhythm:** BPM, beat phase (0–1 inside the current beat), onset strength.
- **Dynamics:** RMS loudness envelope, ~100 ms smoothing.
- **Spectral descriptors:** centroid, rolloff, flatness.
- **Harmony:** 12-dim chroma vector.
- **Pre-trained semantic embedding:** 512-dim [CLAP](https://github.com/LAION-AI/CLAP) embedding per ~1 s window, stretched to per-frame. Acts as a pre-trained "musical understanding" signal to offset the small dataset size.

Raw per-frame vector: ~1200 dims → compressed to **128 dims** by a small learned audio adapter before feeding the sequencer.

Feature extraction for a 3-minute song: ~5–10 s on CPU with librosa + cached CLAP. Cached as `.npz` per song.

## 6. Models

Two models, trained separately in Colab.

### 6.1 Frame VAE

- **Input:** frame tensor `(512, 6)`.
- **Encoder:** 1D conv stack over the point sequence → `μ` and `logσ²` → **64-dim latent `z`**.
- **Decoder:** `z` → transposed 1D conv stack → reconstructed `(512, 6)`.
- **Loss:** Chamfer distance on XY + MSE on RGB + BCE on `is_travel` + KL divergence.
- **Training data:** every frame from every training pair × 8 augmentations.
- **Params:** ~1.5M. Colab T4 training: 20–40 min. CPU inference: ~200 fps.

### 6.2 Audio-to-Latent Sequencer

- **Input:** `(T, 128)` audio features.
- **Output:** `(T, 64)` frame latents.
- **Architecture:** small **Transformer** — 6 layers, 256 hidden dim, 4 attention heads, causal attention (so it can later run real-time). ~2M params.
- **Loss:** MSE on predicted latents + perceptual loss (decode predicted vs. real latent through the VAE, compare frames) + a beat-sync reward encouraging predicted latent changes to align with detected onsets.
- **Training data:** `(audio_features, true_VAE_latents)` extracted from training songs.
- **Params:** ~2M. Colab T4 training: 30–60 min. CPU inference: ~500 fps.

### 6.3 Inference pipeline speed

End-to-end generation for a 3-minute song on typical CPU: **5–15 seconds**. Fast enough to regenerate with different random seeds many times.

## 7. Safety post-processor

Between decoder output and `.ilda`/DAC output. Deterministic rules, not learned. Guarantees safe output regardless of what the VAE produces.

1. **Point-rate cap.** Total points/frame ≤ 1200 (≈ 36k pps at 30 fps, Helios-safe); downsample visible segments by arc-length if exceeded.
2. **Velocity limiting.** Consecutive points must respect galvo slew rate; interpolate extra points or insert blanked transit when jumps are too large.
3. **Dwell points on corners/endpoints.** Auto-insert 3–5 repeated points at sharp angles and stroke endpoints.
4. **Blanking transitions.** Replace direct visible paths between distant segments with blanked moves plus blanking-shift points.
5. **Color clipping.** RGB clamped to `[0, 255]` (or 12-bit for Helios direct output).
6. **Coordinate clamping.** XY clamped to signed-16 ILDA range `[-32768, 32767]` with a user-configurable safety margin.
7. **Frame-to-frame smoothing.** Light temporal filter on *latents* (not points) to avoid jarring 33 ms shape jumps.

**User control:** single "safety strength" slider — tight (conservative, smooth) / medium (default) / loose (more raw model output for creative effect).

## 8. Preview renderer

On-screen laser simulator embedded in the GUI.

- **Tech:** PyQt6 `QOpenGLWidget`, line segments with additive blending + gaussian bloom → characteristic glowy-beam look.
- **Color:** per-point RGB from the frame.
- **Persistence-of-vision:** last 2–3 frames rendered with fading alpha to simulate what an eye actually sees.
- **Controls:** play/pause, scrub, loop, volume, zoom, pan.
- **Debug toggles:** show blanked-transit moves; "compare mode" (training ILDA vs. generated, synchronized, same audio).
- **Audio playback:** `sounddevice` with sample-accurate sync to the laser frame index.

## 9. GUI layout

Single PyQt6 window, left sidebar of tabs.

### 9.1 Dataset tab
- "Add folder" button — auto-pairs `.mp3`/`.wav` with `.ild`/`.ilda` by filename stem.
- Unmatched files listed; "manual pair" dialog to pair any audio with any ILDA.
- Per-pair row: waveform strip, ILDA frame thumbnails, **alignment offset slider** (±2 s, 10 ms steps), play-aligned preview button.
- Auto-extracts and caches audio features on add.
- Status: ✓ paired + aligned + cached / ⚠ needs attention.

### 9.2 Train tab
- **"Prepare training bundle"** — packages dataset + features to `.zip` for Colab.
- **"Open Colab notebook"** — opens the pre-made notebook in a browser; user drags `.zip` in, Run All, downloads `model.pt`.
- **"Import trained model"** — drop downloaded file into app's model slot.
- Training log panel (last run's saved log; live log if Colab API integration used).

### 9.3 Generate tab
- Pick audio file → pick active model → **Generate** button.
- Progress bar.
- Options: random seed, safety-strength slider, variation temperature, target frame rate (default 30 fps).
- On completion: auto-switch to Preview tab; Save `.ilda` / Save `.ild` buttons.

### 9.4 Preview tab
- OpenGL simulator (section 8).
- Synchronized audio.
- Compare-mode toggle.
- "Send to Helios DAC" button (phase 2).

### 9.5 Settings tab
- Model path, default frame rate, default safety strength, CLAP on/off, output folder, Helios DAC settings.

**Visual style:** dark theme, minimal chrome, monitor-friendly.

## 10. Real-time / Helios streaming (phase 2)

After offline generation is solid:

- Add a "Live" mode in the Preview tab.
- Audio in (microphone or loopback of system audio) streamed through the same feature extractor with a small rolling window.
- Sequencer runs in streaming mode (causal attention already supports this), producing latents with ~100–200 ms lookahead latency.
- Latents → VAE decoder → safety post-processor → `HeliosDacAPI` (via `helios_dac` C++ SDK wrapped with `ctypes`) → real laser output.
- Calibration UI: test pattern output (circle, grid) to verify geometry and color.
- Emergency kill button + configurable safety perimeter.

## 11. Training workflow (what the user does)

1. Drop a folder of `song.mp3` + `song.ilda` pairs into the Dataset tab.
2. Confirm pairings and alignments (slider per pair).
3. Click "Prepare training bundle" → a `.zip` file is written.
4. Click "Open Colab notebook" → browser opens the notebook.
5. In Colab: drag `.zip` into the file panel, click Runtime → Run All. Wait ~1 hour.
6. Download `model.pt` when training completes.
7. In the app, click "Import trained model" and pick the file.
8. Done. The model is live.

## 12. Generation workflow (what the user does)

1. Open Generate tab.
2. Pick a song.
3. Click Generate. Wait ~5–15 s.
4. App switches to Preview tab. Watch the show + hear the song synced.
5. Regenerate with a new seed if desired. Tweak safety/temperature sliders.
6. Save as `.ilda` or `.ild`. Drop into LaserShowGen / QuickShow / BEYOND, or (phase 2) send directly to the Helios DAC.

## 13. File and folder layout

```
laser-ai/
  app/                     # PyQt6 GUI
    main.py
    tabs/
      dataset.py
      train.py
      generate.py
      preview.py
      settings.py
    preview/
      gl_renderer.py
      audio_player.py
  core/
    ilda/
      reader.py
      writer.py
      resample.py           # arc-length resampling to N=512
    audio/
      features.py           # librosa + CLAP extraction
      cache.py
    models/
      vae.py
      sequencer.py
      adapter.py
      io.py                 # pack/unpack model.pt
    safety/
      postprocess.py        # deterministic DAC-safe layer
    generate/
      pipeline.py           # song → features → latents → points → safe frames
    helios/
      driver.py             # ctypes wrapper around HeliosDacAPI (phase 2)
  colab/
    train_laser_ai.ipynb    # the Colab notebook user runs
    train_vae.py            # imported by the notebook
    train_sequencer.py      # imported by the notebook
  tests/
    ...                     # unit tests for each core module
  data/                     # user's dataset cache (gitignored)
    pairs/
    features/
  models/                   # imported trained models (gitignored)
  docs/
    superpowers/specs/2026-04-20-laser-ai-design.md
  pyproject.toml            # dependencies
  README.md
```

## 14. Key dependencies

- `PyQt6` — GUI
- `PyOpenGL` — preview renderer
- `numpy`, `scipy` — numerics
- `librosa`, `soundfile` — audio features, audio I/O
- `sounddevice` — audio playback in GUI
- `torch` (CPU wheel locally; CUDA in Colab) — models
- `onnxruntime` — optional faster local inference
- `laff` or custom ILDA codec — `.ild`/`.ilda` read/write (write custom if needed, format is well-specified)
- `transformers` + `laion_clap` — CLAP embeddings
- `pytest` — tests

## 15. Testing approach

Unit tests for the parts where bugs are cheap to create and expensive to debug:

- **ILDA reader/writer:** round-trip a known file, assert byte-equal (or semantically equal after documented normalizations).
- **Arc-length resampler:** N points in, N points out, correct path length preservation within tolerance.
- **Audio feature extractor:** reproducible on a fixture `.wav` (deterministic for fixed inputs).
- **Safety post-processor:** adversarial inputs (out-of-range, >2000 points, zero-length frames) all produce valid ILDA-safe frames.
- **Model I/O:** save → load → inference yields identical output.
- **Generation pipeline end-to-end:** fixture song + tiny fixture model → non-empty, safe, correct-length ILDA file.

Higher-level "is the output pretty" is human-evaluated via the preview tab; no automated test for aesthetics.

## 16. Open questions / future work

- **Multi-style model.** Once there's more data, a style token per training song could let the user pick "more like *song X*" at generation time.
- **Fine-tune Colab model locally.** Possible with LoRA-style adapters if CPU training of small adapters proves viable.
- **Real-time latency tuning.** Phase 2; current transformer could be swapped for a smaller streaming-specific architecture if lookahead is too high.
- **Helios SDK binding.** Decide whether to use `ctypes` to the official C++ SDK or a Python community wrapper.
- **Genre/BPM-aware temperature.** Temperature slider could auto-scale with detected musical intensity.

## 17. Success criteria

- Can add a folder of ILDA+audio pairs and the app pairs, aligns, caches features without user intervention on correctly-named files.
- Can export a training bundle, train in Colab end-to-end, import the resulting model, and generate a show for a new song without touching a terminal.
- Generated `.ilda` files load and play cleanly in LaserShowGen/QuickShow/BEYOND *and* on the Helios DAC without DAC errors or visible artifacts.
- Preview accurately mirrors what real hardware outputs (visual parity check).
- Generated shows feel musically-driven to the user — onsets, drops, and energy changes visibly land on the music.
