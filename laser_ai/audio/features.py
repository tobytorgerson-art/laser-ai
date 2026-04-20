"""Per-frame audio features at a target frame rate (default 30 fps)."""
from __future__ import annotations

import numpy as np


N_MELS = 128
N_CHROMA = 12
# 128 mel + 12 chroma + 3 spectral + 1 rms + 1 onset + 1 beat_phase = 146
FEATURE_DIM = N_MELS + N_CHROMA + 3 + 1 + 1 + 1


def extract_features(samples: np.ndarray, sr: int, fps: float = 30.0) -> np.ndarray:
    """Extract per-frame features at `fps` frames/sec.

    Returns an (T, FEATURE_DIM) float32 array.
    """
    import librosa

    hop_length = max(1, int(round(sr / fps)))
    n_fft = 2048

    # Mel spectrogram (log-power)
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
    log_mel = log_mel.T  # (T, n_mels)
    # Normalize to [0, 1] approx
    log_mel = (log_mel + 80.0) / 80.0
    log_mel = np.clip(log_mel, 0.0, 1.0)

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
    ).T  # (T, 12)

    # Spectral descriptors
    centroid = librosa.feature.spectral_centroid(
        y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
    ).T.squeeze(-1)  # (T,)
    rolloff = librosa.feature.spectral_rolloff(
        y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
    ).T.squeeze(-1)  # (T,)
    flatness = librosa.feature.spectral_flatness(
        y=samples, n_fft=n_fft, hop_length=hop_length
    ).T.squeeze(-1)  # (T,)

    # Normalize descriptors by Nyquist so they live in [0, 1]
    nyquist = sr / 2.0
    centroid = centroid / nyquist
    rolloff = rolloff / nyquist
    # flatness is already [0, 1]

    # RMS
    rms = librosa.feature.rms(
        y=samples, frame_length=n_fft, hop_length=hop_length,
    ).T.squeeze(-1)  # (T,)

    # Onset strength
    onset = librosa.onset.onset_strength(
        y=samples, sr=sr, hop_length=hop_length,
    )  # (T,)
    # Normalize to [0, 1] per-song
    onset_max = max(onset.max(), 1e-6)
    onset = onset / onset_max

    # Beat tracking → phase
    _, beat_frames = librosa.beat.beat_track(
        y=samples, sr=sr, hop_length=hop_length,
    )
    beat_phase = _compute_beat_phase(len(onset), beat_frames)

    # Align everything to the shortest length (librosa can produce +/-1 differences)
    T = min(len(log_mel), len(chroma), len(centroid),
            len(rolloff), len(flatness), len(rms), len(onset), len(beat_phase))
    log_mel = log_mel[:T]
    chroma = chroma[:T]
    centroid = centroid[:T]
    rolloff = rolloff[:T]
    flatness = flatness[:T]
    rms = rms[:T]
    onset = onset[:T]
    beat_phase = beat_phase[:T]

    feats = np.concatenate([
        log_mel,
        chroma,
        np.stack([centroid, rolloff, flatness], axis=-1),
        rms[:, None],
        onset[:, None],
        beat_phase[:, None],
    ], axis=-1).astype(np.float32)

    assert feats.shape[1] == FEATURE_DIM, f"feature dim {feats.shape[1]} != {FEATURE_DIM}"
    return feats


def _compute_beat_phase(n_frames: int, beat_frames: np.ndarray) -> np.ndarray:
    """For each frame, return phase in [0, 1] within its current beat interval."""
    phase = np.zeros(n_frames, dtype=np.float32)
    if len(beat_frames) < 2:
        return phase
    for i in range(len(beat_frames) - 1):
        start = int(beat_frames[i])
        end = int(beat_frames[i + 1])
        end = min(end, n_frames)
        if end <= start:
            continue
        phase[start:end] = np.linspace(0.0, 1.0, end - start, endpoint=False)
    # Frames before first beat: phase 0
    # Frames after last beat: linearly extrapolate from previous interval
    last_start = int(beat_frames[-1])
    if last_start < n_frames:
        interval = int(beat_frames[-1]) - int(beat_frames[-2])
        for j in range(last_start, n_frames):
            phase[j] = ((j - last_start) % max(interval, 1)) / max(interval, 1)
    return phase
