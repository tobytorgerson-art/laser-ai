"""Audio-Laser pairing dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AudioLaserPair:
    """A paired audio file + ILDA file, optionally with an alignment offset."""
    audio_path: Path
    ilda_path: Path
    offset_seconds: float = 0.0  # +ve: ILDA starts this many seconds after audio

    @property
    def stem(self) -> str:
        return self.audio_path.stem
