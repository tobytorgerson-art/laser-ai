"""Auto-discover audio+ILDA pairs in a folder by filename stem."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from laser_ai.dataset.pair import AudioLaserPair


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg"}
ILDA_EXTS = {".ild", ".ilda"}


@dataclass(slots=True)
class DiscoveryResult:
    pairs: list[AudioLaserPair] = field(default_factory=list)
    unmatched_audio: list[Path] = field(default_factory=list)
    unmatched_ilda: list[Path] = field(default_factory=list)


def discover_pairs(folder: str | Path) -> DiscoveryResult:
    """Scan `folder` for paired audio+ILDA files, matched by filename stem."""
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")

    audio: dict[str, Path] = {}
    ilda: dict[str, Path] = {}

    for p in folder.iterdir():
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        stem = p.stem.lower()
        if ext in AUDIO_EXTS:
            audio[stem] = p
        elif ext in ILDA_EXTS:
            ilda[stem] = p

    result = DiscoveryResult()
    paired_stems = set(audio) & set(ilda)
    for stem in sorted(paired_stems):
        result.pairs.append(AudioLaserPair(audio_path=audio[stem], ilda_path=ilda[stem]))
    for stem in sorted(set(audio) - paired_stems):
        result.unmatched_audio.append(audio[stem])
    for stem in sorted(set(ilda) - paired_stems):
        result.unmatched_ilda.append(ilda[stem])
    return result
