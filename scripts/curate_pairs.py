"""Walk an unstructured pile of laser-show folders and curate matched
(audio, ILDA) pairs into a flat directory with consistent stems.

Usage:
    python scripts/curate_pairs.py "ILDA frame shows" pairs_curated
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg"}
ILDA_EXTS = {".ild", ".ilda"}


def slugify(name: str) -> str:
    """lowercase, alnum-only stem suitable for laser-ai's stem-based discovery."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return s or "show"


def stem_similarity(a: str, b: str) -> int:
    """How many leading characters match (case-insensitive). Higher = better."""
    a, b = a.lower(), b.lower()
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


def pick_pair(audio_files: list[Path], ilda_files: list[Path]) -> tuple[Path, Path] | None:
    """Pick the best (audio, ilda) pair from the candidates in one folder.

    For Steve Milani layouts we prefer the *0.ild* file (main projector per his
    README); when no `0` suffix is present we fall back to highest stem-similarity
    with the audio stem, breaking ties by file size.
    """
    if not audio_files or not ilda_files:
        return None
    # One-shot best audio: take the largest mp3 in the folder
    audio = max(audio_files, key=lambda p: p.stat().st_size)

    # Score each ILDA candidate
    def score(p: Path) -> tuple[int, int, int]:
        s = p.stem.lower()
        # Prefer "*0" over "*1" (Steve Milani convention: 0=main, 1=satellite)
        zero_pref = 1 if s.endswith("0") else (-1 if s.endswith("1") else 0)
        return (
            zero_pref,
            stem_similarity(p.stem, audio.stem),
            p.stat().st_size,
        )

    ilda = max(ilda_files, key=score)
    return audio, ilda


def find_leaf_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    """For each folder anywhere under root, return (audio, ilda, suggested_stem)."""
    out: list[tuple[Path, Path, str]] = []
    seen_stems: set[str] = set()

    folders = {p.parent for p in root.rglob("*") if p.is_file()}
    for folder in sorted(folders):
        audio_files = [p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        ilda_files = [p for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() in ILDA_EXTS]
        picked = pick_pair(audio_files, ilda_files)
        if picked is None:
            continue
        audio, ilda = picked

        # Build a unique stem from the folder name + audio stem
        base = slugify(folder.name) or "show"
        stem = base
        i = 2
        while stem in seen_stems:
            stem = f"{base}_{i}"
            i += 1
        seen_stems.add(stem)
        out.append((audio, ilda, stem))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"source not found: {args.source}", file=sys.stderr)
        return 1

    pairs = find_leaf_pairs(args.source)
    if not pairs:
        print("no pairs found", file=sys.stderr)
        return 2

    print(f"found {len(pairs)} candidate pair(s):")
    for audio, ilda, stem in pairs:
        print(f"  {stem}: {audio.name}  <-> {ilda.name}  (from {audio.parent.name})")

    if args.dry_run:
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    for audio, ilda, stem in pairs:
        shutil.copy2(audio, args.dest / f"{stem}{audio.suffix.lower()}")
        shutil.copy2(ilda, args.dest / f"{stem}{ilda.suffix.lower()}")
    print(f"\ncopied {len(pairs)} pair(s) into {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
