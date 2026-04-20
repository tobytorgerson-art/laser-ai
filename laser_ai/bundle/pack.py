"""Pack a training data folder into a portable zip for Colab upload."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from laser_ai.dataset.discovery import discover_pairs


_BUNDLE_VERSION = 1


def pack_dataset(data_dir: str | Path, out_zip: str | Path) -> None:
    """Zip up all audio+ILDA pairs found in `data_dir` into `out_zip`."""
    data_dir = Path(data_dir)
    out_zip = Path(out_zip)

    result = discover_pairs(data_dir)
    if not result.pairs:
        raise ValueError(
            f"no audio+ILDA pairs found in {data_dir}; nothing to bundle"
        )

    pairs_meta = []
    for p in result.pairs:
        pairs_meta.append({
            "stem": p.stem,
            "audio": p.audio_path.name,
            "ilda": p.ilda_path.name,
            "offset_seconds": p.offset_seconds,
            "audio_bytes": p.audio_path.stat().st_size,
            "ilda_bytes": p.ilda_path.stat().st_size,
        })

    index = {
        "version": _BUNDLE_VERSION,
        "pairs": pairs_meta,
    }

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.json", json.dumps(index, indent=2))
        for p in result.pairs:
            zf.write(p.audio_path, arcname=f"pairs/{p.audio_path.name}")
            zf.write(p.ilda_path, arcname=f"pairs/{p.ilda_path.name}")
