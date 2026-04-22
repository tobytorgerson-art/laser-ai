# laser-ai Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, working end-to-end skateboard of the laser-ai system — read ILDA + audio pairs, extract features, generate ILDA output through a stub generator + real safety post-processor, and drive it from a CLI. Real VAE/sequencer models, GUI, Colab notebook, OpenGL preview, and Helios DAC arrive in subsequent plans.

**Architecture:** Python 3.11+, package layout per spec §13. All offline. Stub generator today is a rule-based primitive synthesizer (sine/lissajous/tunnel primitives modulated by audio features) that obeys the same interface the real VAE+Sequencer will implement in plan 2, so plumbing is future-proof. Safety post-processor is the *real* production code from day one — it is deterministic and identical regardless of what produces upstream points.

**Tech Stack:** Python, numpy, scipy, librosa, soundfile, click (CLI), pytest, structlog. No PyTorch yet (added in plan 2). No PyQt yet (added in plan 3).

**Spec:** `docs/superpowers/specs/2026-04-20-laser-ai-design.md`

---

## File Structure

This plan creates:

```
laser-ai/
  pyproject.toml                       # project metadata, dependencies
  README.md                            # quickstart
  .gitignore
  .python-version                      # pin Python 3.11

  laser_ai/
    __init__.py
    cli.py                             # `laser-ai` command entry point

    ilda/
      __init__.py
      types.py                         # Point, Frame, Show dataclasses
      reader.py                        # parse .ild / .ilda bytes → Show
      writer.py                        # Show → .ild / .ilda bytes
      resample.py                      # Frame → N-point Frame (arc-length)

    audio/
      __init__.py
      features.py                      # librosa features per-frame, 30 fps
      loader.py                        # load mp3/wav to float32 mono 44.1k
      cache.py                         # .npz feature caching

    dataset/
      __init__.py
      pair.py                          # AudioLaserPair dataclass
      discovery.py                     # auto-pair files from a folder

    generator/
      __init__.py
      base.py                          # Generator interface (stub + real-later)
      stub.py                          # rule-based primitive synthesizer
      primitives.py                    # sine, lissajous, tunnel, spiral, grid

    safety/
      __init__.py
      postprocess.py                   # deterministic DAC-safe layer

    pipeline/
      __init__.py
      generate.py                      # audio → features → generator → safety → Show

  tests/
    __init__.py
    conftest.py                        # pytest fixtures: sample ILDA + audio
    fixtures/
      tiny_show.ilda                   # tiny hand-rolled 3-frame ILDA
      tiny_audio.wav                   # 1-sec 440Hz sine, mono 44.1k
    ilda/
      test_reader.py
      test_writer.py
      test_resample.py
      test_roundtrip.py
    audio/
      test_features.py
      test_loader.py
    dataset/
      test_discovery.py
    generator/
      test_stub.py
      test_primitives.py
    safety/
      test_postprocess.py
    pipeline/
      test_generate.py
    cli/
      test_cli.py
```

Files that change together live together. Each module has one responsibility. No file grows past ~200 lines in this plan.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `.python-version`
- Create: `laser_ai/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize git**

```bash
cd "F:/laser ai"
git init
git config --local user.name "laser-ai"
git config --local user.email "laser-ai@local"
```

- [ ] **Step 2: Write `.python-version`**

```
3.11
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
dist/
build/

# user data (large, private)
data/pairs/
data/features/
models/
*.ilda
*.ild
*.mp3
*.wav
!tests/fixtures/**

# os / ide
.DS_Store
.vscode/
.idea/
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "laser-ai"
version = "0.1.0"
description = "AI-driven ILDA laser show generation from audio"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "librosa>=0.10",
    "soundfile>=0.12",
    "click>=8.1",
    "structlog>=24.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
]

[project.scripts]
laser-ai = "laser_ai.cli:cli"

[tool.setuptools.packages.find]
include = ["laser_ai*"]
exclude = ["tests*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
```

- [ ] **Step 5: Write `README.md`**

```markdown
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
```

- [ ] **Step 6: Write `laser_ai/__init__.py`**

```python
"""laser-ai: AI-driven ILDA laser show generation from audio."""

__version__ = "0.1.0"
```

- [ ] **Step 7: Write `tests/__init__.py`**

Empty file.

- [ ] **Step 8: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def tiny_wav_path(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "tiny_audio.wav"
    if not path.exists():
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        sr = 44100
        duration_s = 1.0
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(path, samples, sr, subtype="PCM_16")
    return path
```

- [ ] **Step 9: Install and verify**

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
pytest --collect-only
```

Expected: No tests collected yet, but no import errors. Package installs cleanly.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml README.md .gitignore .python-version laser_ai tests
git commit -m "chore: scaffold laser-ai project"
```

---

## Task 2: ILDA data types

**Files:**
- Create: `laser_ai/ilda/__init__.py`
- Create: `laser_ai/ilda/types.py`
- Create: `tests/ilda/__init__.py`

The ILDA format has formats 0/1 (monochrome), 4/5 (true color RGB). We support reading all four and default to writing Format 4 (3D RGB) for broadest compatibility with modern software and Helios.

- [ ] **Step 1: Write `laser_ai/ilda/__init__.py`**

```python
"""ILDA file format I/O."""

from laser_ai.ilda.types import Frame, Point, Show

__all__ = ["Frame", "Point", "Show"]
```

- [ ] **Step 2: Write `tests/ilda/__init__.py`**

Empty file.

- [ ] **Step 3: Write `laser_ai/ilda/types.py`**

```python
"""Core ILDA data types."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Point:
    """A single laser point.

    x, y are signed int16 in the ILDA coordinate space [-32768, 32767].
    r, g, b are uint8 [0, 255].
    is_blank means the laser is off during this point (travel move).
    is_last_point marks the final point of a frame.
    """
    x: int
    y: int
    r: int = 0
    g: int = 0
    b: int = 0
    is_blank: bool = False
    is_last_point: bool = False


@dataclass(slots=True)
class Frame:
    """A single ILDA frame: an ordered list of Points."""
    points: list[Point] = field(default_factory=list)
    name: str = ""
    company: str = ""
    frame_index: int = 0
    total_frames: int = 1

    def to_array(self) -> np.ndarray:
        """Return an (N, 6) float32 array: (x, y, r, g, b, is_blank).

        Coordinates are normalized to [-1, 1]; colors to [0, 1]; is_blank to {0, 1}.
        """
        if not self.points:
            return np.zeros((0, 6), dtype=np.float32)
        arr = np.zeros((len(self.points), 6), dtype=np.float32)
        for i, p in enumerate(self.points):
            arr[i, 0] = p.x / 32768.0
            arr[i, 1] = p.y / 32768.0
            arr[i, 2] = p.r / 255.0
            arr[i, 3] = p.g / 255.0
            arr[i, 4] = p.b / 255.0
            arr[i, 5] = 1.0 if p.is_blank else 0.0
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray, name: str = "") -> "Frame":
        """Build a Frame from an (N, 6) float32 array in the format of to_array()."""
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"expected (N, 6) array, got {arr.shape}")
        pts: list[Point] = []
        n = arr.shape[0]
        for i in range(n):
            x = int(np.clip(arr[i, 0], -1.0, 1.0) * 32767)
            y = int(np.clip(arr[i, 1], -1.0, 1.0) * 32767)
            r = int(np.clip(arr[i, 2], 0.0, 1.0) * 255)
            g = int(np.clip(arr[i, 3], 0.0, 1.0) * 255)
            b = int(np.clip(arr[i, 4], 0.0, 1.0) * 255)
            is_blank = bool(arr[i, 5] >= 0.5)
            is_last = i == n - 1
            pts.append(Point(x=x, y=y, r=r, g=g, b=b, is_blank=is_blank, is_last_point=is_last))
        return cls(points=pts, name=name)


@dataclass(slots=True)
class Show:
    """A complete laser show: ordered list of Frames + metadata."""
    frames: list[Frame] = field(default_factory=list)
    fps: float = 30.0
    name: str = ""
    company: str = "laser-ai"

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0.0
```

- [ ] **Step 4: Write a test for `Frame.to_array` / `from_array`**

Create `tests/ilda/test_types.py`:

```python
"""Tests for ILDA data types."""
from __future__ import annotations

import numpy as np

from laser_ai.ilda.types import Frame, Point


def test_frame_to_array_roundtrips_through_from_array():
    original = Frame(points=[
        Point(x=0, y=0, r=255, g=0, b=0),
        Point(x=16000, y=-16000, r=0, g=255, b=128, is_blank=True),
        Point(x=-32768, y=32767, r=100, g=100, b=100),
    ])
    arr = original.to_array()
    recovered = Frame.from_array(arr)

    assert len(recovered.points) == 3
    # allow ±1 off due to int->float->int quantization
    for p_orig, p_rec in zip(original.points, recovered.points):
        assert abs(p_orig.x - p_rec.x) <= 2
        assert abs(p_orig.y - p_rec.y) <= 2
        assert abs(p_orig.r - p_rec.r) <= 1
        assert abs(p_orig.g - p_rec.g) <= 1
        assert abs(p_orig.b - p_rec.b) <= 1
        assert p_orig.is_blank == p_rec.is_blank


def test_frame_from_array_rejects_wrong_shape():
    import pytest
    with pytest.raises(ValueError, match="expected"):
        Frame.from_array(np.zeros((10, 5), dtype=np.float32))


def test_empty_frame_to_array_is_empty():
    arr = Frame().to_array()
    assert arr.shape == (0, 6)
```

- [ ] **Step 5: Run tests — they should pass**

```bash
pytest tests/ilda/test_types.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add laser_ai/ilda/__init__.py laser_ai/ilda/types.py tests/ilda/
git commit -m "feat(ilda): add Point, Frame, Show data types"
```

---

## Task 3: ILDA reader

**Files:**
- Create: `laser_ai/ilda/reader.py`
- Create: `tests/ilda/test_reader.py`
- Create: `tests/fixtures/make_fixture_ilda.py` (helper, not a test)

The ILDA spec: https://www.ilda.com/resources/StandardsDocs/ILDA_IDTF14_rev011.pdf. Each section starts with `"ILDA"` magic, a format byte, header fields, then point records.

We support formats 0 (3D indexed), 1 (2D indexed), 4 (3D true-color), 5 (2D true-color). We ignore indexed palette resolution of formats 0/1 for now — treat indexed colors as white (simple; real-world laser authoring almost exclusively uses 4/5 today; if a training file is format 0/1 the user will see monochrome-white output in preview which is fine for this foundation pass).

- [ ] **Step 1: Write `tests/fixtures/make_fixture_ilda.py`**

This script generates a known tiny ILDA file deterministically. Run once to create `tests/fixtures/tiny_show.ilda`.

```python
"""Generate a deterministic tiny ILDA fixture for tests.

Run: python tests/fixtures/make_fixture_ilda.py
"""
from __future__ import annotations

import struct
from pathlib import Path


def write_tiny_ilda(path: Path) -> None:
    """Three frames, each with a small square pattern, ILDA format 4."""
    frames = []
    total_frames = 3

    def square_points(size: int, color: tuple[int, int, int]) -> list[tuple]:
        r, g, b = color
        # 4 corners + close, 5 points, last one flagged last-point
        pts = [
            (-size, -size, 0, False, r, g, b),
            ( size, -size, 0, False, r, g, b),
            ( size,  size, 0, False, r, g, b),
            (-size,  size, 0, False, r, g, b),
            (-size, -size, 0, True,  r, g, b),  # last
        ]
        return pts

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    sizes = [8000, 16000, 24000]

    for idx in range(total_frames):
        pts = square_points(sizes[idx], colors[idx])
        # ILDA format 4 header (32 bytes)
        header = b"ILDA" + struct.pack(">I", 4)  # magic + format
        header += b"FRAME   ".ljust(8, b" ")[:8]    # name (8 bytes)
        header += b"LASERAI ".ljust(8, b" ")[:8]   # company (8 bytes)
        header += struct.pack(">HHHHH",
                              len(pts),            # records
                              idx,                 # frame no
                              total_frames,        # total
                              0,                   # projector
                              0)                   # future
        frames.append(header)
        # records: 8 bytes per point for format 4
        for (x, y, z, is_last, r, g, b) in pts:
            status = 0
            if is_last:
                status |= 0b10000000
            frames.append(struct.pack(">hhhBBBB", x, y, z, status, b, g, r))

    # End-of-file section (0 records)
    eof_header = b"ILDA" + struct.pack(">I", 4)
    eof_header += b"        " + b"        "
    eof_header += struct.pack(">HHHHH", 0, total_frames, total_frames, 0, 0)
    frames.append(eof_header)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for chunk in frames:
            f.write(chunk)


if __name__ == "__main__":
    here = Path(__file__).parent
    write_tiny_ilda(here / "tiny_show.ilda")
    print(f"wrote {here / 'tiny_show.ilda'}")
```

- [ ] **Step 2: Run the fixture generator**

```bash
python tests/fixtures/make_fixture_ilda.py
```

Expected: prints path; `tests/fixtures/tiny_show.ilda` now exists (should be ~112 bytes).

- [ ] **Step 3: Write `tests/ilda/test_reader.py`**

```python
"""Tests for the ILDA reader."""
from __future__ import annotations

from pathlib import Path

import pytest

from laser_ai.ilda.reader import read_ilda


FIXTURE = Path(__file__).parent.parent / "fixtures" / "tiny_show.ilda"


def test_read_tiny_fixture_returns_three_frames():
    show = read_ilda(FIXTURE)
    assert len(show.frames) == 3


def test_read_tiny_fixture_first_frame_is_red_square():
    show = read_ilda(FIXTURE)
    frame = show.frames[0]
    assert len(frame.points) == 5
    # All points are red
    for p in frame.points:
        assert p.r == 255
        assert p.g == 0
        assert p.b == 0
    # Last point flagged
    assert frame.points[-1].is_last_point
    # First 4 points not flagged as last
    for p in frame.points[:4]:
        assert not p.is_last_point


def test_read_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_ilda(Path("nonexistent.ilda"))


def test_read_rejects_non_ilda_bytes(tmp_path):
    p = tmp_path / "bad.ilda"
    p.write_bytes(b"not an ilda file at all")
    with pytest.raises(ValueError, match="ILDA magic"):
        read_ilda(p)
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest tests/ilda/test_reader.py -v
```

Expected: all FAIL with ImportError for `laser_ai.ilda.reader`.

- [ ] **Step 5: Write `laser_ai/ilda/reader.py`**

```python
"""Parse .ild / .ilda files into Show objects."""
from __future__ import annotations

import struct
from pathlib import Path

from laser_ai.ilda.types import Frame, Point, Show


_MAGIC = b"ILDA"
_HEADER_SIZE = 32


def read_ilda(path: str | Path) -> Show:
    """Parse an ILDA file at `path` into a Show."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ILDA file not found: {path}")
    data = path.read_bytes()
    return _parse(data, name=path.stem)


def _parse(data: bytes, name: str = "") -> Show:
    show = Show(name=name)
    offset = 0
    while offset + _HEADER_SIZE <= len(data):
        section_magic = data[offset:offset + 4]
        if section_magic != _MAGIC:
            raise ValueError(f"expected ILDA magic at offset {offset}, got {section_magic!r}")
        (format_code,) = struct.unpack(">I", data[offset + 4:offset + 8])
        frame_name = data[offset + 8:offset + 16].rstrip(b" \x00").decode("ascii", errors="replace")
        company = data[offset + 16:offset + 24].rstrip(b" \x00").decode("ascii", errors="replace")
        records, frame_no, total, _projector, _future = struct.unpack(
            ">HHHHH", data[offset + 24:offset + 34]
        )
        offset += _HEADER_SIZE + 2  # header is 32 bytes + 2 reserved = 34; spec has 32-byte header but many files pad to 34. Guard:
        # Actually the spec header is exactly 32 bytes. The 5 unsigned-shorts are 10 bytes, starting at offset 24,
        # which means offset+24..offset+34 is correct, but the header itself is 34 bytes. Some implementations treat
        # it as 32 — re-read below with 32-byte assumption.
        offset -= 2
        offset += 32  # keep header size 32

        if records == 0:
            # End-of-file marker
            break

        frame_points: list[Point] = []
        for i in range(records):
            if format_code == 4:
                rec = data[offset:offset + 10]
                if len(rec) < 10:
                    raise ValueError(f"truncated format-4 record at offset {offset}")
                x, y, z, status, b, g, r = struct.unpack(">hhhBBBB", rec)
                offset += 10
            elif format_code == 5:
                rec = data[offset:offset + 8]
                if len(rec) < 8:
                    raise ValueError(f"truncated format-5 record at offset {offset}")
                x, y, status, b, g, r = struct.unpack(">hhBBBB", rec)
                offset += 8
            elif format_code == 0:
                rec = data[offset:offset + 8]
                if len(rec) < 8:
                    raise ValueError(f"truncated format-0 record at offset {offset}")
                x, y, z, status, _color_idx = struct.unpack(">hhhBB", rec)
                offset += 8
                r = g = b = 255  # indexed; treat as white
            elif format_code == 1:
                rec = data[offset:offset + 6]
                if len(rec) < 6:
                    raise ValueError(f"truncated format-1 record at offset {offset}")
                x, y, status, _color_idx = struct.unpack(">hhBB", rec)
                offset += 6
                r = g = b = 255
            else:
                raise ValueError(f"unsupported ILDA format code {format_code}")

            is_blank = bool(status & 0b01000000)
            is_last = bool(status & 0b10000000)
            frame_points.append(Point(
                x=x, y=y, r=r, g=g, b=b,
                is_blank=is_blank, is_last_point=is_last,
            ))

        show.frames.append(Frame(
            points=frame_points, name=frame_name, company=company,
            frame_index=frame_no, total_frames=total,
        ))
    return show
```

Note on header size: the fixture and most production ILDA files use a 32-byte header. Record sizes (bytes per point) per format:
- Format 0 (3D indexed): 8 bytes (x, y, z int16; status u8; color_index u8)
- Format 1 (2D indexed): 6 bytes
- Format 4 (3D RGB): 10 bytes (x, y, z int16; status u8; B, G, R u8)
- Format 5 (2D RGB): 8 bytes

- [ ] **Step 6: Run reader tests — they should pass**

```bash
pytest tests/ilda/test_reader.py -v
```

Expected: all 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add laser_ai/ilda/reader.py tests/ilda/test_reader.py tests/fixtures/
git commit -m "feat(ilda): add ILDA reader for formats 0/1/4/5"
```

---

## Task 4: ILDA writer

**Files:**
- Create: `laser_ai/ilda/writer.py`
- Create: `tests/ilda/test_writer.py`
- Create: `tests/ilda/test_roundtrip.py`

We only write Format 4 (3D RGB). It's the most widely supported format and the Helios DAC handles it natively via its driver stack.

- [ ] **Step 1: Write `tests/ilda/test_writer.py`**

```python
"""Tests for the ILDA writer."""
from __future__ import annotations

import struct
from pathlib import Path

from laser_ai.ilda.types import Frame, Point, Show
from laser_ai.ilda.writer import write_ilda


def test_write_produces_ilda_magic(tmp_path: Path):
    show = Show(frames=[Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)])])
    out = tmp_path / "test.ilda"
    write_ilda(show, out)
    data = out.read_bytes()
    assert data[:4] == b"ILDA"


def test_write_records_format_4(tmp_path: Path):
    show = Show(frames=[Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)])])
    out = tmp_path / "test.ilda"
    write_ilda(show, out)
    data = out.read_bytes()
    (fmt,) = struct.unpack(">I", data[4:8])
    assert fmt == 4


def test_write_ends_with_zero_record_section(tmp_path: Path):
    show = Show(frames=[Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)])])
    out = tmp_path / "test.ilda"
    write_ilda(show, out)
    data = out.read_bytes()
    # Last 32 bytes should be a header with records=0
    tail = data[-32:]
    assert tail[:4] == b"ILDA"
    records = struct.unpack(">H", tail[24:26])[0]
    assert records == 0
```

- [ ] **Step 2: Run — should fail with ImportError**

```bash
pytest tests/ilda/test_writer.py -v
```

- [ ] **Step 3: Write `laser_ai/ilda/writer.py`**

```python
"""Write Show objects as ILDA Format 4 files."""
from __future__ import annotations

import struct
from pathlib import Path

from laser_ai.ilda.types import Show


_MAGIC = b"ILDA"
_FORMAT = 4


def write_ilda(show: Show, path: str | Path) -> None:
    """Write a Show as an ILDA Format 4 (3D RGB) file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(show.frames)
    buf = bytearray()
    company = (show.company or "laser-ai").encode("ascii", errors="replace")[:8].ljust(8, b" ")

    for idx, frame in enumerate(show.frames):
        name = (frame.name or f"f{idx:04d}").encode("ascii", errors="replace")[:8].ljust(8, b" ")
        records = len(frame.points)

        # 32-byte header
        buf += _MAGIC
        buf += struct.pack(">I", _FORMAT)
        buf += name
        buf += company
        buf += struct.pack(">HHHHH", records, idx, total, 0, 0)

        for i, p in enumerate(frame.points):
            status = 0
            if p.is_blank:
                status |= 0b01000000
            if i == records - 1 or p.is_last_point:
                status |= 0b10000000
            buf += struct.pack(
                ">hhhBBBB",
                int(p.x), int(p.y), 0,  # z always 0 for 2D shows
                status,
                int(p.b), int(p.g), int(p.r),  # ILDA spec: B, G, R order
            )

    # End-of-file section
    buf += _MAGIC
    buf += struct.pack(">I", _FORMAT)
    buf += b" " * 8
    buf += company
    buf += struct.pack(">HHHHH", 0, total, total, 0, 0)

    path.write_bytes(bytes(buf))
```

- [ ] **Step 4: Run writer tests — should pass**

```bash
pytest tests/ilda/test_writer.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Write `tests/ilda/test_roundtrip.py`**

```python
"""Round-trip test: write a Show, read it back, compare."""
from __future__ import annotations

from pathlib import Path

from laser_ai.ilda.reader import read_ilda
from laser_ai.ilda.types import Frame, Point, Show
from laser_ai.ilda.writer import write_ilda


def test_roundtrip_preserves_frame_count(tmp_path: Path):
    show = Show(frames=[
        Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)]),
        Frame(points=[Point(1000, -1000, 0, 255, 0, is_last_point=True)]),
        Frame(points=[Point(-2000, 2000, 0, 0, 255, is_last_point=True)]),
    ])
    path = tmp_path / "r.ilda"
    write_ilda(show, path)
    recovered = read_ilda(path)
    assert len(recovered.frames) == 3


def test_roundtrip_preserves_point_colors(tmp_path: Path):
    show = Show(frames=[Frame(points=[
        Point(0, 0, 200, 150, 75, is_last_point=True),
    ])])
    path = tmp_path / "r.ilda"
    write_ilda(show, path)
    recovered = read_ilda(path)
    p = recovered.frames[0].points[0]
    assert (p.r, p.g, p.b) == (200, 150, 75)


def test_roundtrip_preserves_blanking(tmp_path: Path):
    show = Show(frames=[Frame(points=[
        Point(0, 0, 255, 0, 0),
        Point(1000, 1000, 0, 0, 0, is_blank=True),
        Point(2000, 2000, 0, 255, 0, is_last_point=True),
    ])])
    path = tmp_path / "r.ilda"
    write_ilda(show, path)
    recovered = read_ilda(path)
    pts = recovered.frames[0].points
    assert len(pts) == 3
    assert not pts[0].is_blank
    assert pts[1].is_blank
    assert not pts[2].is_blank


def test_roundtrip_on_existing_fixture(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "tiny_show.ilda"
    show = read_ilda(fixture)
    out = tmp_path / "rt.ilda"
    write_ilda(show, out)
    recovered = read_ilda(out)

    assert len(recovered.frames) == len(show.frames)
    for orig, rec in zip(show.frames, recovered.frames):
        assert len(orig.points) == len(rec.points)
        for po, pr in zip(orig.points, rec.points):
            assert po.x == pr.x
            assert po.y == pr.y
            assert po.r == pr.r
            assert po.g == pr.g
            assert po.b == pr.b
            assert po.is_blank == pr.is_blank
```

- [ ] **Step 6: Run roundtrip tests — should pass**

```bash
pytest tests/ilda/test_roundtrip.py -v
```

Expected: all 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add laser_ai/ilda/writer.py tests/ilda/test_writer.py tests/ilda/test_roundtrip.py
git commit -m "feat(ilda): add ILDA writer + reader/writer round-trip tests"
```

---

## Task 5: Arc-length resampler

**Files:**
- Create: `laser_ai/ilda/resample.py`
- Create: `tests/ilda/test_resample.py`

Resamples any frame to exactly N points via even arc-length spacing along the drawn (non-blanked) path. Blanked segments are preserved — pen-up moves become zero-brightness points at the same resampled positions. This gives us the fixed `(N, 6)` tensor the model will eat in plan 2.

- [ ] **Step 1: Write `tests/ilda/test_resample.py`**

```python
"""Tests for arc-length frame resampling."""
from __future__ import annotations

import numpy as np

from laser_ai.ilda.resample import resample_frame
from laser_ai.ilda.types import Frame, Point


def _line_frame(n: int = 10) -> Frame:
    """A simple horizontal line from (-10000, 0) to (10000, 0)."""
    return Frame(points=[
        Point(x=-10000 + int(20000 * i / (n - 1)), y=0, r=255, g=255, b=255,
              is_last_point=(i == n - 1))
        for i in range(n)
    ])


def test_resample_returns_exact_n_points():
    frame = _line_frame(10)
    out = resample_frame(frame, n=512)
    assert len(out.points) == 512


def test_resample_preserves_start_and_end_position():
    frame = _line_frame(10)
    out = resample_frame(frame, n=512)
    # Start close to (-10000, 0)
    assert abs(out.points[0].x - (-10000)) < 100
    assert abs(out.points[0].y) < 100
    # End close to (10000, 0)
    assert abs(out.points[-1].x - 10000) < 100
    assert abs(out.points[-1].y) < 100


def test_resample_preserves_color():
    frame = _line_frame(10)
    out = resample_frame(frame, n=512)
    for p in out.points:
        assert (p.r, p.g, p.b) == (255, 255, 255)


def test_resample_flags_last_point():
    frame = _line_frame(10)
    out = resample_frame(frame, n=512)
    assert out.points[-1].is_last_point
    assert not any(p.is_last_point for p in out.points[:-1])


def test_resample_handles_single_point_frame():
    frame = Frame(points=[Point(0, 0, 255, 0, 0, is_last_point=True)])
    out = resample_frame(frame, n=64)
    assert len(out.points) == 64
    for p in out.points:
        assert p.x == 0 and p.y == 0


def test_resample_handles_empty_frame():
    out = resample_frame(Frame(points=[]), n=64)
    assert len(out.points) == 64
    for p in out.points:
        assert p.is_blank


def test_resample_arc_length_spacing_is_uniform():
    """Points along a straight line should be roughly evenly spaced after resampling."""
    frame = _line_frame(10)
    out = resample_frame(frame, n=100)
    xs = np.array([p.x for p in out.points], dtype=np.float64)
    diffs = np.diff(xs)
    # All spacings should be within 5% of the mean
    assert np.std(diffs) / np.mean(diffs) < 0.05
```

- [ ] **Step 2: Run — should fail with ImportError**

```bash
pytest tests/ilda/test_resample.py -v
```

- [ ] **Step 3: Write `laser_ai/ilda/resample.py`**

```python
"""Even-arc-length resampling of ILDA frames to a fixed point count."""
from __future__ import annotations

import numpy as np

from laser_ai.ilda.types import Frame, Point


def resample_frame(frame: Frame, n: int = 512) -> Frame:
    """Resample `frame` to exactly `n` points via uniform arc-length spacing.

    - Empty frames become `n` blanked points at origin.
    - Single-point frames become `n` copies of that point.
    - Blanking/color of each resampled point is taken from the nearest original point.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    src = frame.points
    if len(src) == 0:
        return Frame(points=[
            Point(x=0, y=0, r=0, g=0, b=0, is_blank=True, is_last_point=(i == n - 1))
            for i in range(n)
        ], name=frame.name, company=frame.company,
           frame_index=frame.frame_index, total_frames=frame.total_frames)

    if len(src) == 1:
        p = src[0]
        return Frame(points=[
            Point(x=p.x, y=p.y, r=p.r, g=p.g, b=p.b, is_blank=p.is_blank,
                  is_last_point=(i == n - 1))
            for i in range(n)
        ], name=frame.name, company=frame.company,
           frame_index=frame.frame_index, total_frames=frame.total_frames)

    xs = np.array([p.x for p in src], dtype=np.float64)
    ys = np.array([p.y for p in src], dtype=np.float64)
    seg_lens = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = cum[-1]

    if total_len <= 0.0:
        # All points at same location
        p = src[0]
        return Frame(points=[
            Point(x=p.x, y=p.y, r=p.r, g=p.g, b=p.b, is_blank=p.is_blank,
                  is_last_point=(i == n - 1))
            for i in range(n)
        ], name=frame.name, company=frame.company,
           frame_index=frame.frame_index, total_frames=frame.total_frames)

    targets = np.linspace(0.0, total_len, n)
    indices = np.searchsorted(cum, targets, side="right") - 1
    indices = np.clip(indices, 0, len(src) - 2)

    resampled_pts: list[Point] = []
    for i, t in enumerate(targets):
        idx = int(indices[i])
        seg_start = cum[idx]
        seg_len = cum[idx + 1] - cum[idx]
        frac = 0.0 if seg_len <= 0.0 else (t - seg_start) / seg_len
        frac = float(np.clip(frac, 0.0, 1.0))

        a = src[idx]
        b = src[idx + 1]
        x = a.x + frac * (b.x - a.x)
        y = a.y + frac * (b.y - a.y)

        # Take attributes from whichever endpoint we're closer to
        src_pt = a if frac < 0.5 else b
        resampled_pts.append(Point(
            x=int(round(x)),
            y=int(round(y)),
            r=src_pt.r, g=src_pt.g, b=src_pt.b,
            is_blank=src_pt.is_blank,
            is_last_point=(i == n - 1),
        ))

    return Frame(points=resampled_pts, name=frame.name, company=frame.company,
                 frame_index=frame.frame_index, total_frames=frame.total_frames)
```

- [ ] **Step 4: Run resample tests — should pass**

```bash
pytest tests/ilda/test_resample.py -v
```

Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add laser_ai/ilda/resample.py tests/ilda/test_resample.py
git commit -m "feat(ilda): arc-length resampler to fixed point count"
```

---

## Task 6: Audio loader and features

**Files:**
- Create: `laser_ai/audio/__init__.py`
- Create: `laser_ai/audio/loader.py`
- Create: `laser_ai/audio/features.py`
- Create: `tests/audio/__init__.py`
- Create: `tests/audio/test_loader.py`
- Create: `tests/audio/test_features.py`

Librosa-only features for the foundation (no CLAP yet — that lands in plan 2 when the real model needs it). Features are per-frame at 30 fps matching the laser frame rate from the spec.

Per-frame feature vector (foundation version, ~150 dims):
- 128 log-mel bins
- RMS energy
- Spectral centroid / rolloff / flatness
- 12 chroma bins
- Beat phase (0–1)
- Onset strength

- [ ] **Step 1: Write `laser_ai/audio/__init__.py`**

```python
"""Audio loading and feature extraction."""

from laser_ai.audio.features import FEATURE_DIM, extract_features
from laser_ai.audio.loader import load_audio

__all__ = ["FEATURE_DIM", "extract_features", "load_audio"]
```

- [ ] **Step 2: Write `tests/audio/__init__.py`** (empty)

- [ ] **Step 3: Write `tests/audio/test_loader.py`**

```python
"""Tests for audio loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from laser_ai.audio.loader import load_audio


def test_load_returns_mono_float32(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert sr == 44100


def test_load_preserves_duration(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    # Fixture is 1-second 44.1k
    assert abs(len(samples) - sr) < 10


def test_load_missing_file_raises(tmp_path: Path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nope.wav")
```

- [ ] **Step 4: Write `laser_ai/audio/loader.py`**

```python
"""Load audio files to mono float32 at 44.1 kHz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SR = 44100


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as (samples, sr). Samples are mono float32.

    Resampled to 44.1 kHz if the source differs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Mix to mono
    samples = samples.mean(axis=1)

    if sr != TARGET_SR:
        # lazy import; librosa is heavy
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return samples.astype(np.float32, copy=False), sr
```

- [ ] **Step 5: Run loader tests — should pass**

```bash
pytest tests/audio/test_loader.py -v
```

Expected: all 3 PASS.

- [ ] **Step 6: Write `tests/audio/test_features.py`**

```python
"""Tests for audio feature extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from laser_ai.audio.features import FEATURE_DIM, extract_features
from laser_ai.audio.loader import load_audio


def test_extract_returns_2d_array(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats = extract_features(samples, sr, fps=30.0)
    assert feats.ndim == 2


def test_extract_matches_expected_frame_count(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats = extract_features(samples, sr, fps=30.0)
    # 1-second audio at 30 fps → 30 frames (± 1 for boundary rounding)
    assert abs(feats.shape[0] - 30) <= 1


def test_extract_feature_dim_matches_constant(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats = extract_features(samples, sr, fps=30.0)
    assert feats.shape[1] == FEATURE_DIM


def test_extract_is_finite(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats = extract_features(samples, sr, fps=30.0)
    assert np.all(np.isfinite(feats))


def test_extract_is_deterministic(tiny_wav_path: Path):
    samples, sr = load_audio(tiny_wav_path)
    feats1 = extract_features(samples, sr, fps=30.0)
    feats2 = extract_features(samples, sr, fps=30.0)
    np.testing.assert_allclose(feats1, feats2)
```

- [ ] **Step 7: Write `laser_ai/audio/features.py`**

```python
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
```

- [ ] **Step 8: Run feature tests — should pass**

```bash
pytest tests/audio/test_features.py -v
```

Expected: all 5 PASS.

- [ ] **Step 9: Commit**

```bash
git add laser_ai/audio tests/audio
git commit -m "feat(audio): loader + per-frame librosa feature extraction"
```

---

## Task 7: Dataset discovery

**Files:**
- Create: `laser_ai/dataset/__init__.py`
- Create: `laser_ai/dataset/pair.py`
- Create: `laser_ai/dataset/discovery.py`
- Create: `tests/dataset/__init__.py`
- Create: `tests/dataset/test_discovery.py`

Walk a folder, auto-pair `.mp3`/`.wav` with `.ild`/`.ilda` by filename stem. Return unmatched files separately so the GUI (in plan 3) can surface them.

- [ ] **Step 1: Write `laser_ai/dataset/__init__.py`**

```python
"""Training dataset discovery and pair management."""

from laser_ai.dataset.discovery import DiscoveryResult, discover_pairs
from laser_ai.dataset.pair import AudioLaserPair

__all__ = ["AudioLaserPair", "DiscoveryResult", "discover_pairs"]
```

- [ ] **Step 2: Write `laser_ai/dataset/pair.py`**

```python
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
```

- [ ] **Step 3: Write `tests/dataset/__init__.py`** (empty)

- [ ] **Step 4: Write `tests/dataset/test_discovery.py`**

```python
"""Tests for auto-pairing of audio + ILDA files."""
from __future__ import annotations

from pathlib import Path

from laser_ai.dataset.discovery import discover_pairs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_auto_pairs_by_stem(tmp_path: Path):
    _touch(tmp_path / "song1.mp3")
    _touch(tmp_path / "song1.ilda")
    _touch(tmp_path / "song2.wav")
    _touch(tmp_path / "song2.ild")

    result = discover_pairs(tmp_path)
    stems = sorted(p.stem for p in result.pairs)
    assert stems == ["song1", "song2"]


def test_unmatched_files_are_reported(tmp_path: Path):
    _touch(tmp_path / "song1.mp3")
    _touch(tmp_path / "song1.ilda")
    _touch(tmp_path / "lonely_audio.mp3")
    _touch(tmp_path / "lonely_ilda.ilda")

    result = discover_pairs(tmp_path)
    assert len(result.pairs) == 1
    assert any(p.name == "lonely_audio.mp3" for p in result.unmatched_audio)
    assert any(p.name == "lonely_ilda.ilda" for p in result.unmatched_ilda)


def test_case_insensitive_pairing(tmp_path: Path):
    _touch(tmp_path / "SONG.MP3")
    _touch(tmp_path / "song.ilda")

    result = discover_pairs(tmp_path)
    assert len(result.pairs) == 1


def test_empty_folder_yields_empty_result(tmp_path: Path):
    result = discover_pairs(tmp_path)
    assert result.pairs == []
    assert result.unmatched_audio == []
    assert result.unmatched_ilda == []


def test_missing_folder_raises(tmp_path: Path):
    import pytest
    with pytest.raises(FileNotFoundError):
        discover_pairs(tmp_path / "does_not_exist")
```

- [ ] **Step 5: Write `laser_ai/dataset/discovery.py`**

```python
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
```

- [ ] **Step 6: Run discovery tests — should pass**

```bash
pytest tests/dataset/test_discovery.py -v
```

Expected: all 5 PASS.

- [ ] **Step 7: Commit**

```bash
git add laser_ai/dataset tests/dataset
git commit -m "feat(dataset): auto-discover audio+ILDA pairs by stem"
```

---

## Task 8: Primitive library

**Files:**
- Create: `laser_ai/generator/__init__.py`
- Create: `laser_ai/generator/primitives.py`
- Create: `tests/generator/__init__.py`
- Create: `tests/generator/test_primitives.py`

A small library of parametric primitives. Each takes a parameter dict and returns an `(N, 6)` float32 array (the same format `Frame.to_array` produces). Primitives are closed-form and fast — the stub generator (next task) composes them into frames. The real VAE in plan 2 can also use them as data augmentation or as an output head constraint, so getting them in now is a win either way.

Four primitives suffice for the foundation: **sine wave, lissajous, circle, grid**.

- [ ] **Step 1: Write `laser_ai/generator/__init__.py`**

```python
"""Laser frame generators."""

from laser_ai.generator.base import Generator
from laser_ai.generator.stub import StubGenerator

__all__ = ["Generator", "StubGenerator"]
```

- [ ] **Step 2: Write `tests/generator/__init__.py`** (empty)

- [ ] **Step 3: Write `tests/generator/test_primitives.py`**

```python
"""Tests for primitive shape generators."""
from __future__ import annotations

import numpy as np

from laser_ai.generator.primitives import circle, grid, lissajous, sine_wave


def test_sine_wave_returns_n_by_6():
    arr = sine_wave(n=512, amplitude=0.5, frequency=2.0, phase=0.0, color=(1.0, 0.0, 0.0))
    assert arr.shape == (512, 6)
    assert arr.dtype == np.float32


def test_sine_wave_respects_amplitude():
    arr = sine_wave(n=256, amplitude=0.3, frequency=1.0, phase=0.0, color=(1.0, 1.0, 1.0))
    assert arr[:, 1].max() <= 0.31
    assert arr[:, 1].min() >= -0.31


def test_lissajous_returns_n_by_6_in_range():
    arr = lissajous(n=512, a=3, b=2, delta=np.pi / 2, color=(0.0, 1.0, 0.0))
    assert arr.shape == (512, 6)
    assert np.all(arr[:, 0] >= -1.0) and np.all(arr[:, 0] <= 1.0)
    assert np.all(arr[:, 1] >= -1.0) and np.all(arr[:, 1] <= 1.0)


def test_circle_is_closed_loop():
    arr = circle(n=512, radius=0.5, color=(0.0, 0.0, 1.0))
    # First and last point should coincide (closed loop)
    assert np.allclose(arr[0, :2], arr[-1, :2], atol=0.01)


def test_grid_produces_horizontal_lines_with_blanks():
    arr = grid(n=512, rows=3, cols=3, color=(1.0, 1.0, 0.0))
    # Must have some blanked travel points between grid lines
    assert arr[:, 5].sum() > 0
    assert arr.shape == (512, 6)


def test_all_primitives_have_color_set():
    for prim_fn, kwargs in [
        (sine_wave, dict(amplitude=0.5, frequency=1.0, phase=0.0)),
        (lissajous, dict(a=2, b=3, delta=0.0)),
        (circle, dict(radius=0.8)),
        (grid, dict(rows=4, cols=4)),
    ]:
        arr = prim_fn(n=256, color=(0.7, 0.2, 0.4), **kwargs)
        # at least one point should be non-blank and have the target color
        visible = arr[arr[:, 5] < 0.5]
        if len(visible) > 0:
            assert np.allclose(visible[0, 2:5], [0.7, 0.2, 0.4], atol=1e-3)
```

- [ ] **Step 4: Write `laser_ai/generator/primitives.py`**

```python
"""Parametric primitive shapes. Each returns an (N, 6) float32 array.

Format: (x, y, r, g, b, is_blank) with x,y in [-1, 1] and r,g,b in [0, 1].
"""
from __future__ import annotations

import numpy as np


def sine_wave(
    *,
    n: int,
    amplitude: float,
    frequency: float,
    phase: float,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Horizontal sine across x=[-1, 1]."""
    t = np.linspace(-1.0, 1.0, n)
    x = t
    y = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return _assemble(x, y, color, blank=np.zeros(n, dtype=np.float32))


def lissajous(
    *,
    n: int,
    a: float,
    b: float,
    delta: float,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Classic lissajous: x=sin(a*t+delta), y=sin(b*t), t in [0, 2π]."""
    t = np.linspace(0.0, 2 * np.pi, n)
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    return _assemble(x, y, color, blank=np.zeros(n, dtype=np.float32))


def circle(
    *,
    n: int,
    radius: float,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Closed circle at origin."""
    t = np.linspace(0.0, 2 * np.pi, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    # Close the loop exactly
    x[-1] = x[0]
    y[-1] = y[0]
    return _assemble(x, y, color, blank=np.zeros(n, dtype=np.float32))


def grid(
    *,
    n: int,
    rows: int,
    cols: int,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Horizontal + vertical grid lines with blanked transits between strokes."""
    # Build a polyline of rows horizontal lines then cols vertical lines,
    # with blanked moves between them.
    segs_x: list[np.ndarray] = []
    segs_y: list[np.ndarray] = []
    is_blank_chunks: list[np.ndarray] = []

    def add(xs: np.ndarray, ys: np.ndarray, blanks: np.ndarray) -> None:
        segs_x.append(xs)
        segs_y.append(ys)
        is_blank_chunks.append(blanks)

    row_ys = np.linspace(-0.9, 0.9, rows)
    col_xs = np.linspace(-0.9, 0.9, cols)

    # Budget: split n evenly across strokes. Each stroke = k visible pts + 2 blank transit pts.
    total_strokes = rows + cols
    per_stroke = max(8, n // (total_strokes + 1))  # +1 for end padding

    for yv in row_ys:
        xs = np.linspace(-0.9, 0.9, per_stroke)
        ys = np.full(per_stroke, yv, dtype=np.float32)
        add(xs, ys, np.zeros(per_stroke, dtype=np.float32))
        # blank transit (2 pts)
        add(xs[-1:].copy(), ys[-1:].copy(), np.array([1.0], dtype=np.float32))

    for xv in col_xs:
        ys = np.linspace(-0.9, 0.9, per_stroke)
        xs = np.full(per_stroke, xv, dtype=np.float32)
        add(xs, ys, np.zeros(per_stroke, dtype=np.float32))
        add(xs[-1:].copy(), ys[-1:].copy(), np.array([1.0], dtype=np.float32))

    x = np.concatenate(segs_x)
    y = np.concatenate(segs_y)
    blank = np.concatenate(is_blank_chunks)

    # Pad or truncate to exactly n
    if len(x) > n:
        x = x[:n]; y = y[:n]; blank = blank[:n]
    elif len(x) < n:
        pad = n - len(x)
        x = np.concatenate([x, np.full(pad, x[-1])])
        y = np.concatenate([y, np.full(pad, y[-1])])
        blank = np.concatenate([blank, np.ones(pad, dtype=np.float32)])

    return _assemble(x, y, color, blank=blank)


def _assemble(
    x: np.ndarray,
    y: np.ndarray,
    color: tuple[float, float, float],
    blank: np.ndarray,
) -> np.ndarray:
    n = len(x)
    arr = np.zeros((n, 6), dtype=np.float32)
    arr[:, 0] = np.clip(x, -1.0, 1.0)
    arr[:, 1] = np.clip(y, -1.0, 1.0)
    arr[:, 2] = color[0]
    arr[:, 3] = color[1]
    arr[:, 4] = color[2]
    arr[:, 5] = blank
    # Blanked points rendered as zero-brightness too, but color kept for future style info
    arr[blank >= 0.5, 2:5] = 0.0
    return arr
```

- [ ] **Step 5: Run primitive tests — should pass**

```bash
pytest tests/generator/test_primitives.py -v
```

Expected: all 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add laser_ai/generator/__init__.py laser_ai/generator/primitives.py tests/generator
git commit -m "feat(generator): parametric primitives (sine, lissajous, circle, grid)"
```

---

## Task 9: Stub generator

**Files:**
- Create: `laser_ai/generator/base.py`
- Create: `laser_ai/generator/stub.py`
- Create: `tests/generator/test_stub.py`

Defines the `Generator` protocol (so the real VAE+Sequencer in plan 2 slots in without touching the pipeline) and implements a rule-based stub:

- **Primitive selection:** pick among `circle`, `lissajous`, `sine_wave`, `grid` based on spectral flatness and onset strength.
- **Modulation:** scale by RMS energy; rotate by beat phase; hue by chroma argmax.

It's not going to win awards but it demonstrably reacts to music and is a great integration-test substrate.

- [ ] **Step 1: Write `laser_ai/generator/base.py`**

```python
"""Generator protocol — contract for anything that maps audio features to laser frames."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Generator(Protocol):
    """A Generator turns a (T, FEATURE_DIM) feature stream into a (T, N, 6) frame stream."""

    def generate(self, features: np.ndarray, *, n_points: int = 512) -> np.ndarray:
        """Return (T, n_points, 6) frames for T feature rows."""
        ...
```

- [ ] **Step 2: Write `tests/generator/test_stub.py`**

```python
"""Tests for the rule-based stub generator."""
from __future__ import annotations

import numpy as np

from laser_ai.audio.features import FEATURE_DIM
from laser_ai.generator.stub import StubGenerator


def _random_features(T: int = 60, rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.random((T, FEATURE_DIM), dtype=np.float32)


def test_stub_output_shape():
    feats = _random_features(T=60)
    gen = StubGenerator()
    out = gen.generate(feats, n_points=512)
    assert out.shape == (60, 512, 6)


def test_stub_output_is_in_valid_ranges():
    feats = _random_features(T=30)
    gen = StubGenerator()
    out = gen.generate(feats, n_points=256)
    # x, y in [-1, 1]
    assert out[..., 0].min() >= -1.0 - 1e-6
    assert out[..., 0].max() <= 1.0 + 1e-6
    assert out[..., 1].min() >= -1.0 - 1e-6
    assert out[..., 1].max() <= 1.0 + 1e-6
    # colors in [0, 1]
    assert out[..., 2:5].min() >= 0.0 - 1e-6
    assert out[..., 2:5].max() <= 1.0 + 1e-6
    # is_blank in {0, 1}
    assert set(np.unique(out[..., 5]).tolist()).issubset({0.0, 1.0})


def test_stub_is_deterministic_with_fixed_seed():
    feats = _random_features(T=30)
    gen1 = StubGenerator(seed=42)
    gen2 = StubGenerator(seed=42)
    np.testing.assert_allclose(gen1.generate(feats), gen2.generate(feats))


def test_stub_reacts_to_energy_difference():
    feats_quiet = np.zeros((30, FEATURE_DIM), dtype=np.float32)
    feats_loud = np.zeros((30, FEATURE_DIM), dtype=np.float32)
    # set RMS feature (second-to-last before onset+beat_phase)
    rms_idx = 128 + 12 + 3
    feats_loud[:, rms_idx] = 1.0
    gen = StubGenerator(seed=0)
    quiet = gen.generate(feats_quiet)
    loud = gen.generate(feats_loud)
    # Loud frames should have larger shapes (higher x/y span)
    quiet_extent = quiet[..., :2].std()
    loud_extent = loud[..., :2].std()
    assert loud_extent > quiet_extent
```

- [ ] **Step 3: Write `laser_ai/generator/stub.py`**

```python
"""Rule-based stub generator — maps audio features to primitive selections.

Serves as a placeholder for the trained VAE+Sequencer models (arriving in plan 2).
Both implement the same `Generator` protocol so the pipeline code doesn't change.
"""
from __future__ import annotations

import numpy as np

from laser_ai.audio.features import FEATURE_DIM, N_CHROMA, N_MELS
from laser_ai.generator.primitives import circle, grid, lissajous, sine_wave


# Feature layout: [128 mel | 12 chroma | 3 spectral (cent, rolloff, flat) | rms | onset | beat_phase]
_IDX_CHROMA = N_MELS
_IDX_SPEC_CENTROID = N_MELS + N_CHROMA
_IDX_SPEC_ROLLOFF = _IDX_SPEC_CENTROID + 1
_IDX_SPEC_FLATNESS = _IDX_SPEC_CENTROID + 2
_IDX_RMS = N_MELS + N_CHROMA + 3
_IDX_ONSET = _IDX_RMS + 1
_IDX_BEAT_PHASE = _IDX_ONSET + 1


class StubGenerator:
    """Rule-based placeholder Generator.

    Primitive selection:
    - onset > 0.7         → grid burst
    - flatness > 0.5      → lissajous (noisy/percussive)
    - centroid > 0.5      → sine wave (bright)
    - else                → circle

    Modulation:
    - Scale = 0.3 + 0.6 * rms
    - Rotation (applied via lissajous delta or shape phase) = beat_phase * 2π
    - Hue: argmax chroma → 12 evenly spaced hues
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(self, features: np.ndarray, *, n_points: int = 512) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"features must be (T, {FEATURE_DIM}), got {features.shape}"
            )
        T = features.shape[0]
        out = np.zeros((T, n_points, 6), dtype=np.float32)

        for i in range(T):
            f = features[i]
            onset = float(f[_IDX_ONSET])
            flatness = float(f[_IDX_SPEC_FLATNESS])
            centroid = float(f[_IDX_SPEC_CENTROID])
            rms = float(f[_IDX_RMS])
            beat_phase = float(f[_IDX_BEAT_PHASE])
            chroma = f[_IDX_CHROMA:_IDX_CHROMA + N_CHROMA]

            hue_bin = int(np.argmax(chroma)) if chroma.sum() > 0 else 0
            color = _hue_to_rgb(hue_bin / 12.0)

            scale = 0.3 + 0.6 * np.clip(rms, 0.0, 1.0)
            rot_phase = beat_phase * 2 * np.pi

            if onset > 0.7:
                arr = grid(n=n_points, rows=3, cols=3, color=color)
                arr[:, :2] *= scale
            elif flatness > 0.5:
                arr = lissajous(
                    n=n_points, a=3.0, b=2.0, delta=rot_phase, color=color
                )
                arr[:, :2] *= scale
            elif centroid > 0.5:
                arr = sine_wave(
                    n=n_points,
                    amplitude=0.5 * scale,
                    frequency=2.0 + 2.0 * centroid,
                    phase=rot_phase,
                    color=color,
                )
            else:
                arr = circle(n=n_points, radius=scale, color=color)

            # Apply rotation as a rigid transform for variety
            arr[:, :2] = _rotate(arr[:, :2], rot_phase * 0.5)
            arr[:, :2] = np.clip(arr[:, :2], -1.0, 1.0)

            out[i] = arr

        return out


def _hue_to_rgb(h: float) -> tuple[float, float, float]:
    """HSV with S=V=1 to RGB."""
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    q = 1 - f
    t = f
    i = i % 6
    if i == 0: return (1.0, t, 0.0)
    if i == 1: return (q, 1.0, 0.0)
    if i == 2: return (0.0, 1.0, t)
    if i == 3: return (0.0, q, 1.0)
    if i == 4: return (t, 0.0, 1.0)
    return (1.0, 0.0, q)


def _rotate(xy: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    m = np.array([[c, -s], [s, c]], dtype=xy.dtype)
    return xy @ m.T
```

- [ ] **Step 4: Run stub tests — should pass**

```bash
pytest tests/generator/test_stub.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add laser_ai/generator/base.py laser_ai/generator/stub.py tests/generator/test_stub.py
git commit -m "feat(generator): Generator protocol + rule-based stub implementation"
```

---

## Task 10: Safety post-processor

**Files:**
- Create: `laser_ai/safety/__init__.py`
- Create: `laser_ai/safety/postprocess.py`
- Create: `tests/safety/__init__.py`
- Create: `tests/safety/test_postprocess.py`

Per spec §7. Deterministic rules. Input: one `(N, 6)` frame array. Output: an `(M, 6)` array where M ≤ 1200 and everything is DAC-safe.

Foundation version implements: velocity limiting (interpolate on long jumps, insert blanked transits), dwell-point injection at corners and endpoints, color clipping, coord clamping, point-rate cap. Frame-to-frame smoothing lives in the pipeline (task 11), not here.

- [ ] **Step 1: Write `laser_ai/safety/__init__.py`**

```python
"""DAC safety post-processor."""

from laser_ai.safety.postprocess import SafetyConfig, apply_safety

__all__ = ["SafetyConfig", "apply_safety"]
```

- [ ] **Step 2: Write `tests/safety/__init__.py`** (empty)

- [ ] **Step 3: Write `tests/safety/test_postprocess.py`**

```python
"""Tests for the safety post-processor."""
from __future__ import annotations

import numpy as np

from laser_ai.safety.postprocess import SafetyConfig, apply_safety


def test_output_never_exceeds_max_points():
    rng = np.random.default_rng(0)
    arr = rng.random((2000, 6), dtype=np.float32) * 2 - 1  # noisy random
    arr[:, 5] = 0
    out = apply_safety(arr, SafetyConfig(max_points=1200))
    assert len(out) <= 1200


def test_output_coords_are_clamped():
    arr = np.zeros((100, 6), dtype=np.float32)
    arr[:, 0] = 5.0  # out of range
    arr[:, 1] = -5.0
    out = apply_safety(arr, SafetyConfig())
    assert np.all(out[:, 0] <= 1.0) and np.all(out[:, 0] >= -1.0)
    assert np.all(out[:, 1] <= 1.0) and np.all(out[:, 1] >= -1.0)


def test_output_colors_are_clipped():
    arr = np.zeros((100, 6), dtype=np.float32)
    arr[:, 2] = 2.0  # out of range
    arr[:, 3] = -0.5
    out = apply_safety(arr, SafetyConfig())
    assert np.all(out[:, 2:5] <= 1.0) and np.all(out[:, 2:5] >= 0.0)


def test_large_jumps_are_blanked_or_interpolated():
    # Two far-apart visible points
    arr = np.array([
        [-1.0, -1.0, 1.0, 0, 0, 0.0],
        [ 1.0,  1.0, 1.0, 0, 0, 0.0],
    ], dtype=np.float32)
    out = apply_safety(arr, SafetyConfig(max_step=0.1))
    # Output should have more points now (interpolation) and some blanked
    assert len(out) > 2
    assert out[:, 5].sum() > 0


def test_dwell_points_injected_at_endpoint():
    arr = np.array([
        [-1.0, 0.0, 1.0, 0, 0, 0.0],
        [ 0.0, 0.0, 1.0, 0, 0, 0.0],
        [ 1.0, 0.0, 1.0, 0, 0, 0.0],
    ], dtype=np.float32)
    out = apply_safety(arr, SafetyConfig(dwell_count=4, max_step=10.0))
    # Final repeated point should appear >= dwell_count times at the end
    last_xy = out[-1, :2]
    tail_same = 0
    for i in range(len(out) - 1, -1, -1):
        if np.allclose(out[i, :2], last_xy, atol=1e-4):
            tail_same += 1
        else:
            break
    assert tail_same >= 4


def test_safety_is_idempotent_ish():
    arr = np.zeros((100, 6), dtype=np.float32)
    arr[:, 0] = np.linspace(-0.5, 0.5, 100)
    arr[:, 2] = 1.0  # red
    once = apply_safety(arr, SafetyConfig())
    twice = apply_safety(once, SafetyConfig())
    # After two passes with no out-of-range input, length should be stable
    assert abs(len(once) - len(twice)) < 10


def test_empty_input_produces_empty_safe_frame():
    out = apply_safety(np.zeros((0, 6), dtype=np.float32), SafetyConfig())
    # Empty frame passes through (pipeline handles frame-level empty cases)
    assert out.shape[1] == 6
```

- [ ] **Step 4: Write `laser_ai/safety/postprocess.py`**

```python
"""Deterministic DAC-safety post-processor.

Turns whatever the generator emits into a frame that is safe to send to a laser DAC:
- Velocity-limited
- Dwell-padded on endpoints
- Coord-clamped
- Color-clipped
- Point-rate capped
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SafetyConfig:
    """Runtime-tunable safety parameters."""
    max_points: int = 1200           # ~36k pps at 30 fps
    max_step: float = 0.08            # max normalized distance between consecutive points
    dwell_count: int = 4              # repeated points at endpoints
    coord_margin: float = 0.02        # shrink effective XY range by this much
    strength: float = 1.0             # 0.0 = loose, 1.0 = medium, 2.0 = tight


def apply_safety(arr: np.ndarray, cfg: SafetyConfig = SafetyConfig()) -> np.ndarray:
    """Apply all safety rules to a single (N, 6) frame; return (M, 6)."""
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"expected (N, 6) array, got {arr.shape}")
    if arr.shape[0] == 0:
        return arr.copy()

    a = arr.copy()
    # 1. Clip colors
    a[:, 2:5] = np.clip(a[:, 2:5], 0.0, 1.0)
    # 2. Clamp coords
    limit = 1.0 - cfg.coord_margin
    a[:, :2] = np.clip(a[:, :2], -limit, limit)
    # Discretize is_blank to {0, 1}
    a[:, 5] = (a[:, 5] >= 0.5).astype(a.dtype)

    # 3. Velocity limit / blanking insertion (step-scaled by strength)
    effective_max_step = cfg.max_step / max(0.1, cfg.strength)
    a = _velocity_limit(a, max_step=effective_max_step)

    # 4. Endpoint dwell
    a = _add_endpoint_dwell(a, count=cfg.dwell_count)

    # 5. Point-rate cap (downsample)
    if len(a) > cfg.max_points:
        a = _downsample_arc_length(a, target=cfg.max_points)

    return a


def _velocity_limit(a: np.ndarray, *, max_step: float) -> np.ndarray:
    """For consecutive points farther apart than max_step, insert intermediates.

    If the jump is between two visible points, inserted points remain visible
    (interpolation). If either endpoint is blank, inserted points are blank transit.
    """
    if len(a) < 2:
        return a

    out: list[np.ndarray] = [a[0:1]]
    for i in range(1, len(a)):
        prev = a[i - 1]
        cur = a[i]
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        dist = float(np.hypot(dx, dy))
        if dist > max_step:
            steps = int(np.ceil(dist / max_step))
            # interpolate (excluding prev, including cur)
            lerp = np.linspace(0.0, 1.0, steps + 1)[1:, None]
            xy = (1 - lerp) * prev[:2] + lerp * cur[:2]
            # blank transit if either endpoint was blank OR if visual line would cross too far
            is_transit = (prev[5] >= 0.5) or (cur[5] >= 0.5)
            rgb = (prev[2:5] + cur[2:5]) / 2.0  # rough blend
            blank_flag = 1.0 if is_transit else 0.0
            color_fill = np.zeros(3, dtype=a.dtype) if is_transit else rgb
            block = np.zeros((steps, 6), dtype=a.dtype)
            block[:, :2] = xy
            block[:, 2:5] = color_fill
            block[:, 5] = blank_flag
            # Ensure the final inserted point equals cur exactly
            block[-1] = cur
            out.append(block)
        else:
            out.append(cur[None, :])
    return np.concatenate(out, axis=0)


def _add_endpoint_dwell(a: np.ndarray, *, count: int) -> np.ndarray:
    """Append `count` repeated copies of the last point (and prepend for the first)."""
    if len(a) == 0 or count <= 0:
        return a
    head = np.tile(a[0:1], (count - 1, 1))
    tail = np.tile(a[-1:], (count - 1, 1))
    return np.concatenate([head, a, tail], axis=0)


def _downsample_arc_length(a: np.ndarray, *, target: int) -> np.ndarray:
    """Resample to exactly `target` points preserving path shape."""
    if len(a) <= target:
        return a
    xs = a[:, 0]; ys = a[:, 1]
    seg = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = cum[-1]
    if total <= 0:
        return a[:target]
    t = np.linspace(0.0, total, target)
    idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(a) - 2)
    out = np.zeros((target, 6), dtype=a.dtype)
    for i, ti in enumerate(t):
        j = int(idx[i])
        seg_len = cum[j + 1] - cum[j]
        frac = 0.0 if seg_len <= 0 else (ti - cum[j]) / seg_len
        out[i, :2] = (1 - frac) * a[j, :2] + frac * a[j + 1, :2]
        src = a[j] if frac < 0.5 else a[j + 1]
        out[i, 2:6] = src[2:6]
    return out
```

- [ ] **Step 5: Run safety tests — should pass**

```bash
pytest tests/safety/test_postprocess.py -v
```

Expected: all 7 PASS.

- [ ] **Step 6: Commit**

```bash
git add laser_ai/safety tests/safety
git commit -m "feat(safety): DAC-safety post-processor with velocity + dwell + caps"
```

---

## Task 11: Generation pipeline

**Files:**
- Create: `laser_ai/pipeline/__init__.py`
- Create: `laser_ai/pipeline/generate.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_generate.py`

Wires audio → features → generator → safety → `Show`. Uses the `Generator` protocol from task 9; swapping in the real VAE+Sequencer in plan 2 is a one-line change at the call site.

- [ ] **Step 1: Write `laser_ai/pipeline/__init__.py`**

```python
"""End-to-end generation pipelines."""

from laser_ai.pipeline.generate import generate_show_from_audio

__all__ = ["generate_show_from_audio"]
```

- [ ] **Step 2: Write `tests/pipeline/__init__.py`** (empty)

- [ ] **Step 3: Write `tests/pipeline/test_generate.py`**

```python
"""End-to-end pipeline tests."""
from __future__ import annotations

from pathlib import Path

from laser_ai.generator.stub import StubGenerator
from laser_ai.pipeline.generate import generate_show_from_audio
from laser_ai.safety.postprocess import SafetyConfig


def test_generate_returns_show_with_correct_fps_and_frame_count(tiny_wav_path: Path):
    show = generate_show_from_audio(
        audio_path=tiny_wav_path,
        generator=StubGenerator(seed=0),
        fps=30.0,
        safety_cfg=SafetyConfig(),
    )
    # 1-sec audio at 30 fps → 30 frames
    assert abs(len(show.frames) - 30) <= 1
    assert show.fps == 30.0


def test_generate_produces_nonempty_frames(tiny_wav_path: Path):
    show = generate_show_from_audio(
        audio_path=tiny_wav_path,
        generator=StubGenerator(seed=0),
    )
    for f in show.frames:
        assert len(f.points) > 0


def test_generate_writes_valid_ilda_file(tiny_wav_path: Path, tmp_path: Path):
    from laser_ai.ilda.reader import read_ilda
    from laser_ai.ilda.writer import write_ilda

    show = generate_show_from_audio(
        audio_path=tiny_wav_path,
        generator=StubGenerator(seed=0),
    )
    out = tmp_path / "generated.ilda"
    write_ilda(show, out)
    recovered = read_ilda(out)
    assert len(recovered.frames) == len(show.frames)
```

- [ ] **Step 4: Write `laser_ai/pipeline/generate.py`**

```python
"""End-to-end generation: audio file → Show."""
from __future__ import annotations

from pathlib import Path

from laser_ai.audio.features import extract_features
from laser_ai.audio.loader import load_audio
from laser_ai.generator.base import Generator
from laser_ai.generator.stub import StubGenerator
from laser_ai.ilda.types import Frame, Show
from laser_ai.safety.postprocess import SafetyConfig, apply_safety


def generate_show_from_audio(
    *,
    audio_path: str | Path,
    generator: Generator | None = None,
    fps: float = 30.0,
    n_points: int = 512,
    safety_cfg: SafetyConfig | None = None,
    show_name: str = "",
) -> Show:
    """Generate an ILDA Show from an audio file.

    1. Load + resample audio to 44.1k mono float32
    2. Extract per-frame features at `fps`
    3. Run generator on features → (T, N, 6) frames
    4. Apply safety post-processor per frame
    5. Pack into Show via Frame.from_array
    """
    gen = generator if generator is not None else StubGenerator()
    cfg = safety_cfg if safety_cfg is not None else SafetyConfig()

    samples, sr = load_audio(audio_path)
    feats = extract_features(samples, sr, fps=fps)
    raw = gen.generate(feats, n_points=n_points)  # (T, N, 6)

    frames: list[Frame] = []
    for i in range(raw.shape[0]):
        safe = apply_safety(raw[i], cfg)
        frames.append(Frame.from_array(safe, name=f"f{i:04d}"))

    # Fill metadata
    total = len(frames)
    for i, f in enumerate(frames):
        f.frame_index = i
        f.total_frames = total

    return Show(frames=frames, fps=fps, name=show_name or Path(audio_path).stem)
```

- [ ] **Step 5: Run pipeline tests — should pass**

```bash
pytest tests/pipeline/test_generate.py -v
```

Expected: all 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add laser_ai/pipeline tests/pipeline
git commit -m "feat(pipeline): end-to-end audio→Show generation"
```

---

## Task 12: CLI

**Files:**
- Create: `laser_ai/cli.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/cli/test_cli.py`

Two subcommands for the foundation:
- `laser-ai generate SONG -o OUTPUT.ilda` — generate with the stub model
- `laser-ai info FILE.ilda` — print summary (frame count, duration, point stats)

Plan 2 adds `train` (Colab launcher), plan 3 adds `gui`.

- [ ] **Step 1: Write `tests/cli/__init__.py`** (empty)

- [ ] **Step 2: Write `tests/cli/test_cli.py`**

```python
"""CLI smoke tests."""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from laser_ai.cli import cli


def test_cli_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "info" in result.output


def test_generate_command_writes_ilda_file(tiny_wav_path: Path, tmp_path: Path):
    out = tmp_path / "out.ilda"
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", str(tiny_wav_path), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 32  # at least one header


def test_info_command_on_fixture():
    fixture = Path(__file__).parent.parent / "fixtures" / "tiny_show.ilda"
    runner = CliRunner()
    result = runner.invoke(cli, ["info", str(fixture)])
    assert result.exit_code == 0, result.output
    assert "frames" in result.output.lower()
```

- [ ] **Step 3: Write `laser_ai/cli.py`**

```python
"""laser-ai command-line interface."""
from __future__ import annotations

from pathlib import Path

import click

from laser_ai.ilda.reader import read_ilda
from laser_ai.ilda.writer import write_ilda
from laser_ai.pipeline.generate import generate_show_from_audio
from laser_ai.safety.postprocess import SafetyConfig


@click.group()
@click.version_option()
def cli() -> None:
    """laser-ai: AI-driven ILDA laser show generation from audio."""


@cli.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output .ilda path.")
@click.option("--fps", type=float, default=30.0, show_default=True,
              help="Target frame rate.")
@click.option("--n-points", type=int, default=512, show_default=True,
              help="Points per frame (before safety pass).")
@click.option("--safety", type=click.Choice(["loose", "medium", "tight"]),
              default="medium", show_default=True, help="Safety post-processor strength.")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Generator random seed.")
def generate(audio: Path, output: Path, fps: float, n_points: int,
             safety: str, seed: int) -> None:
    """Generate an ILDA show from an audio file using the stub generator."""
    strengths = {"loose": 0.5, "medium": 1.0, "tight": 2.0}
    cfg = SafetyConfig(strength=strengths[safety])

    click.echo(f"loading audio: {audio}")
    click.echo(f"generating at {fps} fps, {n_points} pts/frame, safety={safety}...")

    from laser_ai.generator.stub import StubGenerator
    show = generate_show_from_audio(
        audio_path=audio,
        generator=StubGenerator(seed=seed),
        fps=fps,
        n_points=n_points,
        safety_cfg=cfg,
    )

    write_ilda(show, output)
    click.echo(f"wrote {output} ({len(show.frames)} frames, {show.duration_s:.2f}s)")


@cli.command()
@click.argument("ilda", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(ilda: Path) -> None:
    """Print summary info about an ILDA file."""
    show = read_ilda(ilda)
    total_pts = sum(len(f.points) for f in show.frames)
    click.echo(f"file:      {ilda}")
    click.echo(f"frames:    {len(show.frames)}")
    click.echo(f"duration:  {show.duration_s:.2f}s @ {show.fps} fps")
    click.echo(f"points:    {total_pts} total, avg {total_pts / max(1, len(show.frames)):.1f}/frame")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run CLI tests — should pass**

```bash
pytest tests/cli/test_cli.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: End-to-end smoke check from a shell**

```bash
laser-ai --help
laser-ai generate tests/fixtures/tiny_audio.wav -o /tmp/smoke.ilda
laser-ai info /tmp/smoke.ilda
```

Expected: help text lists `generate` and `info`, generate completes in ~1s with non-zero output, info prints ~30 frames, ~1s duration.

- [ ] **Step 6: Commit**

```bash
git add laser_ai/cli.py tests/cli
git commit -m "feat(cli): laser-ai generate + info commands"
```

---

## Task 13: Full-suite regression + README polish

**Files:**
- Modify: `README.md`

Close out the plan by running the whole test suite and updating the README with the concrete commands that now work.

- [ ] **Step 1: Run full test suite**

```bash
pytest --cov=laser_ai --cov-report=term-missing
```

Expected: all tests PASS. Coverage > 80% on the core modules (`ilda`, `audio`, `safety`, `pipeline`).

- [ ] **Step 2: Update `README.md`**

Replace the "Status" section and add a "Tested on" section:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with foundation status and quickstart"
```

- [ ] **Step 4: Tag the foundation release**

```bash
git tag -a v0.1.0-foundation -m "Foundation: headless end-to-end generation with stub model"
```

---

## Definition of done

- All tests in every module pass (`pytest` green).
- `laser-ai generate <any-mp3> -o out.ilda` produces a file that loads cleanly in LaserShowGen or any standard ILDA-reading software.
- Coverage > 80% on `laser_ai/ilda`, `laser_ai/audio`, `laser_ai/safety`, `laser_ai/pipeline`, `laser_ai/generator`.
- No `TODO`, `FIXME`, or placeholder code in any module.
- The `Generator` protocol is used by the pipeline (not a concrete class), so plan 2 can drop in the real VAE+Sequencer without touching any existing code outside `laser_ai/generator/`.

## Spec coverage map (this plan vs spec §)

| Spec section | Covered in this plan? | Where |
|---|---|---|
| §3 Architecture: inference pipeline shell | ✓ | Task 11, 12 |
| §4 Frame representation (fixed N=512, `(x,y,r,g,b,is_travel)`) | ✓ | Task 2, 5 |
| §5 Audio features (librosa portion) | ✓ | Task 6 |
| §5 CLAP embedding | ✗ — plan 2 | — |
| §6 Models (VAE, Sequencer) | ✗ — plan 2 | `Generator` protocol placeholder here |
| §7 Safety post-processor | ✓ | Task 10 |
| §8 Preview renderer | ✗ — plan 4 | — |
| §9 GUI | ✗ — plan 3 | — |
| §10 Real-time / Helios | ✗ — plan 5 | — |
| §11 Training workflow | ✗ — plan 2 | — |
| §12 Generation workflow | ✓ (CLI version) | Task 12 |
| §13 File layout | ✓ | Task 1 + every task |
| §15 Testing approach | ✓ | Tests in every task |

Everything in the spec is accounted for either in this plan or in an explicitly-deferred future plan.
