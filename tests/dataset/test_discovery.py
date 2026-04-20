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
