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
