"""sweep/grid.py 단위 테스트 — 1축/2축 레이아웃이 조합 수만큼 칸을 만드는지 확인한다."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from styleforge.sweep.grid import _LABEL_H, _PAD, _ROW_LABEL_W, _THUMB, build_grid
from styleforge.sweep.planner import generate_plan, parse_axes


def _make_dummy_images(combos, tmp_path: Path) -> dict[int, Path]:
    paths = {}
    for combo in combos:
        path = tmp_path / f"combo_{combo.index}.png"
        Image.new("RGB", (64, 64), "red").save(path)
        paths[combo.index] = path
    return paths


def test_build_grid_two_axes_matches_expected_size(tmp_path):
    combos = generate_plan(parse_axes("denoise,lora_weight"), steps=3)
    images = _make_dummy_images(combos, tmp_path)

    output_path = build_grid(combos, images, tmp_path / "grid.png")

    with Image.open(output_path) as grid_img:
        cell_w = _THUMB + _PAD
        cell_h = _THUMB + _PAD + _LABEL_H
        assert grid_img.size == (_ROW_LABEL_W + 3 * cell_w, _LABEL_H + 3 * cell_h)


def test_build_grid_one_axis_single_row(tmp_path):
    combos = generate_plan(parse_axes("denoise"), steps=4)
    images = _make_dummy_images(combos, tmp_path)

    output_path = build_grid(combos, images, tmp_path / "grid.png")

    with Image.open(output_path) as grid_img:
        cell_w = _THUMB + _PAD
        cell_h = _THUMB + _PAD + _LABEL_H
        assert grid_img.size == (_ROW_LABEL_W + 4 * cell_w, _LABEL_H + 1 * cell_h)


def test_build_grid_rejects_empty_combinations(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        build_grid([], {}, tmp_path / "grid.png")
