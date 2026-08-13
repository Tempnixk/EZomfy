"""결과 비교 그리드 이미지 생성 (축 라벨 포함) — CLAUDE.md 7장 sweep 사양.

planner가 축을 최대 2개로 제한하므로(sweep/planner.py) 1축이면 가로 1줄,
2축이면 axis_a를 행, axis_b를 열로 배치한다.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from styleforge.sweep.planner import SweepCombination

_THUMB = 256
_LABEL_H = 20
_ROW_LABEL_W = 130
_PAD = 8


def _thumbnail(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((_THUMB, _THUMB))
    canvas = Image.new("RGB", (_THUMB, _THUMB), "black")
    offset = ((_THUMB - img.width) // 2, (_THUMB - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def build_grid(
    combinations: list[SweepCombination],
    image_paths: dict[int, Path],
    output_path: Path,
) -> Path:
    """combo.index -> 결과 이미지 경로 매핑을 받아 라벨 붙은 격자 이미지를 만들고 저장한다."""
    if not combinations:
        raise ValueError("빈 조합 목록으로 그리드를 만들 수 없습니다")

    axes = list(combinations[0].axis_values.keys())
    font = ImageFont.load_default()

    if len(axes) == 1:
        (axis,) = axes
        ordered = sorted(combinations, key=lambda c: c.axis_values[axis])
        row_labels = [""]
        col_labels = [f"{axis}={c.axis_values[axis]:.2f}" for c in ordered]
        cell_of = {(0, col): c for col, c in enumerate(ordered)}
    else:
        axis_a, axis_b = axes
        a_values = sorted({c.axis_values[axis_a] for c in combinations})
        b_values = sorted({c.axis_values[axis_b] for c in combinations})
        row_labels = [f"{axis_a}={v:.2f}" for v in a_values]
        col_labels = [f"{axis_b}={v:.2f}" for v in b_values]
        lookup = {(c.axis_values[axis_a], c.axis_values[axis_b]): c for c in combinations}
        cell_of = {
            (row, col): lookup[(a_values[row], b_values[col])]
            for row in range(len(a_values))
            for col in range(len(b_values))
        }

    rows, cols = len(row_labels), len(col_labels)
    cell_w = _THUMB + _PAD
    cell_h = _THUMB + _PAD + _LABEL_H
    grid_w = _ROW_LABEL_W + cols * cell_w
    grid_h = _LABEL_H + rows * cell_h

    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)

    for col, label in enumerate(col_labels):
        draw.text((_ROW_LABEL_W + col * cell_w + _PAD, _PAD // 2), label, fill="black", font=font)

    for row, label in enumerate(row_labels):
        if label:
            draw.text((_PAD, _LABEL_H + row * cell_h + cell_h // 2), label, fill="black", font=font)

    for (row, col), combo in cell_of.items():
        x = _ROW_LABEL_W + col * cell_w
        y = _LABEL_H + row * cell_h + _LABEL_H
        image_path = image_paths.get(combo.index)
        if image_path is not None and image_path.is_file():
            canvas.paste(_thumbnail(image_path), (x, y))
        draw.text((x, y + _THUMB + 2), combo.label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
