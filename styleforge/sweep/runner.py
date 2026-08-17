"""sweep 명령 실행 흐름 — 조합 생성 -> 순차 실행(동시 1개) -> 비교 그리드 ->
CLIP 기반 평가 -> 마크다운 리포트 -> 권장 조합 콘솔 출력 (CLAUDE.md 7장).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from styleforge.apply.comfy_client import ProgressUpdate
from styleforge.apply.runner import ApplyError, StrengthParams, run_apply_with_params
from styleforge.config import settings
from styleforge.evaluate.metrics import MetricsError, StyleEvalResult, evaluate_style_transfer
from styleforge.evaluate.report import pick_recommended, write_sweep_report
from styleforge.sweep.grid import build_grid
from styleforge.sweep.planner import DEFAULT_AXES, PlanError, generate_plan, parse_axes

_STYLE_REFERENCE_SAMPLE = 5


class SweepError(RuntimeError):
    """sweep 실행 중 발생한 오류."""


@dataclass
class SweepProgress:
    combo_index: int
    total_combos: int
    stage: str  # "applying" | "evaluating"
    step: int | None = None  # ComfyUI KSampler 진행 스텝 (stage="applying" 중에만 채워짐)
    total_steps: int | None = None


def _style_reference_images(style: str) -> list[Path]:
    """학습에 쓰인 전처리 이미지(data/prepared/{style}/*.jpg)에서 일부를 뽑아 화풍 레퍼런스로 쓴다."""
    prepared_dir = settings.data_dir / "prepared" / style
    candidates = sorted(prepared_dir.glob("*.jpg"))
    if not candidates:
        raise SweepError(
            f"화풍 레퍼런스 이미지를 찾지 못했습니다: {prepared_dir} "
            f"(먼저 `styleforge train --name {style} ...`을 실행했어야 합니다)"
        )

    rng = random.Random(0)
    rng.shuffle(candidates)
    return candidates[:_STYLE_REFERENCE_SAMPLE]


def run_sweep(
    *,
    image: Path,
    style: str,
    axes: str = DEFAULT_AXES,
    steps: int = 4,
    workflow_name: str = "style_transfer_lineart",
    on_progress: Callable[[SweepProgress], None] | None = None,
) -> Path:
    """sweep 명령의 실행 흐름. 결과가 저장된 디렉터리 경로를 반환한다."""
    if not image.is_file():
        raise SweepError(f"입력 이미지가 없습니다: {image}")

    try:
        axis_names = parse_axes(axes)
        combinations = generate_plan(axis_names, steps)
    except PlanError as exc:
        raise SweepError(str(exc)) from exc

    style_references = _style_reference_images(style)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/sweeps") / f"{timestamp}_{style}"
    total = len(combinations)

    representative_images: dict[int, Path] = {}
    eval_results: dict[int, StyleEvalResult] = {}

    for combo in combinations:
        if on_progress is not None:
            on_progress(SweepProgress(combo_index=combo.index, total_combos=total, stage="applying"))

        params = StrengthParams(
            denoise=combo.denoise,
            lora_weight=combo.lora_weight,
            controlnet_strength=combo.controlnet_strength,
        )
        combo_dir = output_dir / f"combo_{combo.index:02d}"

        def _on_comfy_progress(update: ProgressUpdate) -> None:
            if on_progress is not None:
                on_progress(
                    SweepProgress(
                        combo_index=combo.index,
                        total_combos=total,
                        stage="applying",
                        step=update.value,
                        total_steps=update.max,
                    )
                )

        try:
            run_apply_with_params(
                image=image,
                style=style,
                params=params,
                workflow_name=workflow_name,
                output_dir=combo_dir,
                extra_meta={"axis_values": combo.axis_values, "combo_index": combo.index},
                on_progress=_on_comfy_progress if on_progress is not None else None,
            )
        except ApplyError as exc:
            raise SweepError(f"조합 #{combo.index}({combo.label}) 실행 실패: {exc}") from exc

        generated_images = sorted(combo_dir.glob("*.png")) + sorted(combo_dir.glob("*.jpg"))
        if not generated_images:
            raise SweepError(f"조합 #{combo.index} 결과 이미지를 찾지 못했습니다: {combo_dir}")
        representative_images[combo.index] = generated_images[0]

        if on_progress is not None:
            on_progress(SweepProgress(combo_index=combo.index, total_combos=total, stage="evaluating"))

        try:
            eval_results[combo.index] = evaluate_style_transfer(
                generated_images[0], image, style_references
            )
        except MetricsError as exc:
            raise SweepError(str(exc)) from exc

    grid_path = build_grid(combinations, representative_images, output_dir / "grid.png")
    recommended_index = pick_recommended(combinations, eval_results)
    write_sweep_report(
        output_dir / "report.md",
        image=image,
        style=style,
        combinations=combinations,
        eval_results=eval_results,
        grid_image_path=grid_path,
        recommended_index=recommended_index,
    )

    recommended = next(c for c in combinations if c.index == recommended_index)
    print(f"[sweep] 권장 조합: {recommended.label}")
    print(
        f"[sweep]   denoise={recommended.denoise:.2f}, lora_weight={recommended.lora_weight:.2f}, "
        f"controlnet_strength={recommended.controlnet_strength:.2f}"
    )

    return output_dir
