"""sweep 결과를 마크다운 리포트로 정리한다 (CLAUDE.md 7장 sweep 명령 사양)."""
from __future__ import annotations

from pathlib import Path

from styleforge.evaluate.metrics import StyleEvalResult
from styleforge.sweep.planner import SweepCombination


def pick_recommended(combinations: list[SweepCombination], eval_results: dict[int, StyleEvalResult]) -> int:
    """스타일 적용도 + 원본 보존도 합이 가장 큰 조합을 권장한다.

    두 지표는 트레이드오프 관계라 진짜 파레토 프론트 분석이 이상적이지만,
    현재는 합산 점수로 근사한다 — E4(docs/experiments.md) 실측 후 다듬을
    대상으로 남겨둔다.
    """
    scored = [
        (combo.index, eval_results[combo.index].style_similarity + eval_results[combo.index].preservation_similarity)
        for combo in combinations
        if combo.index in eval_results
    ]
    if not scored:
        raise ValueError("평가 결과가 없어 권장 조합을 고를 수 없습니다")

    return max(scored, key=lambda pair: pair[1])[0]


def write_sweep_report(
    output_path: Path,
    *,
    image: Path,
    style: str,
    combinations: list[SweepCombination],
    eval_results: dict[int, StyleEvalResult],
    grid_image_path: Path,
    recommended_index: int,
) -> Path:
    lines = [
        "# Sweep 결과",
        "",
        f"- 입력 이미지: `{image}`",
        f"- 스타일: `{style}`",
        f"- 비교 그리드: `{grid_image_path.name}`",
        "",
        "| # | 축 값 | 스타일 적용도 | 원본 보존도 |",
        "|---|---|---:|---:|",
    ]

    for combo in combinations:
        result = eval_results.get(combo.index)
        style_sim = f"{result.style_similarity:.3f}" if result else "-"
        preservation = f"{result.preservation_similarity:.3f}" if result else "-"
        marker = " **★**" if combo.index == recommended_index else ""
        lines.append(f"| {combo.index}{marker} | {combo.label} | {style_sim} | {preservation} |")

    recommended = next(c for c in combinations if c.index == recommended_index)
    lines += [
        "",
        "## 권장 조합",
        "",
        f"`{recommended.label}` "
        f"(denoise={recommended.denoise:.2f}, lora_weight={recommended.lora_weight:.2f}, "
        f"controlnet_strength={recommended.controlnet_strength:.2f})",
        "",
        "스타일 적용도 + 원본 보존도 합이 가장 큰 조합. 두 지표는 트레이드오프 "
        "관계이므로 용도에 따라 표에서 다른 행을 선택해도 된다.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
