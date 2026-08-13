"""파라미터 조합 생성 — denoise x lora_weight x controlnet_strength 그리드.

sweep의 기본 축은 denoise, lora_weight 2개다 (CLAUDE.md 7장). 비교 그리드
이미지(sweep/grid.py)가 2차원 격자이므로 지원 축 수도 최대 2개로 제한한다.
"""
from __future__ import annotations

from dataclasses import dataclass

from styleforge.apply.runner import strength_to_params

# apply/runner.py의 strength_to_params()가 --strength 0~1을 매핑하는
# 구간과 맞춘다. sweep은 이 구간을 직접 촘촘히 탐색해 최적 조합을 찾는다.
AXIS_RANGES: dict[str, tuple[float, float]] = {
    "denoise": (0.35, 0.8),
    "lora_weight": (0.4, 1.0),
    "controlnet_strength": (0.6, 0.95),
}

DEFAULT_AXES = "denoise,lora_weight"
DEFAULT_BASE_STRENGTH = 0.6


class PlanError(RuntimeError):
    """축 파싱 또는 조합 생성 중 발생한 오류."""


@dataclass
class SweepCombination:
    index: int
    axis_values: dict[str, float]
    denoise: float
    lora_weight: float
    controlnet_strength: float

    @property
    def label(self) -> str:
        return ", ".join(f"{axis}={value:.2f}" for axis, value in self.axis_values.items())


def parse_axes(axes: str) -> list[str]:
    """"denoise,lora_weight" 형태의 --axes 값을 검증된 축 이름 목록으로 바꾼다."""
    names = [a.strip() for a in axes.split(",") if a.strip()]
    if not names:
        raise PlanError("--axes가 비어 있습니다")
    if len(names) > 2:
        raise PlanError("비교 그리드 이미지는 2차원이라 축은 최대 2개까지 지원합니다")
    if len(set(names)) != len(names):
        raise PlanError(f"축이 중복되었습니다: {names}")

    unknown = [a for a in names if a not in AXIS_RANGES]
    if unknown:
        raise PlanError(f"알 수 없는 축: {unknown} (사용 가능: {sorted(AXIS_RANGES)})")

    return names


def _linspace(lo: float, hi: float, steps: int) -> list[float]:
    if steps < 2:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]


def generate_plan(
    axes: list[str],
    steps: int = 4,
    *,
    base_strength: float = DEFAULT_BASE_STRENGTH,
) -> list[SweepCombination]:
    """축마다 steps개 값을 뽑아 조합을 만든다. 나머지 축은 base_strength 기본값으로 고정한다."""
    if steps < 2:
        raise PlanError("--steps는 2 이상이어야 합니다")

    base = strength_to_params(base_strength)
    base_values = {
        "denoise": base.denoise,
        "lora_weight": base.lora_weight,
        "controlnet_strength": base.controlnet_strength,
    }

    axis_grids = {axis: _linspace(*AXIS_RANGES[axis], steps) for axis in axes}

    combinations: list[SweepCombination] = []

    if len(axes) == 1:
        (axis,) = axes
        for value in axis_grids[axis]:
            values = {**base_values, axis: value}
            combinations.append(
                SweepCombination(
                    index=len(combinations),
                    axis_values={axis: value},
                    denoise=values["denoise"],
                    lora_weight=values["lora_weight"],
                    controlnet_strength=values["controlnet_strength"],
                )
            )
    else:
        axis_a, axis_b = axes
        for value_a in axis_grids[axis_a]:
            for value_b in axis_grids[axis_b]:
                values = {**base_values, axis_a: value_a, axis_b: value_b}
                combinations.append(
                    SweepCombination(
                        index=len(combinations),
                        axis_values={axis_a: value_a, axis_b: value_b},
                        denoise=values["denoise"],
                        lora_weight=values["lora_weight"],
                        controlnet_strength=values["controlnet_strength"],
                    )
                )

    return combinations
