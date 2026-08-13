"""sweep/planner.py 단위 테스트 — 순수 로직(조합 생성·축 검증)만 다룬다."""
from __future__ import annotations

import pytest

from styleforge.sweep.planner import AXIS_RANGES, PlanError, generate_plan, parse_axes


def test_parse_axes_default_two_axes():
    assert parse_axes("denoise,lora_weight") == ["denoise", "lora_weight"]


def test_parse_axes_rejects_more_than_two():
    with pytest.raises(PlanError):
        parse_axes("denoise,lora_weight,controlnet_strength")


def test_parse_axes_rejects_unknown():
    with pytest.raises(PlanError):
        parse_axes("bogus")


def test_parse_axes_rejects_duplicate():
    with pytest.raises(PlanError):
        parse_axes("denoise,denoise")


def test_generate_plan_two_axes_covers_full_grid():
    axes = parse_axes("denoise,lora_weight")
    combos = generate_plan(axes, steps=3)
    assert len(combos) == 9

    denoise_values = {round(c.denoise, 6) for c in combos}
    expected_denoise = {round(v, 6) for v in _linspace(*AXIS_RANGES["denoise"], 3)}
    assert denoise_values == expected_denoise


def test_generate_plan_one_axis_holds_others_at_base_strength():
    from styleforge.apply.runner import strength_to_params

    axes = parse_axes("denoise")
    combos = generate_plan(axes, steps=4, base_strength=0.6)
    base = strength_to_params(0.6)

    for combo in combos:
        assert combo.lora_weight == pytest.approx(base.lora_weight)
        assert combo.controlnet_strength == pytest.approx(base.controlnet_strength)


def test_generate_plan_rejects_steps_below_two():
    with pytest.raises(PlanError):
        generate_plan(["denoise"], steps=1)


def _linspace(lo: float, hi: float, steps: int) -> list[float]:
    return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
