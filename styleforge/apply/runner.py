"""apply 실행 흐름 — 이미지 업로드부터 결과 저장까지.

--strength 하나로 denoise/lora_weight/controlnet_strength 3축을 조절하는
것이 핵심이며, 사용자에게 3개 축을 직접 노출하지 않는다 (CLAUDE.md 7장).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from styleforge.apply.comfy_client import ComfyClient, ComfyClientError, ProgressUpdate
from styleforge.apply.workflow_loader import WorkflowLoadError, load_and_apply
from styleforge.config import settings

_SD15_TARGET_SIDE = 512  # SD1.5 네이티브 해상도 (CLAUDE.md 3장)


def _fit_to_target_side(width: int, height: int, *, target: int = _SD15_TARGET_SIDE) -> tuple[int, int]:
    """긴 변을 target에 맞춰 비율을 유지한 채 축소/확대한다.

    apply 워크플로우는 원본 이미지를 리사이즈 없이 그대로 VAEEncode/KSampler에
    넣었었다 — 휴대폰 사진처럼 4000px대 원본이 들어오면 SD1.5가 512 기준으로
    학습된 것과 무관하게 그 해상도 그대로 diffusion을 돌려 VAE가 OOM으로
    타일 모드에 빠지고 스텝당 수십 배 느려지는 문제가 있었다. 정사각 크롭 대신
    긴 변만 맞추는 이유는 민화 원본의 극단적 종횡비(CLAUDE.md 3장 버킷팅
    설명과 동일한 이유)를 구도 왜곡 없이 보존하기 위함이다.
    """
    longer = max(width, height)
    scale = target / longer
    new_w = max(8, round(width * scale / 8) * 8)
    new_h = max(8, round(height * scale / 8) * 8)
    return new_w, new_h


class ApplyError(RuntimeError):
    """apply 실행 중 발생한 오류."""


@dataclass
class StrengthParams:
    denoise: float
    lora_weight: float
    controlnet_strength: float


def strength_to_params(strength: float) -> StrengthParams:
    """--strength(0.0~1.0)를 denoise/lora_weight/controlnet_strength로 매핑한다.

    denoise를 올리면 화풍은 강해지지만 원본 구조가 흐려지므로 (CLAUDE.md
    2-(1)), ControlNet 강도도 함께 올려 구조를 붙잡는다. 이 함수가 s=0/1로
    만드는 최솟값·최댓값이 sweep/planner.py의 AXIS_RANGES 그대로다 —
    sweep은 apply가 --strength로 도달할 수 있는 구간을 촘촘히 탐색해
    최적 조합을 찾는 것이므로 두 구간이 어긋나면 안 된다.
    """
    s = max(0.0, min(1.0, strength))
    return StrengthParams(
        denoise=0.35 + 0.45 * s,
        lora_weight=0.4 + 0.6 * s,
        controlnet_strength=0.6 + 0.35 * s,
    )


def run_apply_with_params(
    *,
    image: Path,
    style: str,
    params: StrengthParams,
    workflow_name: str = "style_transfer_lineart",
    prompt: str | None = None,
    seed: int | None = None,
    output_dir: Path | None = None,
    on_progress: Callable[[ProgressUpdate], None] | None = None,
    extra_meta: dict | None = None,
) -> Path:
    """denoise/lora_weight/controlnet_strength을 직접 지정해 1회 실행한다.

    `run_apply()`는 이 함수 위에 strength_to_params() 매핑을 얹은 얇은
    래퍼다. sweep은 --strength 단일 축을 거치지 않고 3축을 직접 조절해야
    하므로 이 함수를 바로 호출한다 (CLAUDE.md 7장 sweep 사양).
    """
    if not image.is_file():
        raise ApplyError(f"Input image not found: {image}")

    resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    positive_prompt = prompt or f"{style}, best quality"

    with Image.open(image) as img:
        resize_width, resize_height = _fit_to_target_side(img.width, img.height)

    client = ComfyClient(settings.comfy_url)
    started_at = time.monotonic()

    try:
        uploaded_filename = client.upload_image(image)

        workflow = load_and_apply(
            workflow_name,
            {
                "input_image": uploaded_filename,
                "lora_name": f"{style}.safetensors",
                "positive_prompt": positive_prompt,
                "denoise": params.denoise,
                "lora_weight": params.lora_weight,
                "controlnet_strength": params.controlnet_strength,
                "seed": resolved_seed,
                "resize_width": resize_width,
                "resize_height": resize_height,
            },
        )

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs/applied") / f"{timestamp}_{style}"

        saved_paths = client.run_workflow(workflow, output_dir, on_progress=on_progress)
    except (ComfyClientError, WorkflowLoadError) as exc:
        raise ApplyError(str(exc)) from exc

    elapsed = time.monotonic() - started_at

    meta = {
        "input_image": str(image),
        "style": style,
        "denoise": params.denoise,
        "lora_weight": params.lora_weight,
        "controlnet_strength": params.controlnet_strength,
        "workflow": workflow_name,
        "prompt": positive_prompt,
        "seed": resolved_seed,
        "resize_width": resize_width,
        "resize_height": resize_height,
        "output_images": [str(p) for p in saved_paths],
        "elapsed_seconds": round(elapsed, 2),
        **(extra_meta or {}),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_dir


def run_apply(
    *,
    image: Path,
    style: str,
    strength: float = 0.6,
    workflow_name: str = "style_transfer_lineart",
    prompt: str | None = None,
    seed: int | None = None,
    on_progress: Callable[[ProgressUpdate], None] | None = None,
) -> Path:
    """apply 명령의 실행 흐름. 결과가 저장된 디렉터리 경로를 반환한다."""
    params = strength_to_params(strength)
    return run_apply_with_params(
        image=image,
        style=style,
        params=params,
        workflow_name=workflow_name,
        prompt=prompt,
        seed=seed,
        on_progress=on_progress,
        extra_meta={"strength": strength},
    )
