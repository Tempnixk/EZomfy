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

from styleforge.apply.comfy_client import ComfyClient, ComfyClientError, ProgressUpdate
from styleforge.apply.workflow_loader import WorkflowLoadError, load_and_apply
from styleforge.config import settings


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
    2-(1)), ControlNet 강도도 함께 올려 구조를 붙잡는다. 구간 값은 sweep이
    구현되기 전까지 쓰는 초기 기본값이며 Phase 4에서 실측 후 조정한다.
    """
    s = max(0.0, min(1.0, strength))
    return StrengthParams(
        denoise=0.35 + 0.45 * s,
        lora_weight=0.4 + 0.6 * s,
        controlnet_strength=0.6 + 0.35 * s,
    )


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
    if not image.is_file():
        raise ApplyError(f"Input image not found: {image}")

    resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    params = strength_to_params(strength)
    positive_prompt = prompt or f"{style}, best quality"

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
            },
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/applied") / f"{timestamp}_{style}"

        saved_paths = client.run_workflow(workflow, output_dir, on_progress=on_progress)
    except (ComfyClientError, WorkflowLoadError) as exc:
        raise ApplyError(str(exc)) from exc

    elapsed = time.monotonic() - started_at

    meta = {
        "input_image": str(image),
        "style": style,
        "strength": strength,
        "denoise": params.denoise,
        "lora_weight": params.lora_weight,
        "controlnet_strength": params.controlnet_strength,
        "workflow": workflow_name,
        "prompt": positive_prompt,
        "seed": resolved_seed,
        "output_images": [str(p) for p in saved_paths],
        "elapsed_seconds": round(elapsed, 2),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_dir
