"""Phase 0 검증 스크립트.

comfy_client.py 단독으로 이미지 업로드 → 워크플로우 제출 → 진행률 수신 →
완료 대기 → 결과 이미지 다운로드까지의 전체 배관이 동작하는지 확인한다.
확산 모델(체크포인트)의 실제 생성 품질은 검증 대상이 아니므로 —
그건 apply/sweep 단계(Phase 1 이후)의 몫이다 — 모델 없이도 실행되는
LoadImage → SaveImage 왕복 워크플로우를 사용한다.

실제 ComfyUI 인스턴스가 COMFY_URL(.env, 기본 http://127.0.0.1:8188)에서
기동 중이어야 한다.

사용 예:
    python scripts/verify_comfy_client.py
    python scripts/verify_comfy_client.py --image samples/photo.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 cp949 대비

from PIL import Image  # noqa: E402
from rich.progress import Progress  # noqa: E402

from styleforge.apply.comfy_client import ComfyClient, ComfyClientError, ProgressUpdate  # noqa: E402
from styleforge.config import settings  # noqa: E402


def make_synthetic_test_image(path: Path) -> Path:
    """--image가 주어지지 않았을 때 업로드용 더미 이미지를 생성한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(120, 180, 220)).save(path)
    return path


def build_roundtrip_workflow(uploaded_filename: str) -> dict[str, Any]:
    """검증용 최소 워크플로우: 업로드된 이미지를 그대로 불러와 저장한다.

    체크포인트가 필요 없으므로 어떤 확산 모델이 설치돼 있든 comfy_client.py의
    통신 경로만 독립적으로 검증할 수 있다.
    """
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": uploaded_filename}},
        "2": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "phase0_verify", "images": ["1", 0]},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyClient 단독 검증 (Phase 0)")
    parser.add_argument("--image", type=Path, default=None, help="업로드할 이미지 (기본: 더미 이미지 자동 생성)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase0_verify"))
    args = parser.parse_args()

    image_path = args.image or make_synthetic_test_image(Path("outputs/phase0_verify/_input.png"))

    client = ComfyClient(settings.comfy_url)

    print(f"[검증] 이미지 업로드 중: {image_path}")
    uploaded_filename = client.upload_image(image_path)

    workflow = build_roundtrip_workflow(uploaded_filename)

    print(f"[검증] ComfyUI({settings.comfy_url})에 워크플로우 제출 중...")

    with Progress() as progress:
        task = progress.add_task("생성 중", total=None)

        def on_progress(update: ProgressUpdate) -> None:
            if update.max:
                progress.update(task, total=update.max, completed=update.value)

        try:
            saved_paths = client.run_workflow(workflow, args.output_dir, on_progress=on_progress)
        except ComfyClientError as exc:
            print(f"[검증] 실패: {exc}")
            raise SystemExit(1) from exc

    print(f"[검증] 성공 - 이미지 {len(saved_paths)}개 저장됨:")
    for path in saved_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
