"""이미지 정규화 + 캡션 사이드카 저장 — kohya 학습 폴더를 만든다.

버킷 배치 자체는 kohya가 학습 시점에 수행하므로(enable_bucket=true,
CLAUDE.md 3장) 여기서는 리사이즈를 최소화한다: kohya의 최대 버킷
해상도(1024px)를 넘는 이미지만 축소해 디스크·IO 부담을 줄이고,
업스케일은 하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from styleforge.config import settings

# configs/train_default.toml의 max_bucket_reso와 맞춘다 (Phase 3, CLAUDE.md 3장).
MAX_LONG_SIDE = 1024


@dataclass
class PreprocessResult:
    output_dir: Path
    image_count: int
    skipped: list[Path] = field(default_factory=list)


def prepare_dataset(name: str, images_with_captions: dict[Path, str]) -> PreprocessResult:
    """이미지를 data/prepared/{name}/에 복사(필요 시 축소)하고 캡션 .txt를 함께 저장한다."""
    output_dir = settings.data_dir / "prepared" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped: list[Path] = []
    saved = 0

    for image_path, caption in images_with_captions.items():
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size
                long_side = max(width, height)
                if long_side > MAX_LONG_SIDE:
                    scale = MAX_LONG_SIDE / long_side
                    img = img.resize((round(width * scale), round(height * scale)), Image.LANCZOS)

                dest_image = output_dir / f"{image_path.stem}.jpg"
                img.save(dest_image, "JPEG", quality=95)
        except Exception as exc:  # noqa: BLE001 — 손상 파일은 건너뛰고 기록만 한다
            print(f"[preprocess] 경고: 처리 실패 - {image_path}: {exc}")
            skipped.append(image_path)
            continue

        dest_caption = output_dir / f"{image_path.stem}.txt"
        dest_caption.write_text(caption, encoding="utf-8")
        saved += 1

    return PreprocessResult(output_dir=output_dir, image_count=saved, skipped=skipped)
