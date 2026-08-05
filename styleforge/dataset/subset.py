"""대규모 AI 허브 데이터셋에서 화목 하나만 선별하는 서브셋 파이프라인.

선별은 디렉터리 선택으로 처리하므로 파일명·캡션을 파싱하지 않는다
(docs/dataset-aihub-minhwa.md 2장). 화목 코드 → 폴더명 매핑, 상세묘사
제외, 이미지-라벨 페어링, 2차 메타 필터, 샘플링까지 이 모듈이 담당한다.

품질 검증(scan.py)은 이 모듈 밖에서 최종 선별 결과 폴더에 대해
수행한다 (train 명령 오케스트레이션, Phase 3) — 선별과 검증의 책임을
분리해 각 모듈을 단일 목적으로 유지한다.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from styleforge.config import settings
from styleforge.dataset.adapters.aihub_minhwa import (
    GENRE_CODE_TO_FOLDER,
    LabelParseError,
    load_label,
    matches_meta_filter,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class SubsetError(RuntimeError):
    """서브셋 선별 중 발생한 오류."""


@dataclass
class SubsetItem:
    image_path: str
    label_path: str
    subject_type_id: str | None
    drawing_type: str | None
    painting_type: str | None


def _dataset_data_root() -> Path:
    if settings.dataset_root is None:
        raise SubsetError("DATASET_ROOT가 설정되지 않았습니다 (.env 확인)")
    return settings.dataset_root / "13.한국 전통 민화 제작 데이터" / "3.개방데이터" / "1.데이터"


def _source_and_label_dirs(data_root: Path, folder_name: str, *, detail: bool) -> tuple[Path, Path]:
    prefix = "상세묘사데이터" if detail else "기본데이터"
    source_dir = data_root / "Training" / "01.원천데이터" / f"TS_{prefix}_{folder_name}"
    # 이미지 ↔ 라벨 페어링은 경로 치환으로 처리한다 (01.원천데이터→02.라벨링데이터, TS_→TL_).
    label_dir = data_root / "Training" / "02.라벨링데이터" / f"TL_{prefix}_{folder_name}"
    return source_dir, label_dir


def select_subset(
    genre_code: str,
    *,
    meta_filter: dict[str, str] | None = None,
    include_detail: bool = False,
    limit: int = 40,
    seed: int = 0,
) -> list[SubsetItem]:
    """화목 코드로 서브셋을 선별한다.

    include_detail=True는 상세묘사 폴더도 후보에 포함하지만, 원본의 부분
    확대본이라 중복 학습으로 스타일이 왜곡될 수 있어 기본값은 False다.
    """
    folder_name = GENRE_CODE_TO_FOLDER.get(genre_code)
    if folder_name is None:
        raise SubsetError(f"알 수 없는 화목 코드: {genre_code}")

    data_root = _dataset_data_root()
    detail_flags = [False, True] if include_detail else [False]

    candidates: list[SubsetItem] = []
    skipped_unpaired = 0
    skipped_unparsable = 0

    for detail in detail_flags:
        source_dir, label_dir = _source_and_label_dirs(data_root, folder_name, detail=detail)
        if not source_dir.is_dir():
            raise SubsetError(f"원천 데이터 폴더가 없습니다: {source_dir}")

        # 순번에 결번이 있으므로 번호를 순회하지 않고 실제 디렉터리 목록을 읽는다.
        for image_path in sorted(source_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.json"
            if not label_path.is_file():
                skipped_unpaired += 1
                continue

            try:
                label = load_label(label_path)
            except LabelParseError:
                skipped_unparsable += 1
                continue

            if meta_filter and not matches_meta_filter(label, meta_filter):
                continue

            candidates.append(
                SubsetItem(
                    image_path=str(image_path),
                    label_path=str(label_path),
                    subject_type_id=label.subject_type_id,
                    drawing_type=label.drawing_type,
                    painting_type=label.painting_type,
                )
            )

    if skipped_unpaired:
        print(f"[subset] 경고: 이미지-라벨 페어링 실패 {skipped_unpaired}건 제외")
    if skipped_unparsable:
        print(f"[subset] 경고: 라벨 파싱 실패 {skipped_unparsable}건 제외")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:limit]


def write_manifest(
    name: str,
    items: list[SubsetItem],
    *,
    genre_code: str,
    meta_filter: dict[str, str] | None,
) -> Path:
    """선별 기준과 최종 목록을 data/prepared/{name}/subset_manifest.json에 기록한다."""
    manifest_dir = settings.data_dir / "prepared" / name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "subset_manifest.json"

    manifest = {
        "genre_code": genre_code,
        "meta_filter": meta_filter or {},
        "count": len(items),
        "items": [asdict(item) for item in items],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
