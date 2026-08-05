"""AI 허브 「한국 전통 민화 제작 데이터」 라벨 JSON 스키마 어댑터.

설명서와 실제 데이터가 다른 부분이 있다 (docs/dataset-aihub-minhwa.md 3장):
화풍 필드명은 painting_style이 아니라 painting_type이고, drawing_type은
"채색, 혁필"처럼 복합값을 가질 수 있다. 필드 접근은 전부 .get()으로
방어적으로 처리하고, 필수 값이 없으면 호출부가 경고 후 제외할 수 있도록
None을 그대로 흘려보낸다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 화목 코드 → 폴더명. 번호가 화목 코드 순서와 일치하지 않으므로 표로 고정한다
# (docs/dataset-aihub-minhwa.md 2장 화목 코드표).
GENRE_CODE_TO_FOLDER: dict[str, str] = {
    "HJ": "09.화조도",
    "MJ": "15.문자도",
    "SS": "10.산수화",
    "AH": "05.어해도",
    "HO": "16.혼성도",
    "HH": "07.화훼도",
    "KY": "13.기용화",
    "IM": "01.인물화",
    "SH": "14.설화화",
    "YS": "04.영수화",
    "SO": "08.소과도",
    "SU": "11.수석도",
    "CS": "03.축수도",
    "CC": "06.초충도",
    "DS": "02.도석화",
    "OU": "12.옥우화",
}


class LabelParseError(RuntimeError):
    """라벨 JSON 파싱 중 발생한 오류 (필수 구조 자체가 깨진 경우)."""


@dataclass
class AihubLabel:
    file_name: str | None
    subject_type_id: str | None  # 화목 코드 (예: HJ)
    drawing_type: str | None  # 복합값 가능 ("채색, 혁필")
    painting_type: str | None  # ★ 설명서의 painting_style이 아니다
    ground_material: str | None
    sources: str | None
    img_width: int | None
    img_height: int | None
    keyword_eng: str | None
    composition_eng: str | None
    overview_eng: str | None


def load_label(label_path: Path) -> AihubLabel:
    """라벨 JSON 파일을 읽어 AihubLabel로 변환한다."""
    if not label_path.is_file():
        raise LabelParseError(f"Label file not found: {label_path}")

    try:
        with label_path.open(encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise LabelParseError(f"Invalid JSON in {label_path}: {exc}") from exc

    images = raw.get("images", {})
    folkpainting = raw.get("folkpainting", {})
    caption_eng = raw.get("caption", {}).get("caption_eng", {})

    return AihubLabel(
        file_name=images.get("file_name"),
        subject_type_id=folkpainting.get("subject_type_id"),
        drawing_type=folkpainting.get("drawing_type"),
        painting_type=folkpainting.get("painting_type"),
        ground_material=folkpainting.get("ground_material"),
        sources=folkpainting.get("sources"),
        img_width=images.get("img_width"),
        img_height=images.get("img_height"),
        keyword_eng=caption_eng.get("keyword_eng"),
        composition_eng=caption_eng.get("composition_eng"),
        overview_eng=caption_eng.get("overview_eng"),
    )


def matches_meta_filter(label: AihubLabel, meta_filter: dict[str, str]) -> bool:
    """drawing_type=채색,painting_type=일필+공필 같은 2차 필터. 완전 일치만 허용한다.

    복합값(예: "채색, 혁필")을 부분 문자열로 매칭하면 문서 3-1절 기준으로
    21장이 잘못 포함되므로, 필드 원문과 완전히 같을 때만 통과시킨다.
    """
    for field_name, expected in meta_filter.items():
        if getattr(label, field_name, None) != expected:
            return False
    return True
