"""A경로: 외부 제공 라벨 JSON에서 학습 캡션을 만든다 (docs/dataset-aihub-minhwa.md 4장).

캡션의 원칙: 내용은 남기고 스타일은 지운다. 무엇이 그려졌는지(keyword_eng,
composition_eng)만 남기고, 화풍·기법·재료 표현은 제거한다 — 그건 트리거
워드가 대신 담당한다.
"""
from __future__ import annotations

import re

from styleforge.dataset.adapters.aihub_minhwa import AihubLabel

# docs/dataset-aihub-minhwa.md 4장 "스타일 토큰 제거 대상"
_STYLE_TOKEN_RE = re.compile(
    r"traditional korean folk painting"
    r"|\bminhwa\b"
    r"|\bfolk painting\b"
    r"|\b\w+[- ]and[- ]\w+ painting\b"  # 예: bird-and-flower painting
    r"|\bfloral painting\b"
    r"|\bink\b"
    r"|color on hanji"
    r"|\bjoseon\b",
    re.IGNORECASE,
)

# 실측 데이터의 keyword_eng는 "Hwajodo (flower and bird painting)"처럼
# "화목 로마자 표기 (영문 화목 설명)" 형태로 화풍을 태그에 함께 심어둔다.
# 괄호 안에 painting이 들어간 태그는 장르 자체를 가리키므로 통째로 버린다.
_GENRE_PAREN_RE = re.compile(r"\([^)]*painting[^)]*\)", re.IGNORECASE)


def _tag_is_style_related(tag: str) -> bool:
    return bool(_STYLE_TOKEN_RE.search(tag)) or bool(_GENRE_PAREN_RE.search(tag))


def _clean_composition(text: str) -> str:
    text = _STYLE_TOKEN_RE.sub("", text)
    text = _GENRE_PAREN_RE.sub("", text)
    text = re.sub(r"\(\s*\)", "", text)  # 내용을 전부 지운 빈 괄호
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)  # " ," -> ","
    return text.strip()


def build_caption(label: AihubLabel, trigger_word: str) -> str | None:
    """라벨에서 학습용 영문 캡션을 만든다. 쓸 수 있는 필드가 없으면 None을 반환한다."""
    if not label.keyword_eng and not label.composition_eng:
        return None

    tags: list[str] = []

    if label.keyword_eng:
        for raw_tag in label.keyword_eng.split(","):
            raw_tag = raw_tag.strip()
            if raw_tag and not _tag_is_style_related(raw_tag):
                tags.append(raw_tag)

    if label.composition_eng:
        composition = _clean_composition(label.composition_eng)
        first_sentence = re.split(r"(?<=[.!?])\s", composition, maxsplit=1)[0].strip(" .,")
        if first_sentence:
            tags.append(first_sentence)

    if not tags:
        return None

    seen: set[str] = set()
    unique_tags: list[str] = []
    for tag in tags:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            unique_tags.append(tag)

    return ", ".join([trigger_word, *unique_tags])
