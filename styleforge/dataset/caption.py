"""캡션 경로 자동 선택 — A(외부 라벨) vs B(WD14 자동) (CLAUDE.md 2-(2)).

입력 폴더에 라벨 JSON이 있으면 A경로, 없으면 B경로를 자동 선택한다.
mode="external"/"auto"로 강제 지정할 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from styleforge.dataset import caption_auto, caption_external
from styleforge.dataset.adapters.aihub_minhwa import LabelParseError, load_label

CaptionMode = Literal["auto", "external"]


class CaptionError(RuntimeError):
    """캡셔닝 중 발생한 오류."""


@dataclass
class CaptionTarget:
    image_path: Path
    label_path: Path | None = None


def build_captions(
    targets: list[CaptionTarget],
    trigger_word: str,
    *,
    mode: CaptionMode | None = None,
) -> dict[Path, str]:
    """각 이미지의 캡션을 만든다. mode가 None이면 라벨 JSON 유무로 자동 선택한다."""
    tagger: caption_auto.WD14Tagger | None = None
    captions: dict[Path, str] = {}
    skipped: list[Path] = []

    for target in targets:
        use_external = (target.label_path is not None) if mode is None else (mode == "external")

        if use_external:
            if target.label_path is None:
                raise CaptionError(f"--caption-mode external이지만 라벨이 없습니다: {target.image_path}")

            try:
                label = load_label(target.label_path)
            except LabelParseError as exc:
                skipped.append(target.image_path)
                print(f"[caption] 경고: 라벨 파싱 실패 - {exc}")
                continue

            caption = caption_external.build_caption(label, trigger_word)
            if caption is None:
                skipped.append(target.image_path)
                print(f"[caption] 경고: 캡션에 쓸 필드가 없어 제외 - {target.image_path}")
                continue

            captions[target.image_path] = caption
        else:
            if tagger is None:
                tagger = caption_auto.WD14Tagger()
            tags = tagger.tag(target.image_path)
            captions[target.image_path] = caption_auto.build_caption(tags, trigger_word)

    if skipped:
        print(f"[caption] {len(skipped)}개 이미지가 캡션 생성에서 제외되었습니다")

    return captions
