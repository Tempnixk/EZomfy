"""B경로: WD14 tagger로 자동 캡셔닝한다 (라벨 없는 임의 폴더용, CLAUDE.md 2-(2)).

라벨 JSON 없이도 학습 가능하도록 태그를 뽑은 뒤, 화풍(매체·기법) 관련
태그를 규칙 기반으로 제거하고 트리거 워드를 선두에 삽입한다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

from styleforge.config import settings

# WD14 태그 중 화풍(매체·기법)을 나타내는 것으로 보고 제거하는 목록.
# 완전한 목록이 아니라 규칙 기반 1차 필터이며, 실사용 중 확장한다.
STYLE_TAG_BLOCKLIST = {
    "traditional media",
    "ink",
    "watercolor (medium)",
    "monochrome",
    "sketch",
    "lineart",
    "greyscale",
    "oil painting (medium)",
    "painting (medium)",
    "scan",
    "artist name",
    "signature",
    "watermark",
}

RATING_CATEGORY = 9  # WD14 tags.csv에서 category=9는 rating(safe/questionable/explicit)


class CaptionAutoError(RuntimeError):
    """WD14 태깅 중 발생한 오류."""


class WD14Tagger:
    """onnxruntime 기반 WD14 태거. 모델·태그 CSV 경로는 .env로 주입한다."""

    def __init__(self) -> None:
        if settings.wd14_model_path is None or settings.wd14_tags_csv is None:
            raise CaptionAutoError("WD14_MODEL_PATH / WD14_TAGS_CSV가 설정되지 않았습니다 (.env 확인)")
        if not settings.wd14_model_path.is_file():
            raise CaptionAutoError(f"WD14 model not found: {settings.wd14_model_path}")
        if not settings.wd14_tags_csv.is_file():
            raise CaptionAutoError(f"WD14 tags csv not found: {settings.wd14_tags_csv}")

        self._session = onnxruntime.InferenceSession(str(settings.wd14_model_path))
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = self._session.get_inputs()[0].shape[1]
        self._tags = self._load_tags(settings.wd14_tags_csv)

    @staticmethod
    def _load_tags(tags_csv: Path) -> list[tuple[str, int]]:
        with tags_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [(row["name"], int(row["category"])) for row in reader]

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((self._input_size, self._input_size))
        array = np.asarray(resized, dtype=np.float32)
        array = array[:, :, ::-1]  # RGB -> BGR (WD14 학습 시 입력 순서)
        return array[np.newaxis, :, :, :]

    def tag(self, image_path: Path) -> list[str]:
        """이미지에서 threshold 이상 확률의 일반 태그만 뽑는다 (rating 카테고리 제외)."""
        with Image.open(image_path) as img:
            batch = self._preprocess(img)

        probs = self._session.run(None, {self._input_name: batch})[0][0]

        return [
            name
            for (name, category), prob in zip(self._tags, probs)
            if category != RATING_CATEGORY and prob >= settings.wd14_tag_threshold
        ]


def build_caption(tags: list[str], trigger_word: str) -> str:
    """WD14 태그에서 화풍 관련 태그를 제거하고 트리거 워드를 선두에 삽입한다."""
    content_tags = [tag for tag in tags if tag not in STYLE_TAG_BLOCKLIST]
    return ", ".join([trigger_word, *content_tags])
