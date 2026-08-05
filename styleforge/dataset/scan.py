"""폴더 스캔 + 유효성 검사 (CLAUDE.md 2-(3)).

train 실행 전 반드시 이 모듈로 검증한다. 이 모듈은 판단 근거(ScanReport)만
만들고, 경고 발생 시 진행 여부를 사용자에게 묻는 것은 호출부(train 명령,
Phase 3)의 책임이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MIN_IMAGE_COUNT = 15
MIN_RESOLUTION_PX = 512
# 1:4까지는 정상으로 간주한다 — 민화 족자·병풍 형태의 극단적 종횡비 때문
# (docs/dataset-aihub-minhwa.md 3-2절).
MAX_ASPECT_RATIO = 4.0
# perceptual hash 해밍 거리가 이 값 이하면 근접 중복으로 간주한다.
HASH_DISTANCE_THRESHOLD = 4


@dataclass
class ScanWarning:
    code: str
    message: str
    files: list[Path] = field(default_factory=list)


@dataclass
class ScanReport:
    total_images: int
    warnings: list[ScanWarning]

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


def scan_folder(folder: Path) -> ScanReport:
    """folder 바로 아래의 이미지들을 검증하고 ScanReport를 반환한다."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    warnings: list[ScanWarning] = []

    if len(image_paths) < MIN_IMAGE_COUNT:
        warnings.append(
            ScanWarning(
                code="low_count",
                message=f"이미지가 {len(image_paths)}장뿐입니다 (권장 {MIN_IMAGE_COUNT}장 이상)",
            )
        )

    low_res: list[Path] = []
    aspect_outliers: list[Path] = []
    hashes: dict[Path, imagehash.ImageHash] = {}

    for path in image_paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
                hashes[path] = imagehash.phash(img)
        except Exception as exc:  # noqa: BLE001 — 손상 파일은 경고로 남기고 다른 검사에서 제외한다
            warnings.append(ScanWarning(code="unreadable", message=f"이미지를 열 수 없음: {exc}", files=[path]))
            continue

        if min(width, height) < MIN_RESOLUTION_PX:
            low_res.append(path)

        ratio = max(width, height) / min(width, height)
        if ratio > MAX_ASPECT_RATIO:
            aspect_outliers.append(path)

    if low_res:
        warnings.append(
            ScanWarning(
                code="low_resolution",
                message=f"해상도 미달(최소 변 {MIN_RESOLUTION_PX}px) 이미지 {len(low_res)}장",
                files=low_res,
            )
        )

    if aspect_outliers:
        warnings.append(
            ScanWarning(
                code="aspect_ratio_outlier",
                message=f"종횡비가 1:{MAX_ASPECT_RATIO:.0f}를 초과하는 이미지 {len(aspect_outliers)}장",
                files=aspect_outliers,
            )
        )

    duplicate_pairs = _find_near_duplicates(hashes)
    if duplicate_pairs:
        dup_files = sorted({p for pair in duplicate_pairs for p in pair})
        warnings.append(
            ScanWarning(
                code="near_duplicate",
                message=f"근접 중복 이미지 쌍 {len(duplicate_pairs)}개 발견",
                files=dup_files,
            )
        )

    return ScanReport(total_images=len(image_paths), warnings=warnings)


def _find_near_duplicates(hashes: dict[Path, imagehash.ImageHash]) -> list[tuple[Path, Path]]:
    """perceptual hash 해밍 거리가 임계값 이하인 쌍을 근접 중복으로 판정한다."""
    items = list(hashes.items())
    pairs: list[tuple[Path, Path]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1] - items[j][1] <= HASH_DISTANCE_THRESHOLD:
                pairs.append((items[i][0], items[j][0]))
    return pairs
