"""정량 평가 — CLIP 기반 스타일 유사도/원본 보존도, CLIP Score, FID.

CLIP Score·FID는 AI 허브 레퍼런스 구현 값(CLAUDE.md 2-(4))과 대조하는 데
쓴다. 학습 조건이 동일하지 않으므로 우열을 주장하지 않고 "레퍼런스 대비
어느 위치인지"만 `docs/benchmarks.md`에 근거와 함께 기록한다 (Phase 5).

무거운 의존성(torch·open_clip)은 이 모듈이 import될 때가 아니라 실제로
쓰일 때 지연 로드한다 — apply/train처럼 CLIP을 쓰지 않는 경로가
불필요하게 torch를 물고 오지 않게 하기 위함이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 구축기관이 동일 데이터로 수행한 레퍼런스 구현 결과 (docs/dataset-aihub-minhwa.md 1장).
REFERENCE_BASELINE = {
    "clip_score": 0.74,
    "fid": 25.5,
}

_CLIP_MODEL_NAME = "ViT-B-32-quickgelu"  # "openai" 프리트레인은 QuickGELU 활성화를 쓴다 (open_clip 이름 규약)
_CLIP_PRETRAINED = "openai"
_CLIPSCORE_WEIGHT = 2.5  # Hessel et al. 2021 "CLIPScore" 정의의 가중치


class MetricsError(RuntimeError):
    """평가 지표 산출 중 발생한 오류."""


_model = None
_preprocess = None
_tokenizer = None


def _load_clip():
    global _model, _preprocess, _tokenizer
    if _model is None:
        try:
            import open_clip
        except ImportError as exc:
            raise MetricsError(
                "open_clip이 설치되어 있지 않습니다. 평가 기능을 쓰려면 "
                "`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu "
                "open-clip-torch`로 설치하세요."
            ) from exc

        _model, _, _preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
        )
        _tokenizer = open_clip.get_tokenizer(_CLIP_MODEL_NAME)
        _model.eval()
    return _model, _preprocess, _tokenizer


def _image_embedding(path: Path):
    import torch

    model, preprocess, _ = _load_clip()
    from PIL import Image

    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image)
    return features / features.norm(dim=-1, keepdim=True)


def _text_embedding(text: str):
    import torch

    model, _, tokenizer = _load_clip()
    tokens = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(tokens)
    return features / features.norm(dim=-1, keepdim=True)


def clip_image_similarity(image_a: Path, image_b: Path) -> float:
    """두 이미지의 CLIP 임베딩 코사인 유사도 (-1~1). 스타일 적용도/원본 보존도 계산의 기반이다."""
    emb_a = _image_embedding(image_a)
    emb_b = _image_embedding(image_b)
    return float((emb_a @ emb_b.T).item())


def clip_text_score(image: Path, text: str) -> float:
    """CLIPScore(Hessel et al., 2021) — w * max(cos(image, text), 0). 레퍼런스 CLIP Score 0.74와 같은 정의다."""
    emb_image = _image_embedding(image)
    emb_text = _text_embedding(text)
    cos = float((emb_image @ emb_text.T).item())
    return _CLIPSCORE_WEIGHT * max(cos, 0.0)


@dataclass
class StyleEvalResult:
    style_similarity: float  # 생성 이미지 <-> 학습에 쓰인 화풍 레퍼런스 이미지들의 평균 CLIP 유사도
    preservation_similarity: float  # 생성 이미지 <-> 입력 원본 이미지의 CLIP 유사도


def evaluate_style_transfer(
    generated: Path,
    original: Path,
    style_reference_images: list[Path],
) -> StyleEvalResult:
    """sweep 한 조합의 결과를 "화풍이 얼마나 입었는가" / "원본이 얼마나 보존됐는가"로 채점한다."""
    if not style_reference_images:
        raise MetricsError("style_reference_images가 비어 있습니다")

    preservation = clip_image_similarity(generated, original)
    style_scores = [clip_image_similarity(generated, ref) for ref in style_reference_images]
    style_similarity = sum(style_scores) / len(style_scores)

    return StyleEvalResult(style_similarity=style_similarity, preservation_similarity=preservation)


def compute_fid(
    generated_dir: Path,
    real_dir: Path,
    *,
    device: str = "cpu",
    num_workers: int = 0,
    batch_size: int = 1,
) -> float:
    """generated_dir와 real_dir(예: AI 허브 Validation 세트) 간 FID.

    pytorch-fid 패키지의 표준 InceptionV3(FID 전용 가중치)를 그대로 쓴다 —
    직접 구현하면 레퍼런스 논문·도구들이 쓰는 것과 다른 Inception 가중치를
    쓰게 되어 절대값 비교가 무의미해지기 때문이다.

    `num_workers` 기본값은 0이다. pytorch-fid 내부 DataLoader는 기본
    `num_workers=1`로 별도 프로세스를 spawn하는데, Windows에서
    `if __name__ == "__main__":` 가드 없는 호출부(예: 인터랙티브 스크립트)와
    조합되면 그 spawn이 죽는다. 메인 프로세스에서 바로 도는 0이 안전하다.

    `batch_size` 기본값은 1이다. pytorch-fid의 DataLoader는 리사이즈 없이
    원본 해상도 그대로 배치를 쌓는데, 민화 데이터셋은 족자·병풍 형태로
    이미지마다 종횡비가 달라 배치 크기가 2 이상이면 텐서 stack이 크기
    불일치로 실패한다. 배치당 1장이면 이 문제가 생기지 않는다.

    Frechet distance는 `pytorch_fid.fid_score.calculate_frechet_distance`를
    쓰지 않고 이 함수 안에서 다시 계산한다 — 그 함수가 `scipy.linalg.sqrtm`을
    `disp=False` 인자로 호출하는데, 이 인자가 최신 scipy(1.18+)에서 제거되어
    `TypeError`가 난다. 활성값 통계(mu, sigma)는 pytorch-fid의 표준
    InceptionV3 경로 그대로 뽑고, 그 이후의 순수 수학 계산만 scipy 최신
    시그니처에 맞춰 다시 구현한다.
    """
    try:
        import numpy as np
        from pytorch_fid.fid_score import compute_statistics_of_path
        from pytorch_fid.inception import InceptionV3
        from scipy import linalg
    except ImportError as exc:
        raise MetricsError(
            "pytorch-fid가 설치되어 있지 않습니다. `pip install pytorch-fid`로 설치하세요."
        ) from exc

    dims = 2048
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device)

    mu1, sigma1 = compute_statistics_of_path(
        str(generated_dir), model, batch_size, dims, device, num_workers
    )
    mu2, sigma2 = compute_statistics_of_path(
        str(real_dir), model, batch_size, dims, device, num_workers
    )

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
