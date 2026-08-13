"""kohya(sd-scripts)용 TOML 설정 생성.

configs/train_default.toml(체크인된 기본 하이퍼파라미터)을 읽어 이번
실행의 런타임 값(데이터 경로, 체크포인트, 출력 위치, 시드)을 얹은 뒤,
kohya가 --dataset_config / --config_file로 바로 읽는 TOML 두 개를
data/prepared/{name}/ 아래에 생성한다.

데이터셋 config는 kohya의 [[datasets]]/[[datasets.subsets]] 최신 스키마를
쓴다. preprocess.py가 만드는 평평한(반복횟수 접두어 없는) 폴더 구조를
폴더명 규칙 변경 없이 그대로 가리킬 수 있기 때문이다 — 반대로 레거시
DreamBooth 방식(--train_data_dir)은 `{반복횟수}_{트리거워드}` 형태의
서브폴더명을 요구해 Phase 2의 출력 형식과 맞지 않는다.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from styleforge.config import settings

DEFAULT_TEMPLATE_PATH = Path("configs/train_default.toml")


class ConfigBuildError(RuntimeError):
    """kohya 설정 생성 중 발생한 오류."""


def _load_template(template_path: Path) -> dict[str, Any]:
    if not template_path.is_file():
        raise ConfigBuildError(f"Training config template not found: {template_path}")
    with template_path.open("rb") as f:
        return tomllib.load(f)


def build_dataset_config(
    prepared_dir: Path,
    *,
    num_repeats: int,
    resolution: int,
    batch_size: int,
    enable_bucket: bool,
    bucket_reso_steps: int,
    min_bucket_reso: int,
    max_bucket_reso: int,
) -> dict[str, Any]:
    """kohya --dataset_config용 TOML 구조를 만든다."""
    return {
        "general": {
            "shuffle_caption": True,
            "caption_extension": ".txt",
        },
        "datasets": [
            {
                "resolution": resolution,
                "batch_size": batch_size,
                "enable_bucket": enable_bucket,
                "bucket_reso_steps": bucket_reso_steps,
                "min_bucket_reso": min_bucket_reso,
                "max_bucket_reso": max_bucket_reso,
                "subsets": [
                    {
                        "image_dir": str(prepared_dir),
                        "num_repeats": num_repeats,
                    }
                ],
            }
        ],
    }


def build_train_config(
    training_template: dict[str, Any],
    *,
    output_dir: Path,
    output_name: str,
    logging_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """training_template(configs/train_default.toml [training])에 런타임 값을 덮어쓴다."""
    if settings.sd15_checkpoint_path is None:
        raise ConfigBuildError("SD15_CHECKPOINT_PATH가 설정되지 않았습니다 (.env 확인)")

    config = dict(training_template)
    config.update(
        {
            "pretrained_model_name_or_path": str(settings.sd15_checkpoint_path),
            "output_dir": str(output_dir),
            "output_name": output_name,
            "logging_dir": str(logging_dir),
            "seed": seed,
        }
    )
    return config


def write_configs(
    name: str,
    prepared_dir: Path,
    *,
    num_repeats: int = 10,
    seed: int = 0,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> tuple[Path, Path]:
    """dataset config + train config TOML을 파일로 쓰고 (dataset_path, train_path)를 반환한다."""
    template = _load_template(template_path)
    res_cfg = template.get("resolution", {})
    training_cfg = template.get("training", {})

    dataset_config = build_dataset_config(
        prepared_dir.resolve(),
        num_repeats=num_repeats,
        resolution=res_cfg.get("resolution", 512),
        batch_size=training_cfg.get("train_batch_size", 1),
        enable_bucket=res_cfg.get("enable_bucket", True),
        bucket_reso_steps=res_cfg.get("bucket_reso_steps", 64),
        min_bucket_reso=res_cfg.get("min_bucket_reso", 256),
        max_bucket_reso=res_cfg.get("max_bucket_reso", 1024),
    )

    run_dir = settings.data_dir / "prepared" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # kohya 서브프로세스는 kohya_script_dir을 cwd로 실행되므로(train/runner.py
    # run_kohya) 상대 경로를 그대로 넘기면 엉뚱한 위치를 가리킨다. TOML에
    # 적히는 모든 경로는 절대 경로로 고정한다.
    train_config = build_train_config(
        training_cfg,
        output_dir=Path("outputs/loras").resolve(),
        output_name=name,
        logging_dir=(run_dir / "logs").resolve(),
        seed=seed,
    )

    dataset_path = run_dir / "dataset_config.toml"
    train_path = run_dir / "train_config.toml"

    dataset_path.write_bytes(tomli_w.dumps(dataset_config).encode("utf-8"))
    train_path.write_bytes(tomli_w.dumps(train_config).encode("utf-8"))

    return dataset_path, train_path
