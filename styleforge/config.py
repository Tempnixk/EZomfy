"""전역 설정 — .env를 읽어 프로젝트 전체에서 쓰는 값을 노출한다.

경로·주소를 코드에 하드코딩하지 않는다는 원칙(CLAUDE.md 3장)에 따라
새 설정값은 여기에 필드로 추가하고 .env.example도 함께 갱신한다.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    comfy_url: str = "http://127.0.0.1:8188"
    workflows_dir: Path = Path("workflows")

    # dataset/ — AI 허브 「한국 전통 민화 제작 데이터」 루트 (docs/dataset-aihub-minhwa.md 2장)
    dataset_root: Path | None = None
    data_dir: Path = Path("data")

    # dataset/caption_auto.py — WD14 tagger (onnxruntime)
    wd14_model_path: Path | None = None
    wd14_tags_csv: Path | None = None
    wd14_tag_threshold: float = 0.35

    # train/ — kohya_ss(sd-scripts)는 별도 venv로 서브프로세스 호출한다 (CLAUDE.md 3장 가상환경 분리)
    kohya_python: Path | None = None
    kohya_script_dir: Path | None = None
    sd15_checkpoint_path: Path | None = None  # kohya 학습용 SD1.5 체크포인트 (ComfyUI와 별개 설정)

    # train 실행 시 ComfyUI를 자동 종료 후 재기동한다 (CLAUDE.md 2-(3))
    comfy_start_command: str | None = None
    comfy_start_cwd: Path | None = None
    comfy_lora_dir: Path | None = None  # 학습된 LoRA를 복사해 넣을 ComfyUI의 models/loras

    @field_validator(
        "dataset_root",
        "wd14_model_path",
        "wd14_tags_csv",
        "kohya_python",
        "kohya_script_dir",
        "sd15_checkpoint_path",
        "comfy_start_command",
        "comfy_start_cwd",
        "comfy_lora_dir",
        mode="before",
    )
    @classmethod
    def _empty_string_as_unset(cls, value: object) -> object:
        """.env에 키만 있고 값이 비어 있으면(WD14_MODEL_PATH=) Path("")=='.'로

        잘못 해석되지 않도록 빈 문자열을 None(미설정)으로 취급한다.
        """
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


settings = Settings()


def update_env_file(values: dict[str, str], *, env_path: Path = Path(".env")) -> None:
    """.env 파일에서 지정된 키만 갱신한다 (없던 키는 파일 끝에 추가).

    GUI 설정 다이얼로그가 사용자에게 .env를 직접 열어 고치라고 하지 않고
    바로 저장할 수 있게 하기 위한 것. 값이 빈 문자열이면 미설정(키만 남기고
    값 비움)으로 저장한다 — Settings의 `_empty_string_as_unset`이 이를
    None으로 읽는다.
    """
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    remaining = dict(values)
    updated_lines = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped and not stripped.startswith("#") else None
        if key is not None and key in remaining:
            updated_lines.append(f"{key}={remaining.pop(key)}")
        else:
            updated_lines.append(line)
    for key, value in remaining.items():
        updated_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
