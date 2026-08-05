"""전역 설정 — .env를 읽어 프로젝트 전체에서 쓰는 값을 노출한다.

경로·주소를 코드에 하드코딩하지 않는다는 원칙(CLAUDE.md 3장)에 따라
새 설정값은 여기에 필드로 추가하고 .env.example도 함께 갱신한다.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    comfy_url: str = "http://127.0.0.1:8188"
    workflows_dir: Path = Path("workflows")


settings = Settings()
