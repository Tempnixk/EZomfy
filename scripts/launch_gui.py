"""배포용 StyleForge.exe의 진입점 소스.

`pyinstaller`로 이 파일 하나만 빌드한다 — torch/onnxruntime 등 무거운
의존성은 이미 대상 컴퓨터의 `.venv`에 설치돼 있다고 가정하고(README 참고),
이 exe는 그 venv의 `pythonw.exe`로 GUI를 대신 실행해주는 얇은 런처일
뿐이다. 그래서 빌드 결과물이 작고 빠르며, 의존성을 다시 번들링하지 않는다.

exe는 프로젝트 루트(`.venv`, `.env`가 있는 위치)에 두고 실행해야 한다 —
자기 자신의 위치를 기준으로 `.venv`를 찾는다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _show_error(message: str) -> None:
    import ctypes

    ctypes.windll.user32.MessageBoxW(0, message, "StyleForge", 0x10)  # MB_ICONERROR


def main() -> None:
    project_root = _project_root()
    pythonw = project_root / ".venv" / "Scripts" / "pythonw.exe"

    if not pythonw.is_file():
        _show_error(
            f"이 컴퓨터에 StyleForge 가상환경이 없습니다:\n{pythonw}\n\n"
            "StyleForge.exe는 .venv가 설치된 프로젝트 폴더 안에 두고 실행해야 합니다.\n"
            "README의 설치 안내를 먼저 따라주세요."
        )
        return

    subprocess.Popen([str(pythonw), "-m", "styleforge.gui.main"], cwd=str(project_root))


if __name__ == "__main__":
    main()
