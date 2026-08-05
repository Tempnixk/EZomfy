"""Typer 진입점. CLI 계층은 얇게 유지하고 로직은 각 모듈에 둔다 (CLAUDE.md 8장)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress

from styleforge.apply.comfy_client import ComfyClientError, ProgressUpdate
from styleforge.apply.runner import ApplyError, run_apply

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 cp949 대비

app = typer.Typer(help="StyleForge — 화풍 학습 · 적용 자동화 CLI 툴")


@app.callback()
def _main() -> None:
    """train/apply/sweep 명령이 늘어나도 서브커맨드 구조를 유지하기 위한 콜백.

    등록된 명령이 apply 하나뿐이면 Typer가 이를 최상위 명령으로 접어버려
    `styleforge apply ...` 형태(CLAUDE.md 1·7·12장)가 깨진다.
    """


@app.command()
def apply(
    image: Path = typer.Option(..., "--image", exists=True, help="입력 이미지"),
    style: str = typer.Option(..., "--style", help="적용할 LoRA 이름"),
    strength: float = typer.Option(0.6, "--strength", min=0.0, max=1.0, help="화풍 적용 강도 0.0~1.0"),
    workflow: str = typer.Option("style_transfer_lineart", "--workflow", help="사용할 워크플로우 이름"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="추가 프롬프트"),
    seed: Optional[int] = typer.Option(None, "--seed", help="시드 (기본: 랜덤)"),
) -> None:
    """이미지에 화풍을 적용한다."""
    try:
        with Progress() as progress:
            task = progress.add_task("적용 중", total=None)

            def on_progress(update: ProgressUpdate) -> None:
                if update.max:
                    progress.update(task, total=update.max, completed=update.value)

            output_dir = run_apply(
                image=image,
                style=style,
                strength=strength,
                workflow_name=workflow,
                prompt=prompt,
                seed=seed,
                on_progress=on_progress,
            )
    except (ApplyError, ComfyClientError) as exc:
        typer.echo(f"실패: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"완료: {output_dir}")


if __name__ == "__main__":
    app()
