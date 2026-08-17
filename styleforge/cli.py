"""Typer 진입점. CLI 계층은 얇게 유지하고 로직은 각 모듈에 둔다 (CLAUDE.md 8장)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress

from styleforge.apply.comfy_client import ComfyClientError, ProgressUpdate
from styleforge.apply.runner import ApplyError, run_apply
from styleforge.sweep.planner import DEFAULT_AXES
from styleforge.sweep.runner import SweepError, SweepProgress, run_sweep
from styleforge.train.runner import TrainProgress, TrainRunnerError, run_train

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 cp949 대비

app = typer.Typer(help="StyleForge — 화풍 학습 · 적용 자동화 CLI 툴")


@app.callback()
def _main() -> None:
    """등록된 명령이 하나뿐이면 Typer가 이를 최상위 명령으로 접어버려

    `styleforge apply ...` 형태(CLAUDE.md 1·7·12장)가 깨지므로, 명령이
    여러 개인 지금도 서브커맨드 구조가 유지되도록 콜백을 남겨둔다.
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


@app.command()
def train(
    input: Path = typer.Option(..., "--input", help="레퍼런스 이미지 폴더 (--filter 사용 시 데이터셋 루트)"),
    name: str = typer.Option(..., "--name", help="스타일 이름 = 트리거 워드 = LoRA 파일명"),
    config: Path = typer.Option(Path("configs/train_default.toml"), "--config", help="학습 설정 TOML"),
    filter: Optional[str] = typer.Option(None, "--filter", help="화목 코드 접두어 (대규모 데이터셋 사용 시)"),
    meta_filter: Optional[str] = typer.Option(
        None, "--meta-filter", help="라벨 필드 기준 2차 필터 (key=value,key=value)"
    ),
    include_detail: bool = typer.Option(False, "--include-detail", help="상세묘사 이미지도 포함"),
    limit: int = typer.Option(40, "--limit", help="학습에 사용할 최대 장수"),
    caption_mode: Optional[str] = typer.Option(None, "--caption-mode", help="auto | external"),
    yes: bool = typer.Option(False, "--yes", help="검증 경고 무시하고 진행"),
) -> None:
    """레퍼런스 이미지 폴더로 화풍 LoRA를 학습한다."""
    parsed_meta_filter = None
    if meta_filter:
        parsed_meta_filter = dict(pair.split("=", 1) for pair in meta_filter.split(","))

    try:
        with Progress() as progress:
            task = progress.add_task("학습 중", total=None)

            def on_progress(update: TrainProgress) -> None:
                if update.total_steps:
                    progress.update(task, total=update.total_steps, completed=update.step)

            lora_path = run_train(
                input_dir=input,
                name=name,
                config_template=config,
                genre_code=filter,
                meta_filter=parsed_meta_filter,
                include_detail=include_detail,
                limit=limit,
                caption_mode=caption_mode,  # type: ignore[arg-type]
                auto_confirm=yes,
                on_progress=on_progress,
            )
    except TrainRunnerError as exc:
        typer.echo(f"실패: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"완료: {lora_path}")


@app.command()
def sweep(
    image: Path = typer.Option(..., "--image", exists=True, help="입력 이미지"),
    style: str = typer.Option(..., "--style", help="LoRA 이름"),
    axes: str = typer.Option(DEFAULT_AXES, "--axes", help="탐색 축 (콤마 구분, 최대 2개: denoise/lora_weight/controlnet_strength)"),
    steps: int = typer.Option(4, "--steps", help="축당 분할 수"),
) -> None:
    """파라미터 조합을 탐색해 비교 그리드와 평가 리포트를 생성한다."""
    try:
        with Progress() as progress:
            task = progress.add_task("탐색 중", total=None)

            def on_progress(update: SweepProgress) -> None:
                completed = update.combo_index + (0.5 if update.stage == "evaluating" else 0.0)
                step_text = (
                    f" — {update.step}/{update.total_steps} 스텝"
                    if update.step is not None and update.total_steps
                    else ""
                )
                progress.update(
                    task,
                    total=update.total_combos,
                    completed=completed,
                    description=f"조합 {update.combo_index + 1}/{update.total_combos} ({update.stage}){step_text}",
                )

            output_dir = run_sweep(
                image=image, style=style, axes=axes, steps=steps, on_progress=on_progress
            )
    except SweepError as exc:
        typer.echo(f"실패: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"완료: {output_dir}")


if __name__ == "__main__":
    app()
