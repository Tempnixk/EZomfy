"""train 명령 실행 흐름 — VRAM 관리(ComfyUI 중지/재기동) + kohya 서브프로세스 실행.

학습과 ComfyUI는 동시에 실행할 수 없으므로(CLAUDE.md 2-(3)/3장), 학습
시작 전 ComfyUI를 종료하고 학습이 끝나면(성공하든 실패하든) 원래
떠있었을 때만 재기동한다.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import psutil
import requests

from styleforge.config import settings
from styleforge.dataset.caption import CaptionMode, CaptionTarget, build_captions
from styleforge.dataset.preprocess import prepare_dataset
from styleforge.dataset.scan import scan_folder
from styleforge.dataset.subset import select_subset, write_manifest
from styleforge.train.config_builder import ConfigBuildError, write_configs

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class TrainRunnerError(RuntimeError):
    """학습 실행 중 발생한 오류."""


@dataclass
class TrainProgress:
    step: int
    total_steps: int | None
    loss: float | None


# --- VRAM 관리: ComfyUI 중지/재기동 -----------------------------------------


def _comfy_is_up() -> bool:
    try:
        response = requests.get(f"{settings.comfy_url}/system_stats", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def stop_comfyui(*, timeout: float = 15.0) -> bool:
    """포트를 점유한 ComfyUI 프로세스를 종료한다. 원래 떠있었으면 True를 반환한다."""
    if not _comfy_is_up():
        return False

    port = urlparse(settings.comfy_url).port
    if port is None:
        raise TrainRunnerError(f"COMFY_URL에서 포트를 읽을 수 없습니다: {settings.comfy_url}")

    pid = None
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            pid = conn.pid
            break

    if pid is None:
        raise TrainRunnerError(f"포트 {port}를 점유한 프로세스를 찾지 못했습니다 (권한 문제일 수 있음)")

    proc = psutil.Process(pid)
    print(f"[train] ComfyUI 종료 중 (PID {pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)

    return True


def start_comfyui(*, wait: bool = True, timeout: float = 60.0) -> None:
    """설정된 명령으로 ComfyUI를 재기동한다."""
    if settings.comfy_start_command is None:
        raise TrainRunnerError("COMFY_START_COMMAND가 설정되지 않아 ComfyUI를 자동으로 재기동할 수 없습니다")

    print("[train] ComfyUI 재기동 중...")
    # shell=True로 실행한다: COMFY_START_COMMAND가 `.\python_embeded\...`처럼
    # 상대경로를 쓰는 경우, argv 리스트로 넘기면 Windows CreateProcess가
    # 실행 파일을 새 cwd가 아니라 "호출자"의 cwd 기준으로 찾아 실패한다.
    # 셸이 먼저 지정된 cwd로 들어간 뒤 실행하므로 상대경로가 그대로 동작한다.
    subprocess.Popen(
        settings.comfy_start_command,
        cwd=settings.comfy_start_cwd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not wait:
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _comfy_is_up():
            return
        time.sleep(1.0)
    raise TrainRunnerError("ComfyUI 재기동 후 응답이 없습니다 (timeout)")


# --- kohya 서브프로세스 -------------------------------------------------------

_STEP_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)")
_LOSS_RE = re.compile(r"loss[=:]\s*([\d.]+)")


def _parse_progress_line(line: str) -> TrainProgress | None:
    step_match = _STEP_RE.search(line)
    if step_match is None:
        return None

    loss_match = _LOSS_RE.search(line)
    return TrainProgress(
        step=int(step_match.group(2)),
        total_steps=int(step_match.group(3)),
        loss=float(loss_match.group(1)) if loss_match else None,
    )


def run_kohya(
    dataset_config: Path,
    train_config: Path,
    *,
    on_progress: Callable[[TrainProgress], None] | None = None,
    log_path: Path | None = None,
) -> None:
    """kohya의 train_network.py를 서브프로세스로 실행한다."""
    if settings.kohya_python is None or settings.kohya_script_dir is None:
        raise TrainRunnerError("KOHYA_PYTHON / KOHYA_SCRIPT_DIR가 설정되지 않았습니다 (.env 확인)")
    if not settings.kohya_python.is_file():
        raise TrainRunnerError(f"kohya python not found: {settings.kohya_python}")

    script = settings.kohya_script_dir / "train_network.py"
    if not script.is_file():
        raise TrainRunnerError(f"kohya train_network.py not found: {script}")

    # kohya 공식 문서(docs/train_network.md)가 권장하는 실행 방식이 `python
    # train_network.py`가 아니라 `accelerate launch ... train_network.py`다.
    # 단일 GPU라 분산 학습이 필요없어 보이지만, mixed_precision 등 Accelerator
    # 초기화를 accelerate CLI가 맡는 경로가 표준 경로이므로 그대로 따른다.
    cmd = [
        str(settings.kohya_python),
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        "1",
        str(script),
        "--dataset_config",
        str(dataset_config.resolve()),
        "--config_file",
        str(train_config.resolve()),
    ]

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    # buffering=1(line buffering) — 기본 블록 버퍼링이면 학습이 오래 걸리는
    # 중에 죽거나 강제 종료됐을 때 로그가 디스크에 전혀 안 남아 원인 파악이
    # 안 된다. 진행 중에도 실시간으로 tail 가능해야 디버깅이 된다.
    log_file = log_path.open("w", encoding="utf-8", buffering=1) if log_path else None

    # kohya 코드베이스에는 UTF-8이 아닌 문자(예: 장음부호 ー)가 섞여 있어, 자식
    # 프로세스가 Windows 로캘 코드페이지(cp949 등)로 stdout을 인코딩하면 그
    # 문자에서 UnicodeEncodeError로 죽는다. PYTHONUTF8=1로 자식의 I/O 자체를
    # UTF-8로 강제한다 — 부모 쪽 encoding="utf-8"은 파이프에서 온 바이트를
    # 읽는 쪽만 담당하므로 이것만으로는 자식의 인코딩 실패를 막지 못한다.
    child_env = {**os.environ, "PYTHONUTF8": "1"}

    try:
        process = subprocess.Popen(
            cmd,
            cwd=settings.kohya_script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            if log_file:
                log_file.write(line)
            progress = _parse_progress_line(line)
            if progress is not None and on_progress is not None:
                on_progress(progress)

        return_code = process.wait()
    finally:
        if log_file:
            log_file.close()

    if return_code != 0:
        raise TrainRunnerError(f"kohya 학습이 실패했습니다 (exit code {return_code}). 로그: {log_path}")


# --- train 명령 오케스트레이션 -------------------------------------------------


def _collect_targets(input_dir: Path, limit: int) -> list[CaptionTarget]:
    """--filter 없이 임의 폴더를 --input으로 받은 경우. 같은 폴더의 동명 .json을 라벨로 취급한다."""
    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)[:limit]
    targets: list[CaptionTarget] = []
    for image_path in image_paths:
        label_path = image_path.with_suffix(".json")
        targets.append(CaptionTarget(image_path=image_path, label_path=label_path if label_path.is_file() else None))
    return targets


def run_train(
    *,
    input_dir: Path,
    name: str,
    config_template: Path = Path("configs/train_default.toml"),
    genre_code: str | None = None,
    meta_filter: dict[str, str] | None = None,
    include_detail: bool = False,
    limit: int = 40,
    caption_mode: CaptionMode | None = None,
    auto_confirm: bool = False,
    on_progress: Callable[[TrainProgress], None] | None = None,
) -> Path:
    """train 명령의 실행 흐름. 완성된 LoRA 파일 경로를 반환한다."""
    if caption_mode is not None and caption_mode not in ("auto", "external"):
        raise TrainRunnerError(f"알 수 없는 --caption-mode 값: {caption_mode}")

    started_at = time.monotonic()

    # 1. 서브셋 선별(--filter 지정 시) 또는 --input 폴더를 그대로 사용
    if genre_code:
        items = select_subset(genre_code, meta_filter=meta_filter, include_detail=include_detail, limit=limit)
        if not items:
            raise TrainRunnerError("선별 조건에 맞는 이미지가 없습니다")
        write_manifest(name, items, genre_code=genre_code, meta_filter=meta_filter)
        targets = [
            CaptionTarget(image_path=Path(it.image_path), label_path=Path(it.label_path)) for it in items
        ]
    else:
        if not input_dir.is_dir():
            raise TrainRunnerError(f"입력 폴더가 없습니다: {input_dir}")
        targets = _collect_targets(input_dir, limit)

    if not targets:
        raise TrainRunnerError("학습에 쓸 이미지를 찾지 못했습니다")

    # 2. 캡셔닝 + 전처리 (kohya가 바로 읽을 수 있는 폴더로 정규화)
    captions = build_captions(targets, name, mode=caption_mode)
    prep_result = prepare_dataset(name, captions)
    if prep_result.image_count == 0:
        raise TrainRunnerError("전처리 후 학습에 쓸 이미지가 없습니다")

    # 3. 품질 검증 → 경고 시 사용자 확인 (전처리 결과 폴더 기준 — 원본 폴더가
    #    --include-detail 등으로 여러 소스에 걸쳐 있어도 단일 폴더로 정리된 뒤라 검증이 일관적이다)
    report = scan_folder(prep_result.output_dir)
    if report.has_warnings:
        for warning in report.warnings:
            print(f"[train] 경고: {warning.message}")
        if not auto_confirm:
            answer = input("[train] 위 경고를 무시하고 계속하시겠습니까? (y/N): ")
            if answer.strip().lower() != "y":
                raise TrainRunnerError("사용자가 학습을 취소했습니다")

    # 4. kohya 설정 생성
    try:
        dataset_config, train_config = write_configs(
            name, prep_result.output_dir, template_path=config_template
        )
    except ConfigBuildError as exc:
        raise TrainRunnerError(str(exc)) from exc

    # 5. ComfyUI 종료 → 학습 실행 → (원래 떠있었다면) 재기동
    was_running = stop_comfyui()
    log_path = settings.data_dir / "prepared" / name / "train.log"
    try:
        run_kohya(dataset_config, train_config, on_progress=on_progress, log_path=log_path)
    finally:
        if was_running:
            start_comfyui()

    # 6. LoRA를 outputs/loras/와 ComfyUI LoRA 폴더에 배치
    lora_filename = f"{name}.safetensors"
    generated_lora = Path("outputs/loras") / lora_filename
    if not generated_lora.is_file():
        raise TrainRunnerError(f"학습은 끝났지만 LoRA 파일을 찾지 못했습니다: {generated_lora}")

    if settings.comfy_lora_dir is not None:
        settings.comfy_lora_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_lora, settings.comfy_lora_dir / lora_filename)

    elapsed = time.monotonic() - started_at

    meta = {
        "name": name,
        "image_count": prep_result.image_count,
        "genre_code": genre_code,
        "meta_filter": meta_filter or {},
        "config_template": str(config_template),
        "elapsed_seconds": round(elapsed, 2),
        "lora_path": str(generated_lora),
    }
    (Path("outputs/loras") / f"{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return generated_lora
