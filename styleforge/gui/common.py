"""세 탭이 공통으로 쓰는 작은 위젯·헬퍼. 로직은 없고 UI 배선만 한다."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from styleforge.config import settings
from styleforge.train.runner import comfy_is_up, start_comfyui


def list_lora_styles() -> list[str]:
    """`outputs/loras/*.safetensors`에서 스타일 이름 목록을 뽑는다 (apply/sweep의 --style 후보)."""
    loras_dir = Path("outputs/loras")
    if not loras_dir.is_dir():
        return []
    return sorted(p.stem for p in loras_dir.glob("*.safetensors"))


def list_workflows() -> list[str]:
    """`workflows/*.json`에서 워크플로우 이름 목록을 뽑는다 (apply의 --workflow 후보)."""
    workflows_dir = settings.workflows_dir
    if not workflows_dir.is_dir():
        return []
    return sorted(p.stem for p in workflows_dir.glob("*.json"))


def open_in_explorer(path: Path) -> None:
    """결과 폴더를 탐색기로 연다 (Windows 전용 — 이 프로젝트 자체가 Windows 네이티브 기준, CLAUDE.md 3장)."""
    if path.is_dir():
        os.startfile(path)  # noqa: S606


class BrowseEntry(ttk.Frame):
    """Entry + '찾아보기' 버튼. 파일/폴더 선택 다이얼로그를 연결한 재사용 위젯."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        mode: str = "file",  # "file" | "directory"
        filetypes: list[tuple[str, str]] | None = None,
        initial: str = "",
    ) -> None:
        super().__init__(parent)
        self.var = tk.StringVar(value=initial)
        entry = ttk.Entry(self, textvariable=self.var, width=52)
        entry.pack(side="left", fill="x", expand=True)
        ttk.Button(self, text="찾아보기", command=lambda: self._browse(mode, filetypes)).pack(
            side="left", padx=(4, 0)
        )

    def _browse(self, mode: str, filetypes: list[tuple[str, str]] | None) -> None:
        if mode == "directory":
            path = filedialog.askdirectory(initialdir=self.var.get() or None)
        else:
            path = filedialog.askopenfilename(
                initialdir=self.var.get() or None, filetypes=filetypes or [("모든 파일", "*.*")]
            )
        if path:
            self.var.set(path)

    def get(self) -> str:
        return self.var.get().strip()


class LogBox(ttk.Frame):
    """읽기 전용 스크롤 로그 창. 진행 메시지·경고·에러를 append로 쌓는다."""

    def __init__(self, parent: tk.Widget, *, height: int = 10) -> None:
        super().__init__(parent)
        self.text = tk.Text(self, height=height, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def append(self, message: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", message + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class ComfyStatusBar(ttk.Frame):
    """ComfyUI 실행 상태 표시 + 켜기 버튼.

    apply/sweep은 ComfyUI가 떠 있어야 동작하는데 이 프로젝트는 ComfyUI를
    별도 프로세스로 취급한다(CLAUDE.md 3장) — 그래서 새 프로세스 관리 로직을
    만들지 않고, train/runner.py가 VRAM 관리용으로 이미 갖고 있는
    comfy_is_up()/start_comfyui()를 그대로 재사용한다.
    """

    _POLL_INTERVAL_MS = 5000

    def __init__(self, parent: tk.Widget, *, on_busy_change: Callable[[bool], None]) -> None:
        super().__init__(parent)
        self._on_busy_change = on_busy_change
        self._up = False
        self._busy_elsewhere = False
        self._starting = False

        ttk.Label(self, text="ComfyUI:").pack(side="left")
        self.status_var = tk.StringVar(value="확인 중...")
        ttk.Label(self, textvariable=self.status_var, width=10).pack(side="left", padx=(4, 8))
        self.start_button = ttk.Button(self, text="켜기", command=self._on_start, state="disabled")
        self.start_button.pack(side="left")

        self._check_status()

    def notify_busy(self, busy: bool) -> None:
        """train/apply/sweep 중 하나라도 실행 중이면 켜기 버튼을 잠근다.

        특히 train은 시작하면서 ComfyUI를 직접 내렸다가 재기동하므로
        (CLAUDE.md 2-(3)), 그 사이에 사용자가 여기서 또 켜면 충돌한다.
        """
        self._busy_elsewhere = busy
        self._refresh_button()

    def _refresh_button(self) -> None:
        enabled = (not self._up) and (not self._busy_elsewhere) and (not self._starting)
        self.start_button.configure(state="normal" if enabled else "disabled")

    def _check_status(self) -> None:
        if self._starting:
            self.after(self._POLL_INTERVAL_MS, self._check_status)
            return

        result: list[bool] = []

        def work() -> None:
            result.append(comfy_is_up())

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self._wait_thread(thread, self._on_status_checked, result)

    def _on_status_checked(self, result: list[bool]) -> None:
        self._up = result[0] if result else False
        self.status_var.set("켜짐" if self._up else "꺼짐")
        self._refresh_button()
        self.after(self._POLL_INTERVAL_MS, self._check_status)

    def _on_start(self) -> None:
        self._starting = True
        self.status_var.set("여는 중...")
        self._refresh_button()
        self._on_busy_change(True)

        result: list[Exception | None] = []

        def work() -> None:
            try:
                start_comfyui()
            except Exception as exc:  # 백그라운드 스레드 예외를 폴링으로 전달하기 위해 잡는다
                result.append(exc)
            else:
                result.append(None)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self._wait_thread(thread, self._on_start_finished, result)

    def _on_start_finished(self, result: list[Exception | None]) -> None:
        self._starting = False
        self._on_busy_change(False)
        exc = result[0] if result else None
        if exc is not None:
            self._up = False
            self.status_var.set("꺼짐")
            messagebox.showerror("ComfyUI 기동 실패", str(exc))
        else:
            self._up = True
            self.status_var.set("켜짐")
        self._refresh_button()

    def _wait_thread(self, thread: threading.Thread, on_done: Callable[[list], None], result: list) -> None:
        """백그라운드 스레드가 끝날 때까지 메인 스레드를 막지 않고 폴링한다."""
        if thread.is_alive():
            self.after(100, lambda: self._wait_thread(thread, on_done, result))
            return
        on_done(result)
