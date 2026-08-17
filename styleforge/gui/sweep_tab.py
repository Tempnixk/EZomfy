"""`sweep` 폼 — styleforge.sweep.runner.run_sweep()을 그대로 호출하는 얇은 와이어링."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from PIL import Image, ImageTk

from styleforge.gui.common import BrowseEntry, LogBox, list_lora_styles, open_in_explorer
from styleforge.gui.worker import BackgroundTask, DoneEvent, ErrorEvent, ProgressEvent
from styleforge.sweep.runner import SweepProgress, run_sweep

_IMAGE_FILETYPES = [("이미지", "*.png *.jpg *.jpeg *.webp *.bmp"), ("모든 파일", "*.*")]
_AXIS_OPTIONS = ["denoise", "lora_weight", "controlnet_strength"]
_NONE_OPTION = "(사용 안 함)"
_PREVIEW_MAX = 360


def _extract_recommendation(report_path: Path) -> str:
    """report.md의 '## 권장 조합' 절 바로 다음 줄(백틱 조합 설명)을 뽑아낸다."""
    if not report_path.is_file():
        return ""
    lines = report_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "## 권장 조합":
            for following in lines[i + 1 :]:
                if following.strip():
                    return following.strip()
    return ""


class SweepTab(ttk.Frame):
    def __init__(self, parent: tk.Widget, *, on_busy_change: Callable[[bool], None]) -> None:
        super().__init__(parent, padding=10)
        self._on_busy_change = on_busy_change
        self._task = BackgroundTask()
        self._output_dir: Path | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None

        form = ttk.Frame(self)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(form, text="입력 이미지").grid(row=row, column=0, sticky="w", pady=3)
        self.image_entry = BrowseEntry(form, mode="file", filetypes=_IMAGE_FILETYPES)
        self.image_entry.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        ttk.Label(form, text="스타일 (LoRA)").grid(row=row, column=0, sticky="w", pady=3)
        style_row = ttk.Frame(form)
        style_row.grid(row=row, column=1, sticky="ew", pady=3)
        self.style_var = tk.StringVar()
        self.style_combo = ttk.Combobox(style_row, textvariable=self.style_var, values=list_lora_styles())
        self.style_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(style_row, text="새로고침", command=self._refresh_styles).pack(side="left", padx=(4, 0))
        row += 1

        ttk.Label(form, text="탐색 축 1").grid(row=row, column=0, sticky="w", pady=3)
        self.axis1_var = tk.StringVar(value="denoise")
        ttk.Combobox(
            form, textvariable=self.axis1_var, values=_AXIS_OPTIONS, state="readonly", width=20
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        ttk.Label(form, text="탐색 축 2").grid(row=row, column=0, sticky="w", pady=3)
        self.axis2_var = tk.StringVar(value="lora_weight")
        ttk.Combobox(
            form,
            textvariable=self.axis2_var,
            values=[_NONE_OPTION, *_AXIS_OPTIONS],
            state="readonly",
            width=20,
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        ttk.Label(form, text="축당 분할 수").grid(row=row, column=0, sticky="w", pady=3)
        self.steps_var = tk.IntVar(value=4)
        ttk.Spinbox(form, from_=2, to=10, textvariable=self.steps_var, width=10).grid(
            row=row, column=1, sticky="w", pady=3
        )
        row += 1

        button_row = ttk.Frame(self)
        button_row.pack(anchor="w", pady=(8, 4))
        self.run_button = ttk.Button(button_row, text="탐색 실행", command=self._on_run)
        self.run_button.pack(side="left")
        self.open_folder_button = ttk.Button(
            button_row, text="결과 폴더 열기", command=self._open_result_folder, state="disabled"
        )
        self.open_folder_button.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w")
        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=(2, 4))
        self.recommendation_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.recommendation_var, foreground="darkgreen").pack(
            anchor="w", pady=(0, 8)
        )

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        log_frame = ttk.Frame(body)
        log_frame.pack(side="left", fill="both", expand=True)
        ttk.Label(log_frame, text="로그").pack(anchor="w")
        self.log = LogBox(log_frame, height=12)
        self.log.pack(fill="both", expand=True)

        preview_frame = ttk.Frame(body, width=_PREVIEW_MAX)
        preview_frame.pack(side="left", padx=(8, 0))
        ttk.Label(preview_frame, text="비교 그리드 미리보기").pack(anchor="w")
        self.preview_label = ttk.Label(preview_frame, text="(아직 없음)", relief="groove", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

    def set_enabled(self, enabled: bool) -> None:
        self.run_button.configure(state="normal" if enabled else "disabled")

    def _refresh_styles(self) -> None:
        self.style_combo.configure(values=list_lora_styles())

    def _build_axes(self) -> str:
        axis1 = self.axis1_var.get()
        axis2 = self.axis2_var.get()
        if axis2 == _NONE_OPTION or axis2 == axis1:
            return axis1
        return f"{axis1},{axis2}"

    def _on_run(self) -> None:
        image_path = self.image_entry.get()
        style = self.style_var.get().strip()
        if not image_path or not style:
            messagebox.showerror("입력 오류", "입력 이미지와 스타일을 모두 지정하세요.")
            return

        self.log.clear()
        self.recommendation_var.set("")
        self.preview_label.configure(image="", text="(생성 중...)")
        self.open_folder_button.configure(state="disabled")
        self.progress.configure(mode="determinate", value=0, maximum=100)
        self.status_var.set("탐색 중...")
        self.run_button.configure(state="disabled")
        self._on_busy_change(True)

        on_progress = self._task.make_progress_callback()
        axes = self._build_axes()
        steps = self.steps_var.get()

        def work() -> Path:
            return run_sweep(
                image=Path(image_path), style=style, axes=axes, steps=steps, on_progress=on_progress
            )

        self._task.start(work)
        self.after(100, self._poll)

    def _poll(self) -> None:
        for event in self._task.poll():
            if isinstance(event, ProgressEvent):
                update: SweepProgress = event.payload
                completed = update.combo_index + (0.5 if update.stage == "evaluating" else 0.0)
                self.progress.configure(maximum=update.total_combos)
                self.progress["value"] = completed
                step_text = (
                    f" — {update.step}/{update.total_steps} 스텝"
                    if update.step is not None and update.total_steps
                    else ""
                )
                self.status_var.set(
                    f"조합 {update.combo_index + 1}/{update.total_combos} ({update.stage}){step_text}"
                )
            elif isinstance(event, DoneEvent):
                self._finish_success(event.result)
            elif isinstance(event, ErrorEvent):
                self._finish_error(event.exc)

        if str(self.run_button["state"]) == "disabled":
            self.after(100, self._poll)

    def _finish_success(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        message = f"완료: {output_dir}"
        self.status_var.set(message)
        self.log.append(message)
        recommendation = _extract_recommendation(output_dir / "report.md")
        if recommendation:
            self.recommendation_var.set(f"권장 조합: {recommendation}")
        self._show_preview(output_dir / "grid.png")
        self.open_folder_button.configure(state="normal")
        self.run_button.configure(state="normal")
        self._on_busy_change(False)

    def _finish_error(self, exc: Exception) -> None:
        message = f"실패: {exc}"
        self.status_var.set(message)
        self.log.append(message)
        self.preview_label.configure(image="", text="(실패)")
        self.run_button.configure(state="normal")
        self._on_busy_change(False)
        messagebox.showerror("탐색 실패", message)

    def _show_preview(self, grid_path: Path) -> None:
        if not grid_path.is_file():
            self.preview_label.configure(image="", text="(그리드 이미지 없음)")
            return
        img = Image.open(grid_path)
        img.thumbnail((_PREVIEW_MAX, _PREVIEW_MAX))
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._preview_photo, text="")

    def _open_result_folder(self) -> None:
        if self._output_dir is not None:
            open_in_explorer(self._output_dir)
