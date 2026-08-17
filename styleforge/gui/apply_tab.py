"""`apply` 폼 — styleforge.apply.runner.run_apply()를 그대로 호출하는 얇은 와이어링."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from PIL import Image, ImageTk

from styleforge.apply.comfy_client import ProgressUpdate
from styleforge.apply.runner import run_apply
from styleforge.gui.common import BrowseEntry, LogBox, list_lora_styles, list_workflows, open_in_explorer
from styleforge.gui.worker import BackgroundTask, DoneEvent, ErrorEvent, ProgressEvent

_IMAGE_FILETYPES = [("이미지", "*.png *.jpg *.jpeg *.webp *.bmp"), ("모든 파일", "*.*")]
_PREVIEW_MAX = 320


class ApplyTab(ttk.Frame):
    def __init__(self, parent: tk.Widget, *, on_busy_change: Callable[[bool], None]) -> None:
        super().__init__(parent, padding=10)
        self._on_busy_change = on_busy_change
        self._task = BackgroundTask()
        self._output_dir: Path | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None  # 참조 유지 안 하면 GC로 사라짐

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

        ttk.Label(form, text="강도").grid(row=row, column=0, sticky="w", pady=3)
        strength_row = ttk.Frame(form)
        strength_row.grid(row=row, column=1, sticky="ew", pady=3)
        self.strength_var = tk.DoubleVar(value=0.6)
        self.strength_label_var = tk.StringVar(value="0.60")
        ttk.Scale(
            strength_row,
            from_=0.0,
            to=1.0,
            variable=self.strength_var,
            command=self._on_strength_change,
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(strength_row, textvariable=self.strength_label_var, width=5).pack(side="left", padx=(4, 0))
        row += 1

        ttk.Label(form, text="워크플로우").grid(row=row, column=0, sticky="w", pady=3)
        self.workflow_var = tk.StringVar(value="style_transfer_lineart")
        ttk.Combobox(
            form, textvariable=self.workflow_var, values=list_workflows(), state="readonly"
        ).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        ttk.Label(form, text="추가 프롬프트").grid(row=row, column=0, sticky="w", pady=3)
        self.prompt_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.prompt_var).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        ttk.Label(form, text="시드 (비우면 랜덤)").grid(row=row, column=0, sticky="w", pady=3)
        self.seed_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.seed_var, width=15).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        button_row = ttk.Frame(self)
        button_row.pack(anchor="w", pady=(8, 4))
        self.run_button = ttk.Button(button_row, text="변환 실행", command=self._on_run)
        self.run_button.pack(side="left")
        self.open_folder_button = ttk.Button(
            button_row, text="결과 폴더 열기", command=self._open_result_folder, state="disabled"
        )
        self.open_folder_button.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w")
        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=(2, 8))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        log_frame = ttk.Frame(body)
        log_frame.pack(side="left", fill="both", expand=True)
        ttk.Label(log_frame, text="로그").pack(anchor="w")
        self.log = LogBox(log_frame, height=12)
        self.log.pack(fill="both", expand=True)

        preview_frame = ttk.Frame(body, width=_PREVIEW_MAX)
        preview_frame.pack(side="left", padx=(8, 0))
        ttk.Label(preview_frame, text="결과 미리보기").pack(anchor="w")
        self.preview_label = ttk.Label(preview_frame, text="(아직 없음)", relief="groove", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

    def set_enabled(self, enabled: bool) -> None:
        self.run_button.configure(state="normal" if enabled else "disabled")

    def _refresh_styles(self) -> None:
        self.style_combo.configure(values=list_lora_styles())

    def _on_strength_change(self, value: str) -> None:
        self.strength_label_var.set(f"{float(value):.2f}")

    def _on_run(self) -> None:
        image_path = self.image_entry.get()
        style = self.style_var.get().strip()
        if not image_path or not style:
            messagebox.showerror("입력 오류", "입력 이미지와 스타일을 모두 지정하세요.")
            return

        seed_raw = self.seed_var.get().strip()
        seed: int | None = None
        if seed_raw:
            try:
                seed = int(seed_raw)
            except ValueError:
                messagebox.showerror("입력 오류", "시드는 정수여야 합니다.")
                return

        self.log.clear()
        self.preview_label.configure(image="", text="(생성 중...)")
        self.open_folder_button.configure(state="disabled")
        self.progress.configure(mode="determinate", value=0, maximum=100)
        self.status_var.set("적용 중...")
        self.run_button.configure(state="disabled")
        self._on_busy_change(True)

        on_progress = self._task.make_progress_callback()
        # tk.Variable.get()은 Tcl 인터프리터를 건드리므로 반드시 메인 스레드에서
        # 미리 읽어 일반 값으로 캡처해둔다 — work()는 백그라운드 스레드에서 돈다.
        strength = self.strength_var.get()
        workflow_name = self.workflow_var.get() or "style_transfer_lineart"
        prompt = self.prompt_var.get().strip() or None

        def work() -> Path:
            return run_apply(
                image=Path(image_path),
                style=style,
                strength=strength,
                workflow_name=workflow_name,
                prompt=prompt,
                seed=seed,
                on_progress=on_progress,
            )

        self._task.start(work)
        self.after(100, self._poll)

    def _poll(self) -> None:
        for event in self._task.poll():
            if isinstance(event, ProgressEvent):
                update: ProgressUpdate = event.payload
                if update.max:
                    self.progress.configure(maximum=update.max)
                    self.progress["value"] = update.value
                    self.status_var.set(f"{update.value}/{update.max}")
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
        self._show_preview(output_dir)
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
        messagebox.showerror("변환 실패", message)

    def _show_preview(self, output_dir: Path) -> None:
        images = sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.jpg"))
        if not images:
            self.preview_label.configure(image="", text="(결과 이미지 없음)")
            return
        img = Image.open(images[0])
        img.thumbnail((_PREVIEW_MAX, _PREVIEW_MAX))
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._preview_photo, text="")

    def _open_result_folder(self) -> None:
        if self._output_dir is not None:
            open_in_explorer(self._output_dir)
