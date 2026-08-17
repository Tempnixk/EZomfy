"""`train` 폼 — styleforge.train.runner.run_train()을 그대로 호출하는 얇은 와이어링."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from styleforge.config import settings
from styleforge.gui.common import BrowseEntry, LogBox
from styleforge.gui.worker import BackgroundTask, ConfirmRequest, DoneEvent, ErrorEvent, ProgressEvent
from styleforge.train.runner import TrainProgress, run_train

_CAPTION_MODE_LABELS = {
    "자동 감지 (라벨 있으면 external, 없으면 auto)": None,
    "external (제공된 라벨 사용)": "external",
    "auto (WD14 자동 태깅)": "auto",
}


class TrainTab(ttk.Frame):
    def __init__(self, parent: tk.Widget, *, on_busy_change: Callable[[bool], None]) -> None:
        super().__init__(parent, padding=10)
        self._on_busy_change = on_busy_change
        self._task = BackgroundTask()

        form = ttk.Frame(self)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        row = 0

        default_input = str(settings.dataset_root) if settings.dataset_root else ""
        ttk.Label(form, text="입력 폴더").grid(row=row, column=0, sticky="w", pady=3)
        self.input_entry = BrowseEntry(form, mode="directory", initial=default_input)
        self.input_entry.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        ttk.Label(
            form,
            text="(화목 필터를 쓰면 이 값은 무시되고 DATASET_ROOT 하위를 씁니다)",
            foreground="gray",
        ).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(form, text="이름 (트리거 워드)").grid(row=row, column=0, sticky="w", pady=3)
        self.name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.name_var).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        ttk.Label(form, text="화목 필터 (예: HJ)").grid(row=row, column=0, sticky="w", pady=3)
        self.filter_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.filter_var).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        ttk.Label(form, text="2차 필터 (key=value,key=value)").grid(row=row, column=0, sticky="w", pady=3)
        self.meta_filter_var = tk.StringVar()
        entry = ttk.Entry(form, textvariable=self.meta_filter_var)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        self.include_detail_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="상세묘사 이미지도 포함", variable=self.include_detail_var).grid(
            row=row, column=1, sticky="w", pady=3
        )
        row += 1

        ttk.Label(form, text="최대 장수").grid(row=row, column=0, sticky="w", pady=3)
        self.limit_var = tk.IntVar(value=40)
        ttk.Spinbox(form, from_=1, to=1000, textvariable=self.limit_var, width=10).grid(
            row=row, column=1, sticky="w", pady=3
        )
        row += 1

        ttk.Label(form, text="캡션 모드").grid(row=row, column=0, sticky="w", pady=3)
        self.caption_mode_var = tk.StringVar(value=next(iter(_CAPTION_MODE_LABELS)))
        ttk.Combobox(
            form,
            textvariable=self.caption_mode_var,
            values=list(_CAPTION_MODE_LABELS),
            state="readonly",
            width=40,
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        ttk.Label(form, text="학습 설정 파일").grid(row=row, column=0, sticky="w", pady=3)
        self.config_entry = BrowseEntry(
            form,
            mode="file",
            filetypes=[("TOML", "*.toml")],
            initial="configs/train_default.toml",
        )
        self.config_entry.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        self.run_button = ttk.Button(self, text="학습 시작", command=self._on_run)
        self.run_button.pack(anchor="w", pady=(8, 4))

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w")
        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=(2, 8))

        ttk.Label(self, text="로그").pack(anchor="w")
        self.log = LogBox(self, height=14)
        self.log.pack(fill="both", expand=True)

    def set_enabled(self, enabled: bool) -> None:
        self.run_button.configure(state="normal" if enabled else "disabled")

    def _on_run(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("입력 오류", "이름(트리거 워드)을 입력하세요.")
            return

        input_dir = self.input_entry.get()
        genre_code = self.filter_var.get().strip() or None
        if not genre_code and not input_dir:
            messagebox.showerror("입력 오류", "입력 폴더 또는 화목 필터 중 하나는 있어야 합니다.")
            return

        meta_filter_raw = self.meta_filter_var.get().strip()
        meta_filter = None
        if meta_filter_raw:
            try:
                meta_filter = dict(pair.split("=", 1) for pair in meta_filter_raw.split(","))
            except ValueError:
                messagebox.showerror("입력 오류", "2차 필터는 key=value,key=value 형식이어야 합니다.")
                return

        config_path = Path(self.config_entry.get() or "configs/train_default.toml")

        self.log.clear()
        self.progress.configure(mode="indeterminate")
        self.progress.start(50)
        self.status_var.set("준비 중...")
        self.run_button.configure(state="disabled")
        self._on_busy_change(True)

        on_progress = self._task.make_progress_callback()
        on_warning = self._task.make_confirm_callback()
        # tk.Variable.get()은 메인 스레드에서 미리 읽어둔다 - work()는 백그라운드
        # 스레드에서 돈다 (apply_tab.py에서와 동일한 이유).
        include_detail = self.include_detail_var.get()
        limit = self.limit_var.get()
        caption_mode = _CAPTION_MODE_LABELS[self.caption_mode_var.get()]

        def work() -> Path:
            return run_train(
                input_dir=Path(input_dir) if input_dir else Path("."),
                name=name,
                config_template=config_path,
                genre_code=genre_code,
                meta_filter=meta_filter,
                include_detail=include_detail,
                limit=limit,
                caption_mode=caption_mode,
                auto_confirm=False,
                on_progress=on_progress,
                on_warning=on_warning,
            )

        self._task.start(work)
        self.after(100, self._poll)

    def _poll(self) -> None:
        for event in self._task.poll():
            if isinstance(event, ProgressEvent):
                update: TrainProgress = event.payload
                if update.total_steps:
                    if str(self.progress["mode"]) != "determinate":
                        self.progress.stop()
                        self.progress.configure(mode="determinate", maximum=update.total_steps)
                    self.progress["value"] = update.step
                    loss_text = f", loss={update.loss:.4f}" if update.loss is not None else ""
                    self.status_var.set(f"{update.step}/{update.total_steps} 스텝{loss_text}")
            elif isinstance(event, ConfirmRequest):
                proceed = messagebox.askyesno(
                    "검증 경고", "\n".join(event.messages) + "\n\n무시하고 계속하시겠습니까?"
                )
                event.result[0] = proceed
                event.event.set()
                self.log.append("경고: " + " / ".join(event.messages))
            elif isinstance(event, DoneEvent):
                self._finish(success=True, message=f"완료: {event.result}")
            elif isinstance(event, ErrorEvent):
                self._finish(success=False, message=f"실패: {event.exc}")

        if str(self.run_button["state"]) == "disabled":
            self.after(100, self._poll)

    def _finish(self, *, success: bool, message: str) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.status_var.set(message)
        self.log.append(message)
        self.run_button.configure(state="normal")
        self._on_busy_change(False)
        if success:
            messagebox.showinfo("학습 완료", message)
        else:
            messagebox.showerror("학습 실패", message)
